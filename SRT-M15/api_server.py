"""Authenticated OpenAI Chat Completions-style facade for experimental SRT-M15.

- Non-streaming one-turn local requests keep the existing dispatcher behavior.
- Multi-turn local requests explicitly serialize history into the toy model's
  tiny context window. This does NOT give the network trained chat memory.
- stream=true performs actual next-token inference for local language weights;
  DeepSeek API stream=true forwards actual upstream deltas.
- DeepSeek Web does NOT support genuine server-side streaming or multi-turn.
"""
from contextlib import asynccontextmanager
from hmac import compare_digest
import json
import os
import threading
import time
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from srt_m15.chat import SRTChatEngine
from srt_m15.api_streaming import (
    as_prompt, from_chat_engine, from_staged_checkpoint,
    iter_local_tokens, prepare_local, validate_messages,
)


LOCAL_MODEL = "srt-m15"
STAGED_MODEL = "srt-m15-staged"
API_MODEL = "srt-m15-deepseek-api"
WEB_MODEL = "srt-m15-deepseek-web"
TRUE_VALUES = {"1", "true", "yes", "on"}


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=400)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(default=LOCAL_MODEL, min_length=1)
    messages: list[ChatMessage] = Field(min_length=1, max_length=32)
    stream: bool = False
    max_tokens: int = Field(default=80, ge=1, le=80)
    temperature: float | None = Field(default=None, ge=0, le=2)


def _frame(payload):
    """SSE data frame; JSON encoded to avoid user text injecting SSE protocol."""
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _error(code, explanation):
    raise HTTPException(status_code=code, detail=explanation)


def create_app(*, api_key=None, engine=None, model_routes=None, staged_checkpoint=None):
    """Create local API; never activate external models or staged weights by accident.

    - SRT_STAGED_CHECKPOINT is a *trusted server-side filesystem path*, not a
      request parameter. Restart the process after replacing the checkpoint.
    - SRT_API_ENABLE_DEEPSEEK_API / WEB are separate explicit opt-ins.
    """
    configured_key = api_key if api_key is not None else os.getenv("SRT_API_KEY", "")
    staged_path = (staged_checkpoint if staged_checkpoint is not None
                   else os.getenv("SRT_STAGED_CHECKPOINT", "")).strip()
    routes = {LOCAL_MODEL: "local"}
    if staged_path:
        routes[STAGED_MODEL] = "staged"
    if model_routes is not None:
        for name, backend in model_routes.items():
            if name not in {API_MODEL, WEB_MODEL} or backend not in {"api", "web"}:
                raise ValueError("Unrecognized model route")
            routes[name] = backend
    else:
        if os.getenv("SRT_API_ENABLE_DEEPSEEK_API", "").lower() in TRUE_VALUES:
            routes[API_MODEL] = "api"
        if os.getenv("SRT_API_ENABLE_DEEPSEEK_WEB", "").lower() in TRUE_VALUES:
            routes[WEB_MODEL] = "web"

    current_engine = engine if engine is not None else SRTChatEngine(
        root=os.environ.get("SRT_MODEL_ROOT") or ".", backend="local",
    )
    # Plain Lock (not thread-owned RLock): StreamingResponse's sync
    # generator may resume on different AnyIO worker threads between yields.
    inference_lock = threading.Lock()
    decoder_cache = {"staged": None}

    @asynccontextmanager
    async def lifespan(_app):
        if not configured_key or len(configured_key) < 16:
            raise RuntimeError("请先配置至少 16 字符的 SRT_API_KEY；不要提交密钥到 GitHub")
        try:
            yield
        finally:
            current_engine.close()

    app = FastAPI(
        title="SRT-M15 API",
        version="0.2.0",
        description="Small bounded, stateless Chat Completions subset; local text streaming.",
        lifespan=lifespan,
        docs_url=None, redoc_url=None, openapi_url=None,
    )
    cors_origins = [
        origin.strip() for origin in
        os.getenv("SRT_API_CORS_ORIGINS", "").split(",") if origin.strip()
    ]
    if "*" in cors_origins:
        raise ValueError("SRT_API_CORS_ORIGINS cannot contain a wildcard origin")
    if cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=cors_origins,
            allow_credentials=False, allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )
    app.state.engine = current_engine
    app.state.model_routes = routes

    def require_auth(authorization: str | None = Header(default=None)):
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer API key required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = authorization[len(prefix):]
        if not configured_key or not compare_digest(token, configured_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/health", dependencies=[Depends(require_auth)])
    def health():
        metadata = current_engine.status()
        from pathlib import Path
        return {
            "service": "SRT-M15", "api": "running",
            "local_math_checkpoint": bool(metadata.get("math_checkpoint")),
            "local_language_checkpoint": bool(metadata.get("language_checkpoint")),
            "staged_checkpoint": bool(staged_path and Path(staged_path).is_file()),
            "model_ids": list(routes),
        }

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    def list_models():
        return {
            "object": "list",
            "data": [{
                "id": name, "object": "model", "created": 0,
                "owned_by": "local" if backend in {"local", "staged"} else "external",
            } for name, backend in routes.items()],
        }

    def acquire_decoder(backend):
        # Caller must hold inference_lock while loading or running a decoder.
        try:
            if backend == "staged":
                if decoder_cache["staged"] is None:
                    decoder_cache["staged"] = from_staged_checkpoint(staged_path)
                return decoder_cache["staged"]
            return from_chat_engine(current_engine)
        except FileNotFoundError:
            _error(503, "Local language model weights are unavailable")
        except (ValueError, KeyError, RuntimeError):
            _error(503, "Unable to load local model; review checkpoint compatibility")

    @app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
    def chat(request: ChatCompletionRequest):
        if request.model not in routes:
            _error(404, "Unknown or disabled model; see GET /v1/models")
        if request.temperature is not None and request.temperature != 0:
            _error(400, "Sampling temperature is not supported; omit it or use 0")
        try:
            has_context = validate_messages(request.messages)
        except ValueError as exc:
            _error(400, str(exc))
        backend = routes[request.model]
        if backend == "web" and (has_context or request.stream):
            _error(400, "DeepSeek Web does not support genuine streaming or chat history")
        if backend == "local" and not has_context and not request.stream:
            # Exactly preserve the existing local one-turn dispatch (including
            # math, error messages, and model identity answers).
            try:
                with inference_lock:
                    reply = current_engine.reply(
                        request.messages[0].content.strip(),
                        new_tokens=request.max_tokens, backend="local",
                    )
            except Exception as exc:
                _error(503, f"Inference failed ({type(exc).__name__}); check server logs")
            if reply.mode == "notice":
                _error(503, reply.text)
            if reply.mode in {"limit", "vocab"}:
                _error(422, reply.text)
            if reply.mode == "error":
                _error(503, "Local model request failed")
            answer, provider, mode = reply.text, reply.provider, reply.mode
            if not answer.strip():
                _error(503, "Model returned empty output")
        elif backend in {"local", "staged"}:
            prompt, packed = as_prompt(
                request.messages, style="staged" if backend == "staged" else "legacy"
            )
            # Preflight tokenization *before* returning HTTP 200 to SSE clients.
            with inference_lock:
                decoder = acquire_decoder(backend)
                try:
                    prepare_local(decoder, prompt, has_context=packed)
                except ValueError as exc:
                    _error(422, str(exc))
            provider, mode = decoder.provider, "text"
            if not request.stream:
                try:
                    with inference_lock:
                        answer = "".join(iter_local_tokens(
                            decoder, prompt, max_tokens=request.max_tokens,
                            has_context=packed,
                        ))
                except Exception as exc:
                    _error(503, f"Local inference failed ({type(exc).__name__})")
                if not answer.strip():
                    _error(503, "Model returned empty output")
            else:
                iterator = lambda: iter_local_tokens(
                    decoder, prompt, max_tokens=request.max_tokens,
                    has_context=packed,
                )
        elif backend == "api":
            from srt_m15.deepseek_api import DeepSeekAPIClient, DeepSeekAPIError
            from urllib.error import URLError
            client = DeepSeekAPIClient()
            if not client.api_key:
                _error(503, "DeepSeek API has not been configured")
            items = [m.model_dump() for m in request.messages]
            provider, mode = "DeepSeek API", "external"
            if not request.stream:
                try:
                    answer = client.chat(items, max_tokens=request.max_tokens)
                except DeepSeekAPIError:
                    _error(502, "External model request failed")
            else:
                iterator = lambda: client.iter_chat(items, max_tokens=request.max_tokens)
        else:
            # DeepSeek Web is explicitly opt-in and supported only for one turn.
            try:
                with inference_lock:
                    reply = current_engine.reply(
                        request.messages[-1].content.strip(),
                        new_tokens=request.max_tokens, backend="web",
                    )
            except Exception as exc:
                _error(503, f"Inference failed ({type(exc).__name__}); check server logs")
            if reply.mode == "error":
                _error(502, "External model request failed")
            answer, provider, mode = reply.text, reply.provider, reply.mode

        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created_at = int(time.time())
        if request.stream:
            def stream_chunks():
                def chunk(delta, finish=None):
                    return _frame({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_at, "model": request.model,
                        "choices": [{"index": 0, "delta": delta,
                                     "finish_reason": finish}],
                    })
                yield chunk({"role": "assistant"})
                try:
                    # Keep a single model at a time on a small-memory host.
                    with inference_lock:
                        for part in iterator():
                            if part:
                                yield chunk({"content": part})
                    yield chunk({}, "stop")
                except Exception as exc:
                    # After SSE starts, HTTP status is already 200. Send an
                    # explicit error event, never a bogus successful completion.
                    yield "event: error\n" + _frame({
                        "error": "stream failed",
                        "type": type(exc).__name__,
                    })
                finally:
                    yield "data: [DONE]\n\n"
            return StreamingResponse(
                stream_chunks(), media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                    "X-Content-Type-Options": "nosniff",
                },
            )

        if not answer.strip():
            _error(503, "Model returned empty output")
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_at,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }],
            "srt_provider": provider,
            "srt_mode": mode,
        }

    return app


app = create_app()
