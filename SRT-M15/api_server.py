"""OpenAI Chat Completions-style HTTP facade for SRT-M15.

A deliberately small, single-turn, non-streaming subset of the API.
The "srt-m15" model ALWAYS calls the local trained SRT-M15 weights.
Remote DeepSeek backends are separately named and explicitly opt-in.
"""
from contextlib import asynccontextmanager
from hmac import compare_digest
import os
import threading
import time
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from srt_m15.chat import SRTChatEngine


LOCAL_MODEL = "srt-m15"
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


def create_app(*, api_key=None, engine=None, model_routes=None):
    """Build an app suitable for Uvicorn and for isolated TestClient tests.

    Env switches SRT_API_ENABLE_DEEPSEEK_API/WEB opt external models in;
    neither switch changes the meaning of the 'srt-m15' local model ID.
    """
    configured_key = api_key if api_key is not None else os.getenv("SRT_API_KEY", "")
    routes = {LOCAL_MODEL: "local"}
    if model_routes is not None:
        # Explicitly injected routes are helpful for integration testing.
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
        root=os.environ.get("SRT_MODEL_ROOT") or ".",
        backend="local",
    )
    inference_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app):
        if not configured_key or len(configured_key) < 16:
            raise RuntimeError(
                "请先配置至少 16 字符的 SRT_API_KEY；"
                "不要将密钥写入 GitHub。"
            )
        try:
            yield
        finally:
            current_engine.close()

    app = FastAPI(
        title="SRT-M15 Local API",
        version="0.1.0",
        description="Small, single-turn OpenAI-compatible chat subset.",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
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
        received = authorization[len(prefix):]
        if not configured_key or not compare_digest(received, configured_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/health", dependencies=[Depends(require_auth)])
    def health():
        # Do not describe the service as 'trained/ready' when weights are missing.
        metadata = current_engine.status()
        return {
            "service": "SRT-M15",
            "api": "running",
            "local_math_checkpoint": bool(metadata.get("math_checkpoint")),
            "local_language_checkpoint": bool(metadata.get("language_checkpoint")),
            "model_ids": list(routes),
        }

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    def list_models():
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model",
                 "created": 0, "owned_by": "local" if backend == "local" else "external"}
                for name, backend in routes.items()
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
    def chat(request: ChatCompletionRequest):
        if request.model not in routes:
            raise HTTPException(
                status_code=404,
                detail="Unknown or disabled model; see GET /v1/models",
            )
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming is not implemented (use stream=false)",
            )
        if request.temperature is not None and request.temperature != 0:
            raise HTTPException(
                status_code=400,
                detail="Sampling temperature is not supported; omit it or use 0",
            )
        if len(request.messages) != 1 or request.messages[0].role != "user":
            raise HTTPException(
                status_code=400,
                detail="Only one user message is supported; multi-turn/system messages "
                       "are not implemented and will not be silently discarded",
            )
        question = request.messages[0].content.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question is empty")
        backend = routes[request.model]
        try:
            # Serializes calls to shared Torch weights and optional browser context.
            with inference_lock:
                reply = current_engine.reply(
                    question, new_tokens=request.max_tokens, backend=backend
                )
        except Exception as exc:
            # Never leak filesystem paths, auth values, or browser session details.
            raise HTTPException(
                status_code=503,
                detail=f"Inference failed ({type(exc).__name__}); check server logs",
            ) from exc

        if reply.mode == "error":
            raise HTTPException(status_code=502, detail="External model request failed")
        if reply.mode == "notice":
            raise HTTPException(status_code=503, detail=reply.text)
        if reply.mode in {"limit", "vocab"}:
            raise HTTPException(status_code=422, detail=reply.text)
        if not reply.text.strip():
            raise HTTPException(status_code=503, detail="Model returned empty output")

        # 'srt_provider' distinguishes local learning from third-party answers.
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply.text},
                "finish_reason": "stop",
            }],
            "srt_provider": reply.provider,
            "srt_mode": reply.mode,
        }

    return app


app = create_app()
