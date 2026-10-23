"""Explicit opt-in DeepSeek API adapter with genuine structured chat and SSE.

This is an external commercial model. Replies are never local SRT-M15 weights.
No API key is logged or included in errors.
"""
import json
import os
from urllib import error, request


class DeepSeekAPIError(RuntimeError):
    pass


class DeepSeekAPIClient:
    URL = "https://api.deepseek.com/chat/completions"

    def __init__(self, api_key=None, model=None, timeout=70):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        self.timeout = max(5, int(timeout))

    def _prepare(self, messages, *, stream, max_tokens):
        if not self.api_key:
            raise DeepSeekAPIError("没有 DEEPSEEK_API_KEY")
        if not isinstance(messages, list) or not messages:
            raise DeepSeekAPIError("messages must be a nonempty list")
        if any(
            not isinstance(msg, dict)
            or msg.get("role") not in ("system", "user", "assistant")
            or not isinstance(msg.get("content"), str)
            or not msg["content"].strip()
            for msg in messages
        ):
            raise DeepSeekAPIError("Invalid messages for DeepSeek API")
        body = {
            "model": self.model,
            "messages": [{"role": msg["role"], "content": msg["content"]}
                         for msg in messages],
            "stream": bool(stream),
        }
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 8192:
                raise DeepSeekAPIError("max_tokens out of bounds")
            body["max_tokens"] = max_tokens
        return request.Request(
            self.URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            method="POST", headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

    def chat(self, messages, *, max_tokens=None):
        req = self._prepare(messages, stream=False, max_tokens=max_tokens)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                payload = json.load(response)
        except error.HTTPError as exc:
            raise DeepSeekAPIError(f"DeepSeek API HTTP {exc.code}") from exc
        except (error.URLError, TimeoutError, ValueError) as exc:
            raise DeepSeekAPIError("DeepSeek API connection or JSON error") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty answer")
            return content.strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise DeepSeekAPIError("DeepSeek API returned no valid answer") from exc

    def iter_chat(self, messages, *, max_tokens=None):
        """Yield real provider deltas as its HTTP SSE stream arrives.

        This is a generator: the network opens on first iteration. The API
        facade sends a streamed error event if the upstream fails midstream.
        """
        req = self._prepare(messages, stream=True, max_tokens=max_tokens)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                for line in response:
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[len("data:"):].strip()
                    if chunk == "[DONE]":
                        return
                    try:
                        parsed = json.loads(chunk)
                        item = parsed["choices"][0]["delta"].get("content")
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        raise DeepSeekAPIError("Malformed upstream SSE delta") from exc
                    if isinstance(item, str) and item:
                        yield item
        except error.HTTPError as exc:
            raise DeepSeekAPIError(f"DeepSeek API HTTP {exc.code}") from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            raise DeepSeekAPIError("DeepSeek API streaming connection failed") from exc

    def ask(self, prompt):
        """Legacy single-turn wrapper. Preserve existing provider behavior."""
        return self.chat([{"role": "user", "content": str(prompt)}])
