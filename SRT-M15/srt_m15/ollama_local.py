"""Optional localhost-only Ollama bridge for conversational SRT-M15 UI.

This provides PRETRAINED LOCAL QWEN inference, not SRT-M15 neural weights.
It requires a separately installed Ollama server and downloaded model.
No DeepSeek API or cloud inference is involved.
"""
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaLocalError(RuntimeError):
    pass


class OllamaLocalClient:
    URL = "http://127.0.0.1:11434/api/chat"

    def __init__(self, *, model="qwen3:4b-instruct", timeout=180):
        if not isinstance(model, str) or not model.strip() or len(model) > 100:
            raise ValueError("invalid local Ollama model")
        self.model = model.strip()
        self.timeout = max(5, min(int(timeout), 300))

    def chat(self, messages, *, max_tokens=256):
        if not isinstance(messages, list) or not 1 <= len(messages) <= 20:
            raise ValueError("messages must have between 1 and 20 turns")
        if not 1 <= max_tokens <= 2048:
            raise ValueError("max_tokens must be 1..2048")
        for item in messages:
            if (not isinstance(item, dict)
                    or item.get("role") not in ("system", "user", "assistant")
                    or not isinstance(item.get("content"), str)
                    or not 1 <= len(item["content"].strip()) <= 4000):
                raise ValueError("invalid Ollama chat message")
        body = json.dumps({
            "model": self.model, "messages": messages, "stream": False,
            "options": {"num_predict": max_tokens, "num_ctx": 4096},
        }, ensure_ascii=False).encode("utf-8")
        request = Request(self.URL, body, headers={"Content-Type": "application/json"},
                          method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            raise OllamaLocalError(
                f"Local Ollama error (HTTP {exc.code}); check that model {self.model} is installed."
            ) from exc
        except (OSError, URLError, TimeoutError, ValueError) as exc:
            raise OllamaLocalError(
                "Cannot reach local Ollama at 127.0.0.1:11434. "
                "Install/start Ollama, then run: ollama pull " + self.model
            ) from exc
        try:
            answer = result["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise OllamaLocalError("Invalid response from local Ollama") from exc
        if not isinstance(answer, str) or not answer.strip():
            raise OllamaLocalError("Local Ollama returned an empty answer")
        return answer.strip()
