"""Optional official DeepSeek API adapter. The browser bridge does not need this."""
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

    def ask(self, prompt):
        if not self.api_key:
            raise DeepSeekAPIError(
                "没有 DEEPSEEK_API_KEY。网页版不需要 API Key，"
                "但本地桌面网页版模式需要安装 Playwright 并手动登录。"
            )
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": str(prompt)}],
            "stream": False,
        }).encode("utf-8")
        req = request.Request(
            self.URL, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                payload = json.load(response)
        except error.HTTPError as exc:
            raise DeepSeekAPIError(f"DeepSeek API 返回 HTTP {exc.code}。") from exc
        except (error.URLError, TimeoutError) as exc:
            raise DeepSeekAPIError(f"DeepSeek API 网络连接失败：{type(exc).__name__}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty answer")
            return content.strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise DeepSeekAPIError("DeepSeek API 未返回有效文本。") from exc
