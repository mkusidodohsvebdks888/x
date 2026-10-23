"""Opt-in DeepSeek WEB assistant via a visible local browser.

This is NOT the official API. No private endpoints, auth-token extraction,
cookie transfer, CAPTCHA bypass, or headless stealth are used.
It requires Playwright and a local desktop with a user-interactive browser.
It cannot control a separately logged-in phone browser from Colab.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import sys
import time


class DeepSeekWebError(RuntimeError):
    """The DeepSeek web page was unavailable or could not be read."""


class DeepSeekWebClient:
    URL = "https://chat.deepseek.com/"

    def __init__(self, *, profile_dir=None, timeout=120):
        self.profile_dir = Path(
            profile_dir or os.environ.get("SRT_DEEPSEEK_WEB_PROFILE")
            or Path.home() / ".srt_m15" / "deepseek_web"
        ).expanduser()
        self.timeout = max(10, int(timeout))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepseek-web")
        self._pw = None
        self._browser = None
        self._page = None

    def _check_environment(self):
        if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
            raise DeepSeekWebError(
                "Google Colab 运行在云端，无法直接使用你手机上已登录的 DeepSeek 网页。"
                "请在本地电脑执行网页版桥接，或者在 Colab 使用官方 API。"
            )
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            raise DeepSeekWebError("网页版桥接需要本地电脑的可视化桌面环境。")

    def _ensure_browser(self):
        self._check_environment()
        if self._page is not None and not self._page.is_closed():
            return self._page
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise DeepSeekWebError(
                "请先安装: pip install playwright && python -m playwright install chromium"
            ) from exc
        self.profile_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(self.profile_dir, 0o700)
        except OSError:
            pass
        try:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir), headless=False,
                viewport={"width": 1200, "height": 850},
            )
            self._page = self._browser.pages[0] if self._browser.pages else self._browser.new_page()
            self._page.goto(self.URL, wait_until="domcontentloaded", timeout=45000)
            return self._page
        except Exception as exc:
            raise DeepSeekWebError(
                "无法启动 DeepSeek 本地浏览器。请确认已经安装 Chromium，"
                "未在另一个进程中占用该浏览器配置目录。"
            ) from exc

    def _find_composer(self, page):
        # These are public DOM controls, not DeepSeek's undocumented network API.
        for selector in ('textarea', '[contenteditable="true"]', '[role="textbox"]'):
            candidates = page.locator(selector)
            for idx in range(candidates.count() - 1, -1, -1):
                candidate = candidates.nth(idx)
                try:
                    if candidate.is_visible() and candidate.is_enabled():
                        return candidate
                except Exception:
                    continue
        return None

    def _login(self):
        page = self._ensure_browser()
        print("请在打开的 DeepSeek 官方网页手动登录。不要把密码提供给 SRT-M15。")
        input("网页登录完成、出现聊天输入框后，回到终端按 Enter：")
        page.reload(wait_until="domcontentloaded", timeout=45000)
        if self._find_composer(page) is None:
            raise DeepSeekWebError("尚未检测到聊天输入框，请检查是否已经登录。")
        return "网页版已准备就绪；浏览器登录资料仅保留在本机独立配置目录。"

    def login(self):
        """Manually log in from the visible browser. Call once on a desktop."""
        return self._executor.submit(self._login).result()

    @staticmethod
    def _answer_blocks(page):
        # Current pages may rename classes: fail clearly instead of scraping
        # unrelated elements or silently returning the user's own question.
        for selector in (
            ".ds-markdown",
            '[data-role="assistant"] .markdown',
            '[data-message-role="assistant"]',
        ):
            loc = page.locator(selector)
            if loc.count():
                return loc
        return page.locator(".ds-markdown")

    def _ask(self, question):
        page = self._ensure_browser()
        if "sign_in" in page.url or "login" in page.url:
            raise DeepSeekWebError(
                "请先在本地电脑运行 python deepseek_web_cli.py --login，"
                "自行登录网页版 DeepSeek。"
            )
        composer = self._find_composer(page)
        if composer is None:
            raise DeepSeekWebError(
                "未找到 DeepSeek 网页输入框。请确认已登录；"
                "网页布局可能发生变化，需要更新选择器。"
            )
        blocks = self._answer_blocks(page)
        previous_count = blocks.count()
        composer.fill(question)
        composer.press("Enter")
        started = time.monotonic()
        previous_text = ""
        stable_since = None
        while time.monotonic() - started < self.timeout:
            page.wait_for_timeout(700)
            blocks = self._answer_blocks(page)
            if blocks.count() <= previous_count:
                continue
            answer = blocks.last.inner_text(timeout=3000).strip()
            if not answer:
                continue
            if answer == previous_text:
                if stable_since and time.monotonic() - stable_since >= 3.0:
                    return answer
            else:
                previous_text = answer
                stable_since = time.monotonic()
        raise DeepSeekWebError(
            "等待网页版回答超时，或页面 DOM 结构已变化。"
            "请直接查看浏览器，不会擅自重发同一问题。"
        )

    def ask(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("问题不能为空")
        return self._executor.submit(self._ask, question.strip()).result(
            timeout=self.timeout + 65
        )

    def _close(self):
        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            self._browser = None
            self._page = None
            if self._pw is not None:
                self._pw.stop()
                self._pw = None

    def close(self):
        try:
            self._executor.submit(self._close).result(timeout=15)
        finally:
            self._executor.shutdown(wait=False)
