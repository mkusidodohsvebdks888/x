"""Opt-in, bounded Internet knowledge acquisition for SRT-M15.

Network learning adds attributed, reviewable *retrieval memory*. It deliberately
DOES NOT modify neural weights or treat public text as verified training labels.
Only the public Wikimedia MediaWiki read API is contacted, over HTTPS with a
fixed hostname, without crawling third-party links or executing remote content.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .practice import atomic_write_json


WIKI_HOSTS = {"zh": "zh.wikipedia.org", "en": "en.wikipedia.org"}
USER_AGENT = "SRTM15KnowledgeBot/0.1 (https://github.com/mkusidodohsvebdks888/x) Python-urllib"
MAX_JSON_BYTES = 160_000
MAX_STORE_ROWS = 200


class WebRateLimited(RuntimeError):
    """Stop the autonomous session instead of retrying a rate-limited request."""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_topic(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("学习主题必须是文本")
    text = unicodedata.normalize("NFKC", text).strip()
    text = " ".join(text.split())
    if not 2 <= len(text) <= 60 or any(ord(c) < 32 for c in text):
        raise ValueError("主题长度须为2～60字符，不能包含控制字符")
    if re.search(r"(?:[a-z][a-z0-9+.-]*://|www\.|\n|\r)", text, flags=re.I):
        raise ValueError("请输入主题词，而不是网址；此版本只访问固定的维基百科 API")
    return text


def allowed_page_url(value: str, language: str) -> bool:
    try:
        url = urlparse(value)
        return (language in WIKI_HOSTS and url.scheme == "https" and
                url.hostname == WIKI_HOSTS[language] and url.username is None and
                url.password is None and (url.port is None or url.port == 443) and
                url.path.startswith("/wiki/") and not url.fragment)
    except (ValueError, TypeError):
        return False


@dataclass(frozen=True)
class WebStudyConfig:
    topics: str = "人工智能, 神经网络, 机器学习"
    language: str = "zh"
    interval_seconds: float = 12.0
    max_requests: int = 12
    max_runtime_seconds: int = 180
    max_new_records: int = 24

    @classmethod
    def from_request(cls, value):
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("联网设置必须是JSON对象")
        if set(value) - set(cls.__dataclass_fields__):
            raise ValueError("联网设置存在未识别参数")
        defaults = cls()
        topics = value.get("topics", defaults.topics)
        if not isinstance(topics, str) or len(topics) > 500:
            raise ValueError("主题列表过长")
        items = [normalize_topic(x) for x in re.split(r"[,，;；\n]+", topics) if x.strip()]
        if not 1 <= len(items) <= 10:
            raise ValueError("请输入1～10个学习主题")
        topics = ", ".join(dict.fromkeys(items))
        language = value.get("language", defaults.language)
        if language not in WIKI_HOSTS:
            raise ValueError("目前只支持 zh / en 维基百科")
        limits = {"interval_seconds": (8, 120), "max_requests": (1, 30),
                  "max_runtime_seconds": (10, 3600), "max_new_records": (1, 60)}
        vals = {}
        for key, (lower, upper) in limits.items():
            item = value.get(key, getattr(defaults, key))
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ValueError(f"{key} 必须是有效数字")
            if not lower <= item <= upper:
                raise ValueError(f"{key} 允许范围：{lower}～{upper}")
            if key != "interval_seconds" and not isinstance(item, int):
                raise ValueError(f"{key} 必须是整数")
            vals[key] = item
        return cls(topics=topics, language=language, **vals)


class _HttpsWikiRedirect(HTTPRedirectHandler):
    def __init__(self, host):
        self.host = host

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        p = urlparse(newurl)
        if p.scheme != "https" or p.hostname != self.host:
            raise ValueError("Blocked non-Wikipedia redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class WikipediaFetcher:
    """Perform one read-only MediaWiki API request, with strict limits."""

    def __call__(self, topic: str, language: str = "zh", timeout: float = 6.0):
        topic = normalize_topic(topic)
        if language not in WIKI_HOSTS:
            raise ValueError("Unsupported encyclopedia")
        host = WIKI_HOSTS[language]
        args = {
            "action": "query", "generator": "search", "gsrsearch": topic,
            "gsrlimit": 3, "prop": "extracts|info", "inprop": "url",
            "exintro": "1", "explaintext": "1", "exchars": "1100",
            "format": "json", "formatversion": "2", "redirects": "1",
        }
        url = f"https://{host}/w/api.php?" + urlencode(args)
        req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        opener = build_opener(_HttpsWikiRedirect(host))
        try:
            with opener.open(req, timeout=max(1.0, min(float(timeout), 8.0))) as resp:
                payload = resp.read(MAX_JSON_BYTES + 1)
                if len(payload) > MAX_JSON_BYTES:
                    raise ValueError("联网响应过大，已拒绝")
                data = json.loads(payload.decode("utf-8"))
        except HTTPError as exc:
            if exc.code in (429, 503):
                raise WebRateLimited("维基百科返回限流或服务不可用，已停止自动请求；请稍后再试") from exc
            raise RuntimeError(f"维基百科接口响应 HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("维基百科连接失败；请检查网络或稍后重试") from exc
        if not isinstance(data, dict) or "error" in data:
            raise RuntimeError("维基百科接口返回了错误或非预期数据")
        pages = data.get("query", {}).get("pages", [])
        if isinstance(pages, dict):
            pages = list(pages.values())
        if not isinstance(pages, list):
            return []
        results = []
        for page in pages[:3]:
            if not isinstance(page, dict):
                continue
            title = page.get("title", "")
            summary = page.get("extract", "")
            link = page.get("fullurl", "")
            if not (isinstance(title, str) and isinstance(summary, str) and isinstance(link, str)):
                continue
            title, summary = title.strip()[:130], " ".join(summary.split())[:1100]
            if len(title) < 2 or len(summary) < 25 or not allowed_page_url(link, language):
                continue
            results.append({"title": title, "summary": summary, "url": link,
                            "language": language, "topic": topic, "fetched_at": utc_now()})
        return results


class WebKnowledgeStore:
    """Bounded SQLite store of attributed, reviewable excerpts, no HTML."""

    def __init__(self, root: Path):
        self.path = Path(root) / "web_memory" / "sources.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL UNIQUE, title TEXT NOT NULL,
                summary TEXT NOT NULL, topic TEXT NOT NULL,
                language TEXT NOT NULL, fetched_at TEXT NOT NULL,
                review TEXT NOT NULL DEFAULT 'pending'
                   CHECK (review IN ('pending','approved','rejected'))
            )""")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=3)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, notes):
        added = 0
        with self._connect() as db:
            for note in notes:
                lang = note.get("language")
                if (not isinstance(note.get("title"), str) or
                    not isinstance(note.get("summary"), str) or
                    not isinstance(note.get("url"), str) or
                    not allowed_page_url(note["url"], lang)):
                    continue
                if not (2 <= len(note["title"]) <= 130 and 25 <= len(note["summary"]) <= 1100):
                    continue
                count = db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
                if count >= MAX_STORE_ROWS:
                    break
                result = db.execute("""INSERT OR IGNORE INTO notes
                    (url,title,summary,topic,language,fetched_at)
                    VALUES (?,?,?,?,?,?)""", (
                        note["url"], note["title"], note["summary"],
                        note.get("topic", "")[:60], lang, note.get("fetched_at", utc_now())))
                added += int(result.rowcount == 1)
        return added

    def stats(self):
        with self._connect() as db:
            rows = db.execute("SELECT review,COUNT(*) AS n FROM notes GROUP BY review").fetchall()
        counts = {x["review"]: x["n"] for x in rows}
        return {"stored": sum(counts.values()), "pending_review": counts.get("pending", 0),
                "approved": counts.get("approved", 0), "rejected": counts.get("rejected", 0)}

    def recent(self, limit=6):
        with self._connect() as db:
            rows = db.execute("""SELECT id,title,url,summary,topic,language,fetched_at,review
                                 FROM notes ORDER BY id DESC LIMIT ?""", (max(1, min(limit, 10)),)).fetchall()
        return [dict(row) for row in rows]

    def review(self, note_id, approved: bool):
        if isinstance(note_id, bool) or not isinstance(note_id, int) or note_id < 1:
            raise ValueError("无效资料ID")
        if not isinstance(approved, bool):
            raise ValueError("审核结果必须是布尔值")
        with self._connect() as db:
            row = db.execute("SELECT review FROM notes WHERE id=?", (note_id,)).fetchone()
            if row is None:
                raise ValueError("找不到资料记录")
            db.execute("UPDATE notes SET review=? WHERE id=?",
                       ("approved" if approved else "rejected", note_id))
        return {"id": note_id, "review": "approved" if approved else "rejected"}

    def approved_note(self, note_id):
        """Only user-reviewed source metadata may be proposed as Q/A drafts."""
        if isinstance(note_id, bool) or not isinstance(note_id, int) or note_id < 1:
            return None
        with self._connect() as db:
            row = db.execute("""SELECT id,title,url,summary,topic,language,fetched_at
                FROM notes WHERE id=? AND review='approved'""", (note_id,)).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _terms(value):
        s = unicodedata.normalize("NFKC", value).lower()
        s = re.sub(r"(请问|什么是|是什么|介绍一下|告诉我|请介绍|联网搜索|联网查询)", "", s)
        chunks = re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+", s)
        out = set()
        for chunk in chunks:
            if len(chunk) == 1:
                continue
            if re.match(r"[\u3400-\u9fff]", chunk):
                out.update(chunk[i:i+2] for i in range(len(chunk) - 1))
            else:
                out.add(chunk)
        return out

    def retrieve(self, query, limit=2):
        if not isinstance(query, str) or len(query) > 400:
            return []
        terms = self._terms(query)
        if not terms:
            return []
        with self._connect() as db:
            rows = db.execute("""SELECT id,title,url,summary,topic,language,fetched_at,review
                                 FROM notes WHERE review!='rejected' ORDER BY id DESC LIMIT 200""").fetchall()
        found = []
        for row in rows:
            note = dict(row)
            title_terms = self._terms(note["title"] + " " + note["topic"])
            body_terms = self._terms(note["summary"])
            title_match = len(terms & title_terms)
            body_match = len(terms & body_terms)
            score = title_match * 4 + body_match
            if score > 0:
                found.append((score, note["review"] == "approved", note["id"], note))
        found.sort(key=lambda t: (-t[0], -int(t[1]), -t[2]))
        return [x[3] for x in found[: max(1, min(limit, 5))]]

    def export_approved_qa(self):
        with self._connect() as db:
            rows = db.execute("SELECT title,summary,url,language FROM notes WHERE review='approved' ORDER BY id").fetchall()
        return "".join(json.dumps({
            "question": "什么是" + row["title"] + "？",
            "answer_excerpt": row["summary"][:650],
            "source": row["url"], "language": row["language"],
            "attribution": row["title"] + " — Wikipedia editors (see page revision history)",
            "license_note": "Wikipedia text is generally CC BY-SA; check the linked page for its exact license and attribution requirements",
            "note": "人工已标记待用于后续监督训练；此文件尚未改变模型参数",
        }, ensure_ascii=False) + "\n" for row in rows)


class WebLearningController:
    """Bounded autonomous read-only retrieval, never a neural-weight trainer."""

    def __init__(self, root, fetcher=None):
        self.root = Path(root)
        self.store = WebKnowledgeStore(self.root)
        self.fetcher = fetcher if fetcher is not None else WikipediaFetcher()
        self.state_file = self.root / "web_memory" / "study_state.json"
        self.lock = threading.RLock()
        self.request_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = None
        self.enabled = False  # fail-closed after EVERY process restart
        self.running = False
        self.config = WebStudyConfig()
        self.session_requests = 0
        self.session_saved = 0
        self.session_started = 0.0
        self.total_requests = 0
        self.total_saved = 0
        self.errors = 0
        self.last_topic = ""
        self.last_error = ""
        self.last_result = "尚未联网检索"
        self.stop_reason = "not_started"
        self.last_request_time = 0.0
        self._load()

    def _load(self):
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for name in ("total_requests", "total_saved", "errors"):
                        val = data.get(name, 0)
                        setattr(self, name, max(0, int(val)) if not isinstance(val, bool) else 0)
                    if isinstance(data.get("config"), dict):
                        self.config = WebStudyConfig.from_request(data["config"])
                    self.last_topic = str(data.get("last_topic", ""))[:60]
                    self.last_result = str(data.get("last_result", self.last_result))[:200]
            except (OSError, ValueError, TypeError, KeyError):
                pass
        self.enabled = False
        self.running = False
        self.stop_reason = "interrupted_by_restart"

    def _save_locked(self):
        atomic_write_json(self.state_file, {
            "total_requests": self.total_requests, "total_saved": self.total_saved,
            "errors": self.errors, "last_topic": self.last_topic,
            "last_result": self.last_result, "config": asdict(self.config),
            "last_update": utc_now(),
        })

    def set_enabled(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("联网授权开关必须为 true/false")
        if not enabled:
            self.pause()
        with self.lock:
            self.enabled = enabled
            self._save_locked()
        return self.status()

    def status(self):
        with self.lock:
            stats = self.store.stats()
            return {
                "enabled": self.enabled, "running": self.running,
                "phase": "collecting" if self.running else "stopped",
                "stop_reason": self.stop_reason,
                "config": asdict(self.config),
                "session_requests": self.session_requests,
                "session_saved": self.session_saved,
                "total_requests": self.total_requests,
                "total_saved": self.total_saved,
                "errors": self.errors, "last_topic": self.last_topic,
                "last_error": self.last_error, "last_result": self.last_result,
                **stats, "recent": self.store.recent(limit=6),
            }

    def start(self, args):
        cfg = WebStudyConfig.from_request(args)
        with self.lock:
            if not self.enabled:
                raise ValueError("请先明确启用联网学习开关")
            if self.running or (self.worker and self.worker.is_alive()):
                raise ValueError("联网学习正在运行或正在停止")
            self.config = cfg
            self.session_requests = 0
            self.session_saved = 0
            self.session_started = time.monotonic()
            self.stop_reason = ""
            self.last_error = ""
            self.stop_event.clear()
            self.running = True
            self._save_locked()
            self.worker = threading.Thread(target=self._worker_loop, name="srt-web-study", daemon=True)
            self.worker.start()
        return self.status()

    def pause(self):
        with self.lock:
            if self.running:
                self.stop_reason = "manual_pause"
                self.stop_event.set()
                self.running = False
                self._save_locked()
            else:
                self.stop_event.set()
        return self.status()

    def _request(self, topic, language, *, stop=None, deadline=None):
        topic = normalize_topic(topic)
        with self.request_lock:
            if stop and stop.is_set():
                return []
            if deadline is not None and time.monotonic() >= deadline:
                return []
            # Shared service-wide rate limit for both manual and scheduled requests.
            wait = max(0, 8.0 - (time.monotonic() - self.last_request_time))
            if deadline is not None:
                wait = min(wait, max(0, deadline - time.monotonic()))
            if wait > 0:
                if stop:
                    if stop.wait(wait):
                        return []
                else:
                    time.sleep(wait)
            if (stop and stop.is_set()) or (deadline is not None and time.monotonic() >= deadline):
                return []
            timeout = min(6.0, max(1.0, deadline - time.monotonic())) if deadline is not None else 6.0
            self.last_request_time = time.monotonic()
            return self.fetcher(topic, language, timeout=timeout)

    def search_once(self, topic, language="zh"):
        with self.lock:
            if not self.enabled:
                raise ValueError("联网功能当前关闭，请先开启")
        topic = normalize_topic(topic)
        if language not in WIKI_HOSTS:
            raise ValueError("仅支持中文或英文维基百科")
        try:
            found = self._request(topic, language)
        except (RuntimeError, ValueError) as exc:
            with self.lock:
                self.errors += 1
                self.last_error = str(exc)[:180]
                self.last_result = "手动检索失败"
                self._save_locked()
            raise ValueError(str(exc)) from exc
        with self.lock:
            if not self.enabled:
                raise ValueError("联网授权已经取消，结果未收录")
        added = self.store.save(found)
        with self.lock:
            self.total_requests += 1
            self.total_saved += added
            self.last_topic = topic
            self.last_error = ""
            self.last_result = f"手动检索“{topic}”：找到{len(found)}条摘要，新收录{added}条"
            self._save_locked()
        return {"topic": topic, "found": len(found), "saved": added,
                "sources": [{k: v for k, v in n.items() if k in ("title", "url", "summary", "fetched_at")} for n in found[:3]]}

    def _worker_loop(self):
        config = self.config
        queue = deque([x.strip() for x in config.topics.split(",") if x.strip()])
        seen = set()
        consecutive_errors = 0
        why = "topics_exhausted"
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    elapsed = time.monotonic() - self.session_started
                    if elapsed >= config.max_runtime_seconds:
                        why = "max_runtime"
                        break
                    if self.session_requests >= config.max_requests:
                        why = "max_requests"
                        break
                    if self.session_saved >= config.max_new_records:
                        why = "max_new_records"
                        break
                while queue and queue[0] in seen:
                    queue.popleft()
                if not queue:
                    why = "topics_exhausted"
                    break
                topic = queue.popleft()
                seen.add(topic)
                if self.stop_event.is_set():
                    break
                try:
                    notes = self._request(topic, config.language, stop=self.stop_event,
                                          deadline=self.session_started + config.max_runtime_seconds)
                    if self.stop_event.is_set():
                        break
                    with self.lock:
                        remaining = config.max_new_records - self.session_saved
                    if time.monotonic() - self.session_started >= config.max_runtime_seconds:
                        why = "max_runtime"
                        break
                    added = self.store.save(notes[: max(0, remaining)])
                    consecutive_errors = 0
                    with self.lock:
                        self.session_requests += 1
                        self.total_requests += 1
                        self.session_saved += added
                        self.total_saved += added
                        self.last_topic = topic
                        self.last_error = ""
                        self.last_result = f"自动学习“{topic}”：找到{len(notes)}篇，新收录{added}篇"
                        self._save_locked()
                    # Bounded keyword expansion from returned article titles.
                    if len(seen) + len(queue) < 35:
                        for note in notes[:2]:
                            title = note["title"]
                            if title != topic and title not in seen and title not in queue and len(title) <= 40:
                                queue.append(title)
                                break
                except (RuntimeError, ValueError, OSError) as exc:
                    consecutive_errors += 1
                    with self.lock:
                        self.session_requests += 1
                        self.total_requests += 1
                        self.errors += 1
                        self.last_topic = topic
                        self.last_error = str(exc)[:180]
                        self.last_result = "联网请求失败"
                        self._save_locked()
                    if isinstance(exc, WebRateLimited):
                        why = "rate_limited"
                        break
                    if consecutive_errors >= 3:
                        why = "too_many_errors"
                        break
                with self.lock:
                    if self.session_requests >= config.max_requests:
                        why = "max_requests"
                        break
                    if self.session_saved >= config.max_new_records:
                        why = "max_new_records"
                        break
                    remaining = config.max_runtime_seconds - (time.monotonic() - self.session_started)
                if remaining <= 0:
                    why = "max_runtime"
                    break
                if self.stop_event.wait(min(config.interval_seconds, remaining)):
                    break
        finally:
            with self.lock:
                if self.stop_reason != "manual_pause":
                    self.stop_reason = why
                self.running = False
                self._save_locked()

    def review(self, note_id, approved):
        return self.store.review(note_id, approved)

    def answer_from_memory(self, question):
        notes = self.store.retrieve(question, limit=2)
        if not notes:
            return None
        segments = ["联网知识库摘录（原文资料，不是 SRT-M15 推理得出的结论）："]
        for note in notes:
            label = "已由用户标记审核" if note["review"] == "approved" else "尚未人工审核"
            segments.append(f"\n• {note['title']}（{label}）\n{note['summary'][:420]}\n来源：{note['url']}")
        segments.append("\n资料可能存在错误或过时；联网记忆不等于神经网络权重已更新。")
        return "\n".join(segments)

    def shutdown(self):
        self.pause()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=8)
