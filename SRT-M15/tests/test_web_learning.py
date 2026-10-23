"""Deterministic tests. NO tests contact Wikipedia or the public Internet."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from srt_m15.web_learning import (
    WikipediaFetcher, WebKnowledgeStore, WebLearningController,
    WebStudyConfig, allowed_page_url, normalize_topic,
)


def sample(topic="人工智能", title="人工智能", language="zh", n=1):
    host = "zh" if language == "zh" else "en"
    return {"title": title, "summary": f"关于{topic}的可靠来源摘录示例，测试专用。机器学习模型需要数据和验证方法才能可靠工作。 " * 3,
            "url": f"https://{host}.wikipedia.org/wiki/{title.replace(' ', '_')}_{n}",
            "topic": topic, "language": language, "fetched_at": "2026-10-23T12:00:00+00:00"}


class ConfigAndFetcherTests(unittest.TestCase):
    def test_config_bounds_and_topics(self):
        self.assertEqual(WebStudyConfig.from_request({}).language, "zh")
        self.assertEqual(WebStudyConfig.from_request({"topics": "数学，物理;化学"}).topics, "数学, 物理, 化学")
        for bad in ({"interval_seconds": 0.1}, {"max_requests": 100}, {"max_requests": 1.3},
                    {"topics": "http://127.0.0.1/private"}, {"language": "random"},
                    {"max_runtime_seconds": 0}, {"topics": "\n"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                WebStudyConfig.from_request(bad)

    def test_only_fixed_encyclopedia_urls_are_accepted(self):
        self.assertTrue(allowed_page_url("https://zh.wikipedia.org/wiki/机器学习", "zh"))
        for malicious in ("https://evil.test/wiki/x", "http://zh.wikipedia.org/wiki/x",
                          "https://zh.wikipedia.org.evil.test/wiki/x", "file:///etc/passwd",
                          "https://zh.wikipedia.org@evil.test/wiki/x", "https://127.0.0.1/wiki/test"):
            self.assertFalse(allowed_page_url(malicious, "zh"))
        with self.assertRaises(ValueError):
            normalize_topic("file:///etc/passwd")

    def test_api_response_parser_and_fixed_host(self):
        api = {"query": {"pages": [
            {"pageid": 12, "title": "人工智能", "extract": "机器学习是一种通过数据来学习规律的人工智能技术。" * 3,
             "fullurl": "https://zh.wikipedia.org/wiki/人工智能"},
            {"pageid": 13, "title": "恶意网页", "extract": "恶意代码要执行" * 9,
             "fullurl": "http://localhost/admin"},
        ]}}

        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self, n):
                return json.dumps(api, ensure_ascii=False).encode("utf-8")
        class Opener:
            def open(self, request, timeout):
                self_host = request.full_url.split("/w/")[0]
                assert self_host == "https://zh.wikipedia.org", self_host
                assert "SRTM15KnowledgeBot" in request.get_header("User-agent")
                return Response()
        with patch("srt_m15.web_learning.build_opener", return_value=Opener()):
            notes = WikipediaFetcher()("人工智能", "zh")
        self.assertEqual(len(notes), 1)
        self.assertIn("wikipedia.org", notes[0]["url"])


class StoreAndControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls = []

        def fetch(topic, language, timeout):
            self.calls.append((topic, language))
            # Add two different pages, one is a related followup topic.
            return [sample(topic, title=topic, n=1),
                    sample(topic, title=topic + "技术", n=2)]
        self.fetch = fetch
        self.controller = WebLearningController(self.root, fetcher=self.fetch)
        self.controller.last_request_time = time.monotonic() - 10

    def tearDown(self):
        self.controller.shutdown()

    def test_disabled_by_default_and_no_spontaneous_network(self):
        self.assertFalse(self.controller.status()["enabled"])
        self.assertFalse(self.controller.status()["running"])
        self.assertEqual(self.calls, [])
        with self.assertRaises(ValueError):
            self.controller.search_once("人工智能")
        with self.assertRaises(ValueError):
            self.controller.start({"topics": "人工智能"})

    def test_explicit_search_cites_provenance_and_review(self):
        self.controller.set_enabled(True)
        result = self.controller.search_once("人工智能")
        self.assertEqual(result["found"], 2)
        self.assertEqual(result["saved"], 2)
        self.assertEqual(self.controller.status()["pending_review"], 2)
        answer = self.controller.answer_from_memory("人工智能是什么？")
        self.assertIn("来源：https://zh.wikipedia.org", answer)
        self.assertIn("尚未人工审核", answer)
        ids = [item["id"] for item in self.controller.status()["recent"]]
        self.controller.review(ids[0], True)
        self.assertIn("已由用户标记审核", self.controller.answer_from_memory("人工智能是什么？"))
        qa = self.controller.store.export_approved_qa()
        self.assertIn("source", qa)
        self.assertEqual(len(qa.strip().splitlines()), 1)
        self.controller.review(ids[1], False)
        self.assertEqual(self.controller.status()["rejected"], 1)
        self.assertEqual(len(self.controller.store.retrieve("人工智能")), 1)

    def test_invalid_review_and_invalid_source_rejected(self):
        with self.assertRaises(ValueError):
            self.controller.review("1", True)
        with self.assertRaises(ValueError):
            self.controller.review(1, True)
        with self.assertRaises(ValueError):
            self.controller.review(1, "approved")
        malicious = sample()
        malicious["url"] = "https://evil.example/wiki/人工智能"
        self.assertEqual(self.controller.store.save([malicious]), 0)
        self.assertEqual(self.controller.store.stats()["stored"], 0)

    def test_auto_fetch_limit_topic_expansion_and_restart_pause(self):
        self.controller.set_enabled(True)
        self.controller.start({"topics": "人工智能", "max_requests": 1,
                               "interval_seconds": 8, "max_runtime_seconds": 20,
                               "max_new_records": 5})
        self.controller.worker.join(timeout=5)
        snapshot = self.controller.status()
        self.assertFalse(snapshot["running"])
        self.assertEqual(snapshot["stop_reason"], "max_requests")
        self.assertEqual(snapshot["session_requests"], 1)
        self.assertEqual(snapshot["session_saved"], 2)
        self.assertEqual(len(self.calls), 1)
        saved = WebLearningController(self.root, fetcher=self.fetch)
        self.assertFalse(saved.status()["enabled"])
        self.assertFalse(saved.status()["running"])
        self.assertEqual(saved.status()["stored"], 2)
        self.assertEqual(saved.status()["total_requests"], 1)

    def test_manual_pause_stops_before_next_request(self):
        self.controller.set_enabled(True)
        self.controller.start({"topics": "人工智能, 神经网络", "max_requests": 3,
                               "interval_seconds": 8, "max_runtime_seconds": 35,
                               "max_new_records": 10})
        deadline = time.monotonic() + 4
        while len(self.calls) < 1 and time.monotonic() < deadline:
            time.sleep(.025)
        self.controller.pause()
        self.controller.worker.join(timeout=3)
        self.assertEqual(self.controller.status()["stop_reason"], "manual_pause")
        self.assertFalse(self.controller.status()["running"])
        self.assertEqual(len(self.calls), 1)

    def test_max_runtime_even_if_request_interval_is_long(self):
        self.controller.set_enabled(True)
        self.controller.start({"topics": "人工智能, 神经网络", "max_requests": 4,
                               "interval_seconds": 80, "max_runtime_seconds": 10,
                               "max_new_records": 10})
        # Avoid waiting 10s in the test: simulate elapsed session start.
        with self.controller.lock:
            self.controller.session_started -= 9.7
        self.controller.worker.join(timeout=3)
        self.assertFalse(self.controller.status()["running"])
        self.assertEqual(self.controller.status()["stop_reason"], "max_runtime")

    def test_network_failure_stops_after_three_errors(self):
        def failure(topic, language, timeout):
            self.calls.append((topic, language))
            raise RuntimeError("网站暂时不可达")
        self.controller.fetcher = failure
        self.controller.last_request_time = time.monotonic() - 10
        self.controller.set_enabled(True)
        # Set brief interval and time to bypass real 8s rate limit in this test.
        # Simulate time progression by patching _request instead.
        self.controller._request = lambda topic, language, stop=None, deadline=None: failure(topic, language, 1)
        self.controller.start({"topics": "数学, 化学, 物理", "max_requests": 10,
                               "interval_seconds": 8, "max_runtime_seconds": 100,
                               "max_new_records": 10})
        self.controller.worker.join(timeout=22)
        self.assertFalse(self.controller.status()["running"])
        self.assertEqual(self.controller.status()["stop_reason"], "too_many_errors")
        self.assertEqual(self.controller.status()["session_requests"], 3)


if __name__ == "__main__":
    unittest.main()
