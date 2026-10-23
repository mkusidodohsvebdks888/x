"""Offline tests for SRT-M15's authenticated HTTP API; no model weights needed."""
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

# Importing the module does NOT start the server or contact any external provider.
from api_server import (
    API_MODEL, LOCAL_MODEL, WEB_MODEL, create_app,
)
from srt_m15.chat import ChatReply


KEY = "test-local-secret-1234567890"


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.next = ChatReply("text", "我是SRT-M15。", "SRT-M15")

    def reply(self, message, *, new_tokens=80, backend=None):
        self.calls.append((message, new_tokens, backend))
        return self.next

    def status(self):
        return {
            "math_checkpoint": True,
            "language_checkpoint": True,
        }

    def close(self):
        self.closed = True


class APITests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        # Explicit route opt-in is intentionally NOT supplied.
        self.app = create_app(api_key=KEY, engine=self.engine, model_routes={})
        self.client_ctx = TestClient(self.app)
        self.client = self.client_ctx.__enter__()
        self.auth = {"Authorization": f"Bearer {KEY}"}

    def tearDown(self):
        self.client_ctx.__exit__(None, None, None)

    def test_rejects_missing_and_bad_keys(self):
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            result = self.client.get("/v1/models", headers=headers)
            self.assertEqual(result.status_code, 401)
        self.assertEqual(self.engine.calls, [])

    def test_lists_only_local_model_by_default(self):
        resp = self.client.get("/v1/models", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual([v["id"] for v in resp.json()["data"]], [LOCAL_MODEL])

    def test_chat_shape_and_provider(self):
        resp = self.client.post(
            "/v1/chat/completions", headers=self.auth,
            json={"model": LOCAL_MODEL,
                  "messages": [{"role": "user", "content": "你是谁"}],
                  "max_tokens": 30},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["choices"][0]["message"]["content"], "我是SRT-M15。")
        self.assertEqual(data["srt_provider"], "SRT-M15")
        self.assertEqual(self.engine.calls, [("你是谁", 30, "local")])

    def test_requires_single_turn_instead_of_dropping_history(self):
        body = {"model": LOCAL_MODEL, "messages": [
            {"role": "system", "content": "请简短回答"},
            {"role": "user", "content": "你是谁"},
        ]}
        resp = self.client.post("/v1/chat/completions", headers=self.auth, json=body)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("multi-turn", resp.json()["detail"])
        self.assertEqual(self.engine.calls, [])

    def test_rejects_streaming_and_unsupported_temperature(self):
        base = {"model": LOCAL_MODEL,
                "messages": [{"role": "user", "content": "你好"}]}
        for extra in ({"stream": True}, {"temperature": 0.7}):
            res = self.client.post("/v1/chat/completions",
                                   headers=self.auth, json={**base, **extra})
            self.assertEqual(res.status_code, 400)
        self.assertEqual(self.engine.calls, [])

    def test_external_model_is_not_exposed_by_default(self):
        resp = self.client.post("/v1/chat/completions", headers=self.auth, json={
            "model": API_MODEL,
            "messages": [{"role": "user", "content": "你好"}],
        })
        self.assertEqual(resp.status_code, 404)

    def test_disabled_model_not_routed_even_if_requested(self):
        resp = self.client.post("/v1/chat/completions", headers=self.auth, json={
            "model": WEB_MODEL,
            "messages": [{"role": "user", "content": "你好"}],
        })
        self.assertEqual(resp.status_code, 404)

    def test_unavailable_weights_return_service_unavailable(self):
        self.engine.next = ChatReply("notice", "请训练并加载权重")
        res = self.client.post("/v1/chat/completions", headers=self.auth, json={
            "model": LOCAL_MODEL, "messages": [{"role": "user", "content": "你好"}],
        })
        self.assertEqual(res.status_code, 503)

    def test_limited_input_return_422(self):
        self.engine.next = ChatReply("vocab", "词表缺少字符")
        res = self.client.post("/v1/chat/completions", headers=self.auth, json={
            "model": LOCAL_MODEL, "messages": [{"role": "user", "content": "你好"}],
        })
        self.assertEqual(res.status_code, 422)

    def test_health_requires_auth(self):
        self.assertEqual(self.client.get("/health").status_code, 401)
        res = self.client.get("/health", headers=self.auth)
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["local_language_checkpoint"])

    def test_explicitly_enabled_external_model_has_distinct_source(self):
        fake = FakeEngine()
        fake.next = ChatReply("external", "远程回答", "DeepSeek API")
        app = create_app(api_key=KEY, engine=fake,
                         model_routes={API_MODEL: "api"})
        with TestClient(app) as client:
            models = client.get("/v1/models", headers=self.auth).json()["data"]
            self.assertEqual(set(x["id"] for x in models),
                             {LOCAL_MODEL, API_MODEL})
            res = client.post("/v1/chat/completions", headers=self.auth, json={
                "model": API_MODEL, "messages": [{"role": "user", "content": "你好"}],
            })
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["srt_provider"], "DeepSeek API")
            self.assertEqual(fake.calls, [("你好", 80, "api")])

    def test_external_failure_returns_bad_gateway(self):
        fake = FakeEngine()
        fake.next = ChatReply("error", "远端超时", "DeepSeek API")
        app = create_app(api_key=KEY, engine=fake,
                         model_routes={API_MODEL: "api"})
        with TestClient(app) as client:
            res = client.post("/v1/chat/completions", headers=self.auth, json={
                "model": API_MODEL, "messages": [{"role": "user", "content": "你好"}],
            })
            self.assertEqual(res.status_code, 502)
            self.assertNotIn("远端超时", res.text)

    def test_api_key_must_be_configured(self):
        app = create_app(api_key="", engine=FakeEngine())
        with self.assertRaises(RuntimeError):
            with TestClient(app):
                pass

    def test_shutdown_releases_engine(self):
        # The engine shared by the app created in setUp is closed in tearDown.
        app = create_app(api_key=KEY, engine=FakeEngine())
        with TestClient(app):
            self.assertFalse(app.state.engine.closed)
        self.assertTrue(app.state.engine.closed)


if __name__ == "__main__":
    unittest.main()
