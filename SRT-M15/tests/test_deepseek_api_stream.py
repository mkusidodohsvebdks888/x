"""Deterministic tests for DeepSeek structured chat and upstream SSE parsing."""
from contextlib import contextmanager
import io
import json
import unittest
from unittest.mock import patch

from srt_m15.deepseek_api import DeepSeekAPIClient, DeepSeekAPIError


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class DeepSeekAPIStreamingTests(unittest.TestCase):
    def setUp(self):
        self.client = DeepSeekAPIClient(api_key="fake-test-token", model="deepseek-chat")
        self.messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "我们刚才说了什么"},
        ]

    def test_structured_nonstream_preserves_history(self):
        payload = b'{"choices":[{"message":{"content":"We greeted each other."}}]}'
        captured = {}

        def fake_open(req, timeout):
            captured["body"] = json.loads(req.data)
            return FakeResponse(payload)

        with patch("srt_m15.deepseek_api.request.urlopen", side_effect=fake_open):
            output = self.client.chat(self.messages, max_tokens=64)
        self.assertEqual(output, "We greeted each other.")
        self.assertEqual(captured["body"]["messages"], self.messages)
        self.assertFalse(captured["body"]["stream"])
        self.assertEqual(captured["body"]["max_tokens"], 64)

    def test_true_provider_stream_yields_incremental_content(self):
        frames = [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"你"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"好"}}]}\n\n',
            'data: [DONE]\n\n',
        ]
        captured = {}

        def fake_open(req, timeout):
            captured["body"] = json.loads(req.data)
            return FakeResponse("".join(frames).encode("utf-8"))

        with patch("srt_m15.deepseek_api.request.urlopen", side_effect=fake_open):
            pieces = list(self.client.iter_chat(self.messages, max_tokens=24))
        self.assertEqual(pieces, ["你", "好"])
        self.assertTrue(captured["body"]["stream"])
        self.assertEqual(captured["body"]["messages"], self.messages)

    def test_stream_parse_errors_are_explicit(self):
        with patch("srt_m15.deepseek_api.request.urlopen", return_value=FakeResponse(b"data: {bad}\n\n")):
            with self.assertRaises(DeepSeekAPIError):
                list(self.client.iter_chat(self.messages))

    def test_key_required_without_network(self):
        client = DeepSeekAPIClient(api_key="", model="deepseek-chat")
        with patch.dict(__import__("os").environ, {"DEEPSEEK_API_KEY": ""}):
            with self.assertRaises(DeepSeekAPIError):
                client.chat(self.messages)


if __name__ == "__main__":
    unittest.main()
