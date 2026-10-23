"""Local-only adapter tests; no Ollama service or downloads required."""
import io
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from srt_m15.ollama_local import OllamaLocalClient, OllamaLocalError


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class OllamaLocalClientTests(unittest.TestCase):
    def test_valid_chat_sends_only_to_localhost_with_history(self):
        messages = [
            {"role": "system", "content": "You are a local test."},
            {"role": "user", "content": "我叫小明。"},
            {"role": "assistant", "content": "你好，小明！"},
            {"role": "user", "content": "我叫什么？"},
        ]

        def fake_open(request, timeout):
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
            body = json.loads(request.data.decode("utf-8"))
            self.assertEqual(body["model"], "qwen3:4b-instruct")
            self.assertEqual(body["messages"], messages)
            self.assertIs(body["stream"], False)
            self.assertEqual(body["options"]["num_predict"], 256)
            self.assertEqual(body["options"]["num_ctx"], 4096)
            self.assertGreater(timeout, 0)
            return FakeResponse(json.dumps({
                "message": {"role": "assistant", "content": "你叫小明。"},
                "done": True,
            }).encode("utf-8"))

        client = OllamaLocalClient()
        with patch("srt_m15.ollama_local.urlopen", side_effect=fake_open):
            self.assertEqual(client.chat(messages), "你叫小明。")

    def test_error_if_local_server_missing(self):
        with patch("srt_m15.ollama_local.urlopen", side_effect=URLError("refused")):
            with self.assertRaisesRegex(OllamaLocalError, "Cannot reach local Ollama"):
                OllamaLocalClient().chat([{"role": "user", "content": "Hi"}])

    def test_empty_and_malformed_answers_are_not_successes(self):
        for body in ({}, {"message": {"content": ""}}, {"message": {"content": None}}):
            with self.subTest(body=body):
                data = json.dumps(body).encode("utf-8")
                with patch("srt_m15.ollama_local.urlopen",
                           side_effect=lambda *_a, **_kw: FakeResponse(data)):
                    with self.assertRaises(OllamaLocalError):
                        OllamaLocalClient().chat([{"role": "user", "content": "Hi"}])

    def test_rejects_unexpected_message_shapes(self):
        client = OllamaLocalClient()
        for message in ([], [{"role": "server", "content": "Hi"}],
                        [{"role": "user", "content": ""}]):
            with self.assertRaises(ValueError):
                client.chat(message)

    def test_start_local_contains_explicit_model_attribution(self):
        from start_local import HTML_PAGE
        self.assertIn("d.chat_backend==='ollama'", HTML_PAGE)
        self.assertIn("Ollama 本地 Qwen 推理", HTML_PAGE)
        self.assertIn("chatHistory", HTML_PAGE)
        self.assertIn("reset-chat", HTML_PAGE)


if __name__ == "__main__":
    unittest.main()
