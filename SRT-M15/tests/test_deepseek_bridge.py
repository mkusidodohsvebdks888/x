"""Offline routing checks; these tests never sign in or contact DeepSeek."""
import os
import tempfile
import unittest
from unittest.mock import patch

from srt_m15.chat import ChatReply, SRTChatEngine
from srt_m15.deepseek_api import DeepSeekAPIClient, DeepSeekAPIError
from srt_m15.deepseek_web import DeepSeekWebClient, DeepSeekWebError


class ExternalRoutingTests(unittest.TestCase):
    def test_explicit_website_labels_source(self):
        with tempfile.TemporaryDirectory() as root:
            engine = SRTChatEngine(root=root, backend="web")
            fake = ChatReply("external", "由外部模型生成的回答", "DeepSeek Web")
            with patch.object(engine, "_external_reply", return_value=fake) as send:
                result = engine.reply("什么是神经网络")
            self.assertEqual(result.provider, "DeepSeek Web")
            send.assert_called_once_with("什么是神经网络", "web")

    def test_default_does_not_call_external_service(self):
        with tempfile.TemporaryDirectory() as root:
            engine = SRTChatEngine(root=root)
            with patch.object(engine, "_external_reply", side_effect=AssertionError("called")):
                result = engine.reply("你是谁")
            self.assertEqual(result.provider, "SRT-M15")
            self.assertEqual(result.mode, "notice")  # no checkpoint yet

    def test_raw_forces_local_even_if_external_selected(self):
        with tempfile.TemporaryDirectory() as root:
            engine = SRTChatEngine(root=root, backend="web")
            with patch.object(engine, "_external_reply", side_effect=AssertionError("called")):
                result = engine.reply("/raw 你是谁")
            self.assertEqual(result.provider, "SRT-M15")

    def test_auto_identity_stays_local(self):
        with tempfile.TemporaryDirectory() as root:
            engine = SRTChatEngine(root=root, backend="auto", external_backend="web")
            with patch.object(engine, "_external_reply", side_effect=AssertionError("called")):
                result = engine.reply("你是谁")
            self.assertEqual(result.provider, "SRT-M15")

    def test_auto_general_question_can_use_web(self):
        with tempfile.TemporaryDirectory() as root:
            engine = SRTChatEngine(root=root, backend="auto", external_backend="web")
            with patch.object(engine, "_external_reply", return_value=ChatReply(
                "external", "回答", "DeepSeek Web"
            )) as send:
                result = engine.reply("什么是神经网络")
            send.assert_called_once()
            self.assertEqual(result.provider, "DeepSeek Web")

    def test_colab_cannot_drive_logged_in_phone_browser(self):
        with patch.dict(os.environ, {"COLAB_RELEASE_TAG": "test"}):
            client = DeepSeekWebClient()
            try:
                with self.assertRaises(DeepSeekWebError):
                    client._check_environment()
            finally:
                client.close()

    def test_api_key_is_required_in_api_mode(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            with self.assertRaises(DeepSeekAPIError):
                DeepSeekAPIClient().ask("你好")


if __name__ == "__main__":
    unittest.main()
