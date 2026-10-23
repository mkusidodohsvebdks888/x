import unittest
from unittest.mock import patch

from srt_m15.chat import ChatReply, SRTChatEngine


class BackendIntegrationTests(unittest.TestCase):
    def test_default_stays_local_offline(self):
        engine = SRTChatEngine()
        self.assertEqual(engine.backend, "local")
        self.assertFalse(hasattr(engine, "api_key"))
        with patch.object(engine, "_external_reply", side_effect=AssertionError("unexpected network")):
            result = engine.reply("1+9")
        self.assertEqual(result.mode, "math")
        self.assertEqual(result.provider, "SRT-M15")

    def test_chat_reply_provider_positional_compatibility(self):
        result = ChatReply("external", "remote", "DeepSeek API")
        self.assertEqual(result.provider, "DeepSeek API")
        self.assertIsNone(result.observed_pair)

    def test_explicit_external_backend_is_labeled(self):
        engine = SRTChatEngine(backend="api")
        with patch.object(engine, "_external_reply", return_value=ChatReply("external", "fake", provider="DeepSeek API")) as mocked:
            result = engine.reply("What is a model?", language="en")
        mocked.assert_called_once_with("What is a model?", "api")
        self.assertEqual(result.provider, "DeepSeek API")

    def test_raw_forces_local_even_with_external_config(self):
        engine = SRTChatEngine(backend="web")
        with patch.object(engine, "_external_reply", side_effect=AssertionError("unexpected network")):
            result = engine.reply("/raw 1+9")
        self.assertEqual(result.mode, "math")

    def test_auto_keeps_addition_local(self):
        engine = SRTChatEngine(backend="auto", external_backend="api")
        with patch.object(engine, "_external_reply", side_effect=AssertionError("unexpected network")):
            result = engine.reply("3+7")
        self.assertEqual(result.mode, "math")

    def test_auto_external_when_appropriate(self):
        engine = SRTChatEngine(backend="auto", external_backend="api")
        with patch.object(engine, "_external_reply", return_value=ChatReply("external", "response", provider="DeepSeek API")) as mocked:
            result = engine.reply("What is a model?", language="en")
        mocked.assert_called_once_with("What is a model?", "api")
        self.assertEqual(result.mode, "external")


if __name__ == "__main__":
    unittest.main()
