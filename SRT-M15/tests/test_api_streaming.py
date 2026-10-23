"""Purely local verification of real per-token SSE-compatible text generation."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from srt_m15.api_streaming import (
    LocalDecoder, as_prompt, from_staged_checkpoint,
    iter_local_tokens, prepare_local, validate_messages,
)
from srt_m15.language import CharVocabulary, SRTM15LanguageModel
from srt_m15.model import ModelConfig
from srt_m15.supervised_language import Utf8ByteVocabulary


class DeterministicForward:
    """A fake logits-only network: one symbol per inference call, never a cached answer."""
    def __init__(self, token_ids, vocab_size, context=120):
        self.token_ids = list(token_ids)
        self.cfg = SimpleNamespace(max_length=context)
        self.n = 0
        self.vocab_size = vocab_size

    def eval(self):
        return self

    def __call__(self, tokens):
        token_id = self.token_ids[min(self.n, len(self.token_ids) - 1)]
        self.n += 1
        logits = torch.full((1, tokens.size(1), self.vocab_size), -100000.0)
        logits[0, -1, token_id] = 100.0
        return {"logits": logits}


class APIDecodingTests(unittest.TestCase):
    def test_strict_history_validation(self):
        m = lambda r, s: SimpleNamespace(role=r, content=s)
        self.assertFalse(validate_messages([m("user", "hi")]))
        self.assertTrue(validate_messages([m("user", "hi"), m("assistant", "hello"),
                                           m("user", "again")]))
        prompt, multi = as_prompt([m("user", "早上好"), m("assistant", "你好"),
                                   m("user", "还记得吗")])
        self.assertTrue(multi)
        self.assertIn("问：早上好", prompt)
        self.assertIn("答：你好", prompt)
        self.assertIn("问：还记得吗", prompt)
        with self.assertRaises(ValueError):
            validate_messages([m("user", "hi"), m("user", "again")])
        with self.assertRaises(ValueError):
            validate_messages([m("system", "rules"), m("assistant", "wrong"),
                               m("user", "hi")])
        with self.assertRaises(ValueError):
            validate_messages([m("user", "x" * 601)])

    def test_character_decoder_generates_inference_deltas(self):
        vocab = CharVocabulary.from_text("问：你好\n答：")
        ids = [vocab.stoi["你"], vocab.stoi["好"], vocab.stoi["\n"]]
        fake = DeterministicForward(ids, len(vocab.chars))
        dec = LocalDecoder(fake, vocab, "character", qa_training="masked_answer")
        output = list(iter_local_tokens(dec, "你好", max_tokens=7))
        self.assertEqual(output, ["你", "好"])
        self.assertEqual(fake.n, 3)

    def test_utf8_decoder_waits_for_whole_character(self):
        byte_vocab = Utf8ByteVocabulary()
        pieces = list("好".encode("utf-8")) + [10]
        fake = DeterministicForward(pieces, 256)
        decoder = LocalDecoder(fake, byte_vocab, "utf8-bytes-v1")
        answer = list(iter_local_tokens(decoder, "Hi", language="en",
                                        max_tokens=6))
        self.assertEqual(answer, ["好"])

    def test_early_rejection_if_context_or_vocabulary_is_incompatible(self):
        vocab = CharVocabulary.from_text("问：好\n答：")
        fake = DeterministicForward([vocab.stoi["好"]], len(vocab.chars), context=8)
        dec = LocalDecoder(fake, vocab, "character")
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            prepare_local(dec, "unknown")
        with self.assertRaisesRegex(ValueError, "context window"):
            prepare_local(dec, "好" * 10)

    def test_real_staged_checkpoint_loader(self):
        cfg = ModelConfig(vocab_size=12, num_classes=12,
                          max_length=24, dim=16, heads=4,
                          num_experts=2, reasoning_steps=1)
        model = SRTM15LanguageModel(cfg)
        # Valid text vocab to match checkpoint's class count.
        vocab = list("问答：\n你好你是谁a")
        cfg = ModelConfig(vocab_size=len(vocab), num_classes=len(vocab),
                          max_length=24, dim=16, heads=4,
                          num_experts=2, reasoning_steps=1)
        model = SRTM15LanguageModel(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latest.pt"
            torch.save({"format": "srt-m15-bilingual-staged-v1",
                        "config": asdict(cfg), "vocab": vocab,
                        "model": model.state_dict()}, path)
            decoder = from_staged_checkpoint(path)
            self.assertEqual(decoder.provider, "SRT-M15 staged (local)")
            self.assertEqual(decoder.tokenizer_type, "character")
            self.assertEqual(decoder.model.cfg.max_length, 24)


if __name__ == "__main__":
    unittest.main()
