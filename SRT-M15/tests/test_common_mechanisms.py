"""Checks the new shared train/evaluation/inference pipeline with real SRT-M15 weights."""
import math
import tempfile
import unittest
from pathlib import Path

import torch

from srt_m15 import ModelConfig
from srt_m15.language import SRTM15LanguageModel
from srt_m15.common_mechanisms import (
    IGNORE_INDEX, LanguageExample, character_vocabulary, evaluate_language,
    instruction_example, make_windows, masked_token_loss, sample_tokens,
    split_examples, train_language_steps,
)
from train_common import load_examples


class CommonMechanismsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(12)
        torch.set_num_threads(2)

    def test_instruction_targets_only_answer(self):
        example = instruction_example("你好", "世界")
        self.assertEqual(example.text[example.target_start:],
                         "世界\n")
        vocab = character_vocabulary([example])
        windows = make_windows([example], vocab, context=8)
        labels = [int(t) for _, y in windows for t in y if int(t) != IGNORE_INDEX]
        self.assertEqual(vocab.decode(labels), "世界\n")

    def test_nonoverlapping_example_split(self):
        examples = [LanguageExample("词条%d" % i) for i in range(15)]
        train, valid = split_examples(examples, validation_fraction=.2, seed=7)
        self.assertEqual(len(valid), 3)
        self.assertEqual(set(e.text for e in train) & set(e.text for e in valid), set())
        self.assertEqual(
            [e.text for e in valid],
            [e.text for e in split_examples(examples, validation_fraction=.2, seed=7)[1]])

    def test_text_windows_targets_padding(self):
        example = LanguageExample("abcde", target_start=3)
        vocab = character_vocabulary([example])
        windows = make_windows([example], vocab, context=2)
        # The first two characters only predict masked prompt labels,
        # so the first window is intentionally dropped.
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0][1].tolist(), [vocab.stoi["d"], vocab.stoi["e"]])
        padded = make_windows([LanguageExample("abcd")],
                              vocab, context=2)
        self.assertEqual(padded[-1][1].tolist(), [vocab.stoi["d"], IGNORE_INDEX])

    def test_masked_loss(self):
        logits = torch.zeros(1, 3, 4, requires_grad=True)
        labels = torch.tensor([[IGNORE_INDEX, 1, 2]])
        loss = masked_token_loss(logits, labels)
        self.assertAlmostEqual(float(loss.item()), math.log(4.0), places=5)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[0, 0] == 0))

    def test_scratch_train_eval_and_sampling(self):
        train = [LanguageExample("abcabc"), LanguageExample("abca")]
        validation = [LanguageExample("bcbcbc")]
        vocab = character_vocabulary(train, validation)
        a = make_windows(train, vocab, context=6)
        b = make_windows(validation, vocab, context=6)
        config = ModelConfig(vocab_size=len(vocab.chars),
                             num_classes=len(vocab.chars),
                             max_length=6, dim=16, heads=4,
                             num_experts=2, reasoning_steps=1, adapter_dim=4)
        model = SRTM15LanguageModel(config)
        old = model.backbone.embed.weight.detach().clone()
        hist = train_language_steps(
            model, a, b, steps=2, batch_size=2, lr=.001, log_every=1)
        self.assertEqual(len(hist), 2)
        self.assertFalse(torch.equal(old, model.backbone.embed.weight))
        metric = evaluate_language(model, b)
        self.assertTrue(math.isfinite(metric["perplexity"]))
        self.assertTrue(0 <= metric["token_accuracy"] <= 1)
        prompt = torch.tensor([vocab.encode("ab")])
        result = sample_tokens(
            model, prompt, max_new_tokens=3, top_k=1, top_p=.8)
        self.assertEqual(tuple(result.shape), (1, 5))

    def test_bad_sampling_input(self):
        config = ModelConfig(vocab_size=4, num_classes=4, dim=16,
                             heads=4, max_length=4, num_experts=2,
                             reasoning_steps=1)
        model = SRTM15LanguageModel(config)
        with self.assertRaises(ValueError):
            sample_tokens(model, torch.ones((1, 1), dtype=torch.long),
                          temperature=0)

    def test_load_examples_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.txt"
            path.write_text("abc\n\nabc", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicated"):
                load_examples("pretrain", path)


if __name__ == "__main__":
    unittest.main()
