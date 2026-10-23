"""Pure-Python tests: multilingual provenance and leak-resistant splitting."""
import json
from pathlib import Path
import tempfile
import unittest

from bilingual_data import (
    BilingualRecord, language_counts, load_records, normalize_prompt, split_records,
)
from finetune_bilingual import eligible


class BilingualPipelineTests(unittest.TestCase):
    def make_file(self, rows):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "data.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
        return path

    def test_demo_requires_explicit_opt_in(self):
        path = Path(__file__).resolve().parents[1] / "examples/bilingual_toy_sft.jsonl"
        with self.assertRaisesRegex(ValueError, "verified"):
            load_records(path)
        records = load_records(path, allow_demo=True)
        self.assertEqual(language_counts(records), {"zh": 20, "en": 20})
        train, dev = split_records(records)
        self.assertEqual(set(train) & set(dev), set())
        self.assertGreaterEqual(language_counts(dev)["zh"], 1)
        self.assertGreaterEqual(language_counts(dev)["en"], 1)

    def test_accepted_only_when_verified_and_provenance_present(self):
        rows = []
        for lang in ("en", "zh"):
            for i in range(5):
                rows.append({"language": lang, "prompt": f"{lang} question {i}",
                             "response": f"answer {i}", "source": "human-review",
                             "verified": True})
        train, dev = split_records(load_records(self.make_file(rows)))
        self.assertEqual(language_counts(train), {"zh": 4, "en": 4})
        self.assertEqual(language_counts(dev), {"zh": 1, "en": 1})

    def test_punctuation_normalization_and_duplicates(self):
        self.assertEqual(normalize_prompt("ＨＥＬＬＯ！"), normalize_prompt("hello"))
        rows = [
            {"language": "en", "prompt": "Hello!", "response": "hi"},
            {"language": "en", "prompt": "hello", "response": "hello"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_records(self.make_file(rows), allow_demo=True)

    def test_review_source_must_be_set(self):
        data = [{"language": "zh", "prompt": "你好", "response": "你好",
                 "verified": True, "source": ""}]
        with self.assertRaisesRegex(ValueError, "source"):
            load_records(self.make_file(data))

    def test_manual_holdout_requires_both_sides(self):
        rows = [BilingualRecord("zh", f"问题{i}", "回答", "reviewed", "train")
                for i in range(4)]
        rows += [BilingualRecord("en", f"question{i}", "response", "reviewed", "train")
                 for i in range(4)]
        with self.assertRaisesRegex(ValueError, "needs independent"):
            split_records(rows)

    def test_eval_gate_per_language(self):
        baseline = {"zh": {"loss": 2.0, "tokens": 10}, "en": {"loss": 2.0, "tokens": 10}}
        pass_result = {"zh": {"loss": 1.8, "tokens": 10}, "en": {"loss": 2.02, "tokens": 10}}
        self.assertTrue(eligible(baseline, pass_result)[0])
        regress_result = {"zh": {"loss": 1.6, "tokens": 10}, "en": {"loss": 2.2, "tokens": 10}}
        self.assertFalse(eligible(baseline, regress_result)[0])
        no_gain = {"zh": {"loss": 2.0, "tokens": 10}, "en": {"loss": 2.0, "tokens": 10}}
        self.assertFalse(eligible(baseline, no_gain)[0])


if __name__ == "__main__":
    unittest.main()
