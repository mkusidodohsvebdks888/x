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

    def test_parallel_synthetic_data_never_leaks_topics(self):
        path = Path(__file__).resolve().parents[1] / "examples/bilingual_synthetic_v1.jsonl"
        records = load_records(path, allow_demo=True)
        self.assertEqual(language_counts(records), {"zh": 130, "en": 130})
        self.assertEqual(len({r.topic_id for r in records}), 130)
        train, dev = split_records(records, seed=42)
        self.assertEqual(language_counts(train), {"zh": 104, "en": 104})
        self.assertEqual(language_counts(dev), {"zh": 26, "en": 26})
        self.assertFalse({r.topic_id for r in train} & {r.topic_id for r in dev})
        self.assertFalse({normalize_prompt(r.prompt) for r in train} &
                         {normalize_prompt(r.prompt) for r in dev})
        self.assertEqual(
            [(r.topic_id, r.language) for r in dev],
            [(r.topic_id, r.language) for r in split_records(records, seed=42)[1]],
        )

    def test_v2_expansion_integrity_and_topic_holdout(self):
        root = Path(__file__).resolve().parents[1] / "examples"
        v1 = load_records(root / "bilingual_synthetic_v1.jsonl", allow_demo=True)
        v2 = load_records(root / "bilingual_synthetic_v2_1000.jsonl", allow_demo=True)
        self.assertEqual(len(v2), 1000)
        self.assertEqual(language_counts(v2), {"zh": 500, "en": 500})
        self.assertEqual(len({r.topic_id for r in v2}), 500)
        # Previously recorded examples must remain intact in the expanded
        # dataset. No old training split can be quietly mutated.
        self.assertEqual(v1, v2[:len(v1)])
        self.assertTrue(all(x.source and x.topic_id for x in v2))
        self.assertTrue(all(x.split == "auto" for x in v2))
        self.assertTrue(all("synthetic" in x.source for x in v2))
        train, dev = split_records(v2, seed=42)
        self.assertEqual(language_counts(train), {"zh": 400, "en": 400})
        self.assertEqual(language_counts(dev), {"zh": 100, "en": 100})
        self.assertFalse({r.topic_id for r in train} & {r.topic_id for r in dev})
        self.assertFalse({normalize_prompt(r.prompt) for r in train} &
                         {normalize_prompt(r.prompt) for r in dev})

    def test_paired_data_rejects_conflicting_splits(self):
        rows = [
            {"language": "zh", "prompt": f"问题{i}", "response": f"回答{i}",
             "topic_id": f"topic{i}", "split": "train"}
            for i in range(4)
        ] + [
            {"language": "en", "prompt": f"question{i}", "response": f"answer{i}",
             "topic_id": f"topic{i}", "split": "dev" if i == 0 else "train"}
            for i in range(4)
        ]
        with self.assertRaisesRegex(ValueError, "conflicting"):
            split_records(load_records(self.make_file(rows), allow_demo=True))

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
