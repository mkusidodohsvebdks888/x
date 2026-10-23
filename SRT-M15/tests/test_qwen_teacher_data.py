"""Offline tests for Qwen output-distillation pipeline; no Ollama calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qwen_teacher_data import make, planned, reject_reason, teacher_messages
from bilingual_data import load_records, split_records
from review_deepseek_learning import add_review, get_approved_rows


class FakeOllama:
    model = "qwen3:4b-instruct"

    def __init__(self):
        self.calls = []

    def chat(self, messages, *, max_tokens=220):
        self.calls.append((messages, max_tokens))
        language = "zh" if "中文" in messages[0]["content"] else "en"
        if language == "zh":
            return "你可以先了解基础概念，然后用一个小练习巩固知识。"
        return "Start with the basic concepts, then practice with one concrete example."


class QwenTeacherDataTests(unittest.TestCase):
    def test_planned_prompts_distinct(self):
        self.assertNotEqual(planned(0)[1:], planned(1)[1:])
        self.assertNotEqual(planned(0)[1:], planned(20)[1:])
        self.assertEqual(len({planned(i)[1] + planned(i)[2]
                              for i in range(200)}), 200)

    def test_rejection_rules(self):
        self.assertEqual(reject_reason("请回复", "请回复", 100), "echo")
        self.assertEqual(reject_reason("What is x?", "email jane@example.com", 100),
                         "possible-private-data")
        self.assertEqual(reject_reason("Tell me", "I am Qwen", 100),
                         "wrong-model-identity")
        self.assertEqual(reject_reason("What is x?", "This explanation is fine.", 100), "")
        self.assertEqual(teacher_messages("en", "hello")[-1]["content"], "hello")

    def test_locally_generated_bilingual_data_requires_review(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "teacher.jsonl"
            fake = FakeOllama()
            summary = make(fake, pairs=20, output=output, log=lambda _: None)
            self.assertEqual(summary["records"], 40)
            self.assertEqual(summary["accepted_pairs"], 20)
            self.assertEqual(len(fake.calls), 40)
            self.assertTrue(output.with_suffix(".manifest.json").exists())
            rows = [json.loads(line) for line in output.read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual({r["source"] for r in rows},
                             {"local-ollama:qwen3:4b-instruct"})
            self.assertTrue(all(not r["verified"] and not r["training_approved"]
                                for r in rows))
            with self.assertRaises(ValueError):
                load_records(output, allow_demo=False)
            train, dev = split_records(load_records(output, allow_demo=True))
            self.assertEqual(len(train) + len(dev), 40)
            self.assertFalse({r.topic_id for r in train} & {r.topic_id for r in dev})
            with self.assertRaisesRegex(ValueError, "already exists"):
                make(fake, pairs=20, output=output, log=lambda _: None)

    def test_local_ollama_draft_is_not_misattributed_to_deepseek(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "drafts.jsonl"
            reviews = Path(folder) / "reviews.jsonl"
            make(FakeOllama(), pairs=20, output=path, log=lambda _: None)
            add_review(drafts=path, reviews=reviews,
                       identity="qwen-pair-000000-zh", decision="approve",
                       evidence="Independently reviewed the accuracy and content provenance.",
                       rights_confirmed=True)
            rows, _ = get_approved_rows(path, reviews)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source"],
                             "human-reviewed-local-ollama:qwen3:4b-instruct:qwen-pair-000000-zh")


if __name__ == "__main__":
    unittest.main()
