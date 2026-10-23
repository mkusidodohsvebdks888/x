"""Standard-library learning-data tests; run without GPU."""
import json
from pathlib import Path
import tempfile
import unittest

from learning_pipeline import (
    VerifiedQA, normalized_question, promotion_gate, read_verified_jsonl,
    split_verified,
)


class VerifiedLearningTests(unittest.TestCase):
    def load_rows(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                            encoding="utf-8")
            return read_verified_jsonl(path)

    def test_only_verified_qa_is_loaded(self):
        rows = [
            {"question": "甲是什么", "answer": "答案甲", "source": "human", "verified": True},
            {"question": "乙是什么", "answer": "答案乙", "source": "model", "verified": False},
            {"question": "丙是什么", "answer": "答案丙", "source": "unknown"},
        ]
        self.assertEqual(len(self.load_rows(rows)), 1)

    def test_normalization_closes_punctuation_leak(self):
        self.assertEqual(normalized_question("你 是 谁？"), normalized_question("你是谁?"))
        self.assertEqual(normalized_question("GPT-6"), normalized_question("gpt６"))

    def test_conflicting_answers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting answer"):
            self.load_rows([
                {"question": "你是谁？", "answer": "答案一", "source": "a", "verified": True},
                {"question": "你 是 谁!", "answer": "答案二", "source": "b", "verified": True},
            ])

    def test_auto_split_by_unseen_questions(self):
        rows = [VerifiedQA("新知识是什么", "知识答复", "manual"),
                VerifiedQA("请解释新知识", "知识答复", "manual"),
                VerifiedQA("新理论是什么", "理论答复", "manual"),
                VerifiedQA("如何理解新理论", "理论答复", "manual"),
                VerifiedQA("仅有的问题", "单次答复", "manual")]
        train, dev = split_verified(rows, seed=42)
        self.assertEqual((len(train), len(dev)), (3, 2))
        self.assertFalse({normalized_question(p.question) for p in train} &
                         {normalized_question(p.question) for p in dev})
        self.assertEqual(split_verified(rows, seed=42), (train, dev))

    def test_manual_heldout_split(self):
        rows = [VerifiedQA("问题A", "答案A", "manual", "train"),
                VerifiedQA("问题B", "答案A", "manual", "dev")]
        train, dev = split_verified(rows)
        self.assertEqual(train[0].question, "问题A")
        self.assertEqual(dev[0].question, "问题B")

    def test_gate_rejects_forgetting(self):
        before = {"old_exact": 4, "old_loss": .4, "new_exact": 0,
                  "new_loss": 1.0, "new_total": 2}
        worse_old = dict(before, old_exact=3, new_exact=1)
        ok, reason = promotion_gate(before, worse_old)
        self.assertFalse(ok)
        self.assertIn("旧任务", reason)

    def test_gate_only_promotes_real_heldout_progress(self):
        before = {"old_exact": 4, "old_loss": .4, "new_exact": 0,
                  "new_loss": 1.0, "new_total": 2}
        improved = dict(before, new_exact=1, new_loss=.7)
        self.assertTrue(promotion_gate(before, improved)[0])
        self.assertFalse(promotion_gate(before, before)[0])
        # Slightly lower loss with zero new exact matches is not mastery.
        self.assertFalse(promotion_gate(before, dict(before, new_loss=.5))[0])
        missing = dict(improved, new_total=0)
        self.assertFalse(promotion_gate(before, missing)[0])


if __name__ == "__main__":
    unittest.main()