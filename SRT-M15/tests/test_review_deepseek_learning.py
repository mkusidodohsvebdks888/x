"""Offline tests for explicit DeepSeek draft human-review pipeline."""
import json
from pathlib import Path
import tempfile
import unittest

from review_deepseek_learning import (
    candidate_issues, draft_fingerprint, draft_index, review_index,
    add_review, get_approved_rows, export_dataset,
)
from diagnose_local_gaps import overlap_f1, tokens


class HumanReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.drafts = self.dir / "drafts.jsonl"
        self.reviews = self.dir / "reviews.jsonl"
        # Five examples per language are needed by the real validator.
        self.rows = []
        for lang in ("zh", "en"):
            for i in range(5):
                prompt = f"请解释数字{i}的作用？" if lang == "zh" else f"Explain number {i} please?"
                answer = f"这是一个关于数字{i}的简单示例。" if lang == "zh" else f"This is a simple example for number {i}."
                self.rows.append({
                    "id": f"{lang}-{i}", "language": lang,
                    "prompt": prompt, "response": answer,
                    "verified": False, "review_status": "unreviewed",
                    "source": "deepseek-web-interactive-session",
                    "answer_truncated": False,
                })
        self._write_drafts()

    def _write_drafts(self):
        self.drafts.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                        for r in self.rows), encoding="utf-8")

    def _approve(self, id_, **extras):
        return add_review(drafts=self.drafts, reviews=self.reviews,
                          identity=id_, decision="approve",
                          evidence="Human checked this against an independent reliable source.",
                          rights_confirmed=True, **extras)

    def test_drafts_start_unreviewed_and_require_consent(self):
        self.assertEqual(len(draft_index(self.drafts)), 10)
        self.assertEqual(len(review_index(self.reviews)), 0)
        with self.assertRaisesRegex(ValueError, "rights"):
            add_review(drafts=self.drafts, reviews=self.reviews,
                       identity="zh-0", decision="approve",
                       evidence="Independently checked correctness")
        with self.assertRaisesRegex(ValueError, "evidence"):
            add_review(drafts=self.drafts, reviews=self.reviews,
                       identity="zh-0", decision="approve",
                       rights_confirmed=True, evidence="")
        self.assertFalse(self.reviews.exists())

    def test_explicit_decision_records_and_fingerprint_revocation(self):
        self._approve("zh-0")
        accepted, _ = get_approved_rows(self.drafts, self.reviews)
        self.assertEqual(len(accepted), 1)
        self.assertTrue(accepted[0]["verified"])
        self.assertEqual(accepted[0]["source"], "human-reviewed-deepseek-web:zh-0")
        self.rows[0]["response"] = "已经被篡改的草稿。"
        self._write_drafts()
        accepted, skipped = get_approved_rows(self.drafts, self.reviews)
        self.assertEqual(accepted, [])
        self.assertEqual(skipped["draft-changed"], 1)

    def test_latest_rejection_overrides_approval(self):
        self._approve("zh-0")
        add_review(drafts=self.drafts, reviews=self.reviews, identity="zh-0",
                   decision="reject", reason="Could not verify the original claim")
        accepted, skipped = get_approved_rows(self.drafts, self.reviews)
        self.assertFalse(accepted)
        self.assertEqual(skipped["rejected"], 1)

    def test_possible_pii_blocks_approval(self):
        self.rows[0]["response"] = "请写信到 name@example.com."
        self._write_drafts()
        self.assertIn("possible-personal-information", candidate_issues(self.rows[0]))
        with self.assertRaisesRegex(ValueError, "possible-personal-information"):
            self._approve("zh-0")

    def test_truncated_answer_blocks_approval(self):
        self.rows[0]["answer_truncated"] = True
        self._write_drafts()
        with self.assertRaisesRegex(ValueError, "web-answer-truncated"):
            self._approve("zh-0")

    def test_human_edited_answer_file(self):
        answer_file = self.dir / "edited.txt"
        answer_file.write_text("我独立核验并改写了这段较短的解释。", encoding="utf-8")
        self._approve("zh-0", answer_file=answer_file)
        approved, _ = get_approved_rows(self.drafts, self.reviews)
        self.assertEqual(approved[0]["response"], answer_file.read_text(encoding="utf-8"))

    def test_export_enforces_original_data_loader_and_holdout(self):
        output = self.dir / "reviewed.jsonl"
        with self.assertRaisesRegex(ValueError, "No human-approved"):
            export_dataset(drafts=self.drafts, reviews=self.reviews, output=output)
        self.assertFalse(output.exists())
        for row in self.rows:
            self._approve(row["id"])
        result = export_dataset(drafts=self.drafts, reviews=self.reviews, output=output)
        self.assertEqual(result["exported"], 10)
        self.assertEqual(result["languages"], {"zh": 5, "en": 5})
        self.assertTrue(result["train"] > 0 and result["dev"] > 0)
        saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(saved), 10)
        self.assertTrue(all(s["verified"] is True for s in saved))

    def test_prompt_dedup_on_export(self):
        self.rows[1]["prompt"] = self.rows[0]["prompt"]
        self._write_drafts()
        self._approve("zh-0")
        self._approve("zh-1")
        approved, rejected = get_approved_rows(self.drafts, self.reviews)
        self.assertEqual(len(approved), 1)
        self.assertEqual(rejected["duplicate-question"], 1)


class GapTriageTests(unittest.TestCase):
    def test_language_and_score(self):
        self.assertIn("机", tokens("机器学习"))
        self.assertEqual(overlap_f1("a b", "a b"), 1.0)
        self.assertEqual(overlap_f1("", "hello"), 0.0)
        self.assertLess(overlap_f1("not correct", "correct"), 1.0)


if __name__ == "__main__":
    unittest.main()
