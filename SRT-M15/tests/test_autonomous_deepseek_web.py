"""Offline tests for bounded SRT-M15 DeepSeek Web questioning.

Tests never start a browser and never contact DeepSeek.
"""
import json
from pathlib import Path
import tempfile
import unittest

from autonomous_deepseek_web import (
    Question, parse_topics, question_at, load_gap_plan, run, validate_limits,
)


class FakeDeepSeek:
    def __init__(self, *, fail_after=None):
        self.questions = []
        self.fail_after = fail_after

    def ask(self, question):
        if self.fail_after is not None and len(self.questions) >= self.fail_after:
            raise RuntimeError("website response unknown")
        self.questions.append(question)
        return f"An unverified example reply for {len(self.questions)}."


class PoorProposer:
    def propose(self, fallback, language):
        return fallback


class AutonomousDeepSeekTests(unittest.TestCase):
    def test_topics_are_distinct_and_bounded(self):
        self.assertEqual(parse_topics("AI,AI，Python\n英语"), ["AI", "Python", "英语"])
        with self.assertRaises(ValueError):
            parse_topics("")
        with self.assertRaises(ValueError):
            parse_topics(",".join(str(i) for i in range(11)))
        with self.assertRaises(ValueError):
            parse_topics("X" * 61)

    def test_research_plan_and_language(self):
        topics = ["数学", "历史"]
        self.assertIn("数学", question_at(topics, 0).text)
        self.assertIn("历史", question_at(topics, 8).text)
        self.assertIn("math", question_at(["math"], 0, "en").text)
        self.assertEqual(question_at(topics, 9).origin, "rule-based-planner")
        with self.assertRaises(ValueError):
            question_at(topics, 1, "fr")

    def test_rejects_unbounded_runs(self):
        for args in ((21, 30, 15), (2, 1, 15), (2, 30, 61)):
            with self.assertRaises(ValueError):
                validate_limits(*args)

    def test_creates_drafts_without_approving_training(self):
        clock = [100.0]
        def fake_sleep(seconds):
            clock[0] += seconds

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "drafts.jsonl"
            client = FakeDeepSeek()
            messages = []
            count = run(
                client=client, topics=["人工智能"], language="zh",
                max_questions=3, interval=20, max_minutes=5,
                output=path, proposer=PoorProposer(), sleep=fake_sleep,
                now=lambda: clock[0], echo=messages.append,
            )
            self.assertEqual(count, 3)
            self.assertEqual(len(client.questions), 3)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({row["prompt"] for row in rows}), 3)
            self.assertEqual(len({row["session_id"] for row in rows}), 1)
            self.assertTrue(all(not row["verified"] and
                                row["review_status"] == "unreviewed" and
                                row["training_approved"] is False
                                for row in rows))
            self.assertTrue(all(row["provider"].startswith("DeepSeek Web") for row in rows))

    def test_local_gap_question_queue_requires_valid_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            gaps = Path(folder) / "gaps.jsonl"
            rows = [
                {"language": "zh", "origin": "local-staged-evaluation",
                 "review_status": "candidate-only",
                 "prompt": "请说明变压器与注意力机制的区别是什么？"},
                {"language": "en", "origin": "local-staged-evaluation",
                 "review_status": "candidate-only",
                 "prompt": "What is a transformer model?"},
            ]
            gaps.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                    for row in rows), encoding="utf-8")
            plan = load_gap_plan(gaps, language="zh", limit=10)
            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0].origin, "local-staged-evaluation-candidate")
            client = FakeDeepSeek()
            out = Path(folder) / "drafts.jsonl"
            self.assertEqual(run(client=client, topics=["unused"],
                                 language="zh", max_questions=5, interval=20,
                                 max_minutes=5, output=out, plan=plan,
                                 sleep=lambda _: None, now=lambda: 0,
                                 echo=lambda _: None), 1)
            self.assertEqual(client.questions, [rows[0]["prompt"]])
            rows[0]["prompt"] = "Send to me@example.com quickly"
            gaps.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "private information"):
                load_gap_plan(gaps, language="zh", limit=10)

    def test_stops_on_ambiguous_website_error_without_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "drafts.jsonl"
            client = FakeDeepSeek(fail_after=1)
            count = run(
                client=client, topics=["math"], language="en",
                max_questions=3, interval=20, max_minutes=5,
                output=path, sleep=lambda _: None, now=lambda: 0,
                echo=lambda _: None,
            )
            self.assertEqual(count, 1)
            self.assertEqual(len(client.questions), 1)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
