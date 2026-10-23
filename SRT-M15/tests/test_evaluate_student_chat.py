"""Test acceptance reporting without loading real weights or contacting Ollama."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluate_student_chat import evaluate


class StudentAcceptanceTests(unittest.TestCase):
    def test_human_review_is_pending_even_when_generation_succeeds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            input_file = root / "test_cases.jsonl"
            output_file = root / "report.json"
            scenarios = [
                {"language": "zh", "prompt": "你好，你是谁？", "check": "说清楚自己的来源"},
                {"language": "en", "prompt": "Hello, what can you do?",
                 "check": "Respond in natural English"},
            ]
            input_file.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n"
                                          for c in scenarios), encoding="utf-8")
            with patch("chat_staged.load_checkpoint", return_value=(object(), object())):
                with patch("chat_staged.reply", side_effect=[
                    ("你好，我可以提供帮助。", None),
                    (None, "unknown-character"),
                ]):
                    report = evaluate(checkpoint="dummy.pt", cases=input_file,
                                      output=output_file, device="cpu", max_tokens=10)
            self.assertEqual(report["cases"], 2)
            self.assertEqual(report["nonempty_output"], 1)
            self.assertEqual([r["human_judgement"] for r in report["results"]],
                             ["pending", "pending"])
            self.assertTrue(report["results"][1]["generation_error"])
            self.assertEqual(json.loads(output_file.read_text(encoding="utf-8"))["cases"], 2)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                with patch("chat_staged.load_checkpoint", return_value=(object(), object())):
                    with patch("chat_staged.reply", return_value=("", None)):
                        evaluate(checkpoint="dummy.pt", cases=input_file,
                                 output=output_file, device="cpu", max_tokens=10)

    def test_bad_cases_do_not_generate_false_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cases = root / "cases.jsonl"
            output = root / "report.json"
            cases.write_text('{"language":"en","prompt":"Hi"}\n', encoding="utf-8")
            with patch("chat_staged.load_checkpoint", return_value=(object(), object())):
                with self.assertRaisesRegex(ValueError, "Invalid evaluation scenario"):
                    evaluate(checkpoint="dummy.pt", cases=cases,
                             output=output, device="cpu")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
