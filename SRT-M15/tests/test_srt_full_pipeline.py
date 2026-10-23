"""Offline checks for SRT-M15 complete teacher->review->student workflow."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from srt_full_pipeline import batch_filename, run
from merge_reviewed_sft import merge
from qwen_teacher_data import planned


class FakeTeacher:
    model = "qwen3:4b-instruct"

    def chat(self, messages, *, max_tokens=220):
        return ("这是一个简短的示例答案，请在学习过程中验证。"
                if "中文" in messages[0]["content"]
                else "This is a short teaching example. Please check it before training.")


class PipelineTests(unittest.TestCase):
    def test_unique_pair_offsets(self):
        self.assertNotEqual(planned(0), planned(40))
        self.assertTrue(str(batch_filename(40, 20)).endswith("batch_000040_000060.jsonl"))

    def test_student_training_rejects_unreviewed_data_before_subprocess(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "unreviewed.jsonl"
            data.write_text(json.dumps({"language": "zh", "prompt": "你好",
                                        "response": "你好", "verified": False,
                                        "source": "local-ollama:test"}) + "\n",
                            encoding="utf-8")
            args = argparse.Namespace(action="train", data=data, device="cpu",
                                      output=Path(folder)/"ckpt",
                                      steps=1, stages=1, context=32,
                                      batch_size=2, dim=16, resume=False)
            with patch("srt_full_pipeline.invoke") as called:
                with self.assertRaises(ValueError):
                    run(args)
                called.assert_not_called()

    def test_merge_approved_batches_and_refuse_duplicate_questions(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            part1, part2 = directory / "p1.jsonl", directory / "p2.jsonl"
            all_rows = []
            for language in ("zh", "en"):
                for index in range(6):
                    split = "dev" if index in (0, 5) else "train"
                    prompt = (f"请解释常见概念{index}。" if language == "zh"
                              else f"Explain a common concept {index}.")
                    answer = (f"这是对于概念{index}的简单解释。" if language == "zh"
                              else f"This is a short explanation of concept {index}.")
                    all_rows.append({
                        "language": language, "prompt": prompt, "response": answer,
                        "verified": True, "review_status": "human-reviewed",
                        "source": "human-reviewed-local-ollama:qwen3:4b-instruct:test",
                        "family_id": f"subject-{index}", "split": split,
                    })
            part1.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n"
                                     for row in all_rows[:6]), encoding="utf-8")
            part2.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n"
                                     for row in all_rows[6:]), encoding="utf-8")
            output = directory / "joined.jsonl"
            summary = merge([part1, part2], output)
            self.assertEqual(summary["records"], 12)
            self.assertEqual(summary["languages"], {"zh": 6, "en": 6})
            self.assertEqual(summary["train"], {"zh": 4, "en": 4})
            self.assertEqual(summary["dev"], {"zh": 2, "en": 2})
            self.assertTrue(output.is_file())
            with self.assertRaisesRegex(ValueError, "overwrite"):
                merge([part1, part2], output)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                merge([part1, part1], directory / "duplicate.jsonl")

    def test_merge_does_not_accept_unreviewed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder)/"source.jsonl", Path(folder)/"final.jsonl"
            source.write_text(json.dumps({"language": "zh", "prompt": "你好",
                                          "response": "你也好", "verified": False,
                                          "review_status": "unreviewed",
                                          "source": "local-ollama:qwen3:4b-instruct"},
                                         ensure_ascii=False)+"\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                merge([source], output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
