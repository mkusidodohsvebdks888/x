"""Offline smoke tests for 1M streaming conversational dataset assembly."""
import argparse
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from build_conversation_million import canonical, extract_pairs, run, split_for


class MillionDataTests(unittest.TestCase):
    def test_conversation_turns_and_grouping(self):
        row = {"messages": [
            {"role": "user", "content": "Where is the key?"},
            {"role": "assistant", "content": "I do not know."},
            {"role": "user", "content": "Is it in the box?"},
            {"role": "assistant", "content": "You have not said where it is."}
        ]}
        examples = list(extract_pairs(row))
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0][0], examples[1][0])
        self.assertIn("I do not know.", examples[1][1])
        self.assertEqual(split_for(examples[0][0], 42),
                         split_for(examples[1][0], 42))

    def test_instruction_formats(self):
        self.assertEqual(list(extract_pairs({
            "instruction": "Greet", "input": "a friend", "output": "Hi"
        }))[0][2], "Hi")
        self.assertEqual(list(extract_pairs({
            "input": "Explain this", "target": "Here is why"
        }))[0][2], "Here is why")
        self.assertEqual(list(extract_pairs({
            "inputs": "请解释什么是计算机", "targets": "计算机用来处理信息。"
        }))[0][2], "计算机用来处理信息。")
        self.assertEqual(canonical("Hello  WORLD"), "hello world")

    def test_local_streaming_shards_and_dedup(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            zh = directory / "zh.jsonl"
            en = directory / "en.jsonl"
            zh.write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in [
                {"prompt": "你好，今天好吗？", "response": "你好，我很好，谢谢。"},
                {"prompt": "你好，今天好吗？", "response": "这是重复的文本。"},
                {"input": "请解释什么是模型", "target": "模型是可计算的表示。"},
                {"input": "这封邮件发给谁？", "target": "没有提供收件人，需要询问。"},
            ]) + "\n", encoding="utf-8")
            en.write_text("\n".join(json.dumps(v) for v in [
                {"prompt": "How are you today?", "response": "I'm well, thanks."},
                {"prompt": "Where is my notebook?", "response": "I don't know the location."},
                {"prompt": "What is a token?", "response": "It's an input unit of the model."},
            ]) + "\n", encoding="utf-8")
            output = directory / "out"
            args = argparse.Namespace(input=[str(zh) + "::zh", str(en) + "::en"],
                                      hf_source=[], accept_source_terms=False,
                                      target=6, zh_target=3, shard_size=2,
                                      max_prompt_chars=200, max_response_chars=200,
                                      output=str(output), seed=42)
            self.assertEqual(run(args), 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["accepted"], 6)
            self.assertEqual(manifest["language_accepted"], {"zh": 3, "en": 3})
            rows = []
            for path in (output / "shards").rglob("*.jsonl.gz"):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    rows.extend(json.loads(line) for line in f)
            self.assertEqual(len(rows), 6)
            self.assertEqual(len({r["prompt"] for r in rows}), 6)
            self.assertTrue(all(r["verified"] is False for r in rows))


if __name__ == "__main__":
    unittest.main()
