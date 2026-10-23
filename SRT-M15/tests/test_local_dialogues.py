"""Smoke checks for the offline local-dialogue dataset generator."""
import json
import tempfile
from collections import Counter
from pathlib import Path
import unittest

from bilingual_data import load_records, normalize_prompt, split_records
from generate_local_dialogues import generate


class LocalDialogueDataTests(unittest.TestCase):
    def test_unique_paired_examples_and_heldout_situations(self):
        rows, categories = generate()
        self.assertGreaterEqual(len(rows), 2000)
        self.assertEqual(Counter(r["language"] for r in rows)["zh"],
                         Counter(r["language"] for r in rows)["en"])
        self.assertEqual(len({normalize_prompt(r["prompt"]) for r in rows}),
                         len(rows))
        self.assertTrue(all(r["synthetic"] and not r["verified"] for r in rows))
        self.assertGreaterEqual(len(categories), 8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dialogues.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                    for r in rows), encoding="utf-8")
            train, dev = split_records(load_records(path, allow_demo=True))
            self.assertTrue(train)
            self.assertTrue(dev)
            self.assertFalse({r.topic_id for r in train} &
                             {r.topic_id for r in dev})
            # Same scenario and paraphrase variations must not land on both sides.
            topic_to_family = {r["topic_id"]: r["family_id"] for r in rows}
            self.assertFalse({topic_to_family[r.topic_id] for r in train} &
                             {topic_to_family[r.topic_id] for r in dev})


if __name__ == "__main__":
    unittest.main()
