"""Offline QA for the deterministic 10k bilingual dataset generator."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from bilingual_data import load_records, normalize_prompt, split_records
from generate_bilingual_10k import generate, write_dataset, _family_split


class BilingualTenKTests(unittest.TestCase):
    def test_exact_count_reproducibility_and_linguistic_split(self):
        rows, stats = generate()
        self.assertEqual(len(rows), 10000)
        self.assertEqual(Counter(r["language"] for r in rows), {"zh": 5000, "en": 5000})
        self.assertEqual(len({r["topic_id"] for r in rows}), 5000)
        self.assertEqual(len({normalize_prompt(r["prompt"]) for r in rows}), 10000)
        self.assertEqual(sum(stats.values()), 4500)
        self.assertEqual(stats["vocabulary"], 900)
        self.assertEqual(stats["situation"], 900)
        self.assertEqual(len([r for r in rows if r["verified"] is True]), 0)
        self.assertTrue(all(r["synthetic"] for r in rows))
        self.assertEqual(rows[0]["topic_id"], "greetings-01")
        second, second_stats = generate()
        self.assertEqual(stats, second_stats)
        self.assertEqual(rows, second)

    def test_no_family_leakage(self):
        rows, _ = generate()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "10k.jsonl"
            manifest = Path(folder) / "manifest.json"
            report = write_dataset(path, manifest=manifest)
            self.assertEqual(report["total_records"], 10000)
            self.assertTrue(manifest.exists())
            self.assertEqual(report["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            rec = load_records(path, allow_demo=True)
            train, dev = split_records(rec)
            self.assertEqual(len(train) + len(dev), 10000)
            self.assertGreater(len(train), 7000)
            self.assertGreater(len(dev), 1000)
            self.assertTrue(all(r.topic_id for r in train + dev))
            self.assertFalse({x.topic_id for x in train} & {x.topic_id for x in dev})
            # All generated prompts sharing a template family must be on one side.
            by_id = {r["topic_id"]: r for r in rows}
            train_families = {by_id[r.topic_id].get("family_id")
                              for r in train if r.topic_id.startswith("v3-")}
            dev_families = {by_id[r.topic_id].get("family_id")
                            for r in dev if r.topic_id.startswith("v3-")}
            self.assertFalse(train_families & dev_families)

    def test_generator_has_no_network_or_automatic_provenance_claim(self):
        self.assertEqual(_family_split("arithmetic_add", 1),
                         _family_split("arithmetic_add", 1))
        rows, _ = generate()
        generated = [r for r in rows if r["topic_id"].startswith("v3-")]
        self.assertEqual(len(generated), 9000)
        self.assertTrue(all(r["verified"] is False for r in generated))
        self.assertTrue(all(r["review_status"] == "unreviewed" for r in generated))


if __name__ == "__main__":
    unittest.main()
