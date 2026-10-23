"""Checks determinism, source preservation, provenance and family isolation."""
from collections import Counter
import hashlib
from pathlib import Path
import tempfile
import unittest

from bilingual_data import load_records, normalize_prompt, split_records
from generate_bilingual_110k import generate, build


class Bilingual110KTests(unittest.TestCase):
    def test_count_provenance_and_compatibility(self):
        rows, stats = generate()
        self.assertEqual(len(rows), 110_000)
        self.assertEqual(Counter(r["language"] for r in rows),
                         {"zh": 55_000, "en": 55_000})
        self.assertEqual(len({r["topic_id"] for r in rows}), 55_000)
        self.assertEqual(len({normalize_prompt(r["prompt"]) for r in rows}),
                         110_000)
        self.assertEqual(sum(stats["new_paired_topics"].values()), 50_000)
        self.assertTrue(all(r.get("verified") is False for r in rows[10_000:]))
        self.assertTrue(all(r.get("synthetic") is True for r in rows[10_000:]))
        self.assertEqual(rows[0]["topic_id"], "greetings-01")
        self.assertEqual(rows[10_000]["topic_id"], "v4-add-00000")
        self.assertEqual(rows[-1]["topic_id"], "v4-extract-02999")
        self.assertEqual(len({r["family_id"] for r in rows[10_000:]} &
                             {r.get("family_id") for r in rows[:10_000]}), 0)

    def test_write_and_split_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "generated.jsonl"
            manifest = Path(tmp) / "manifest.json"
            info = build(out, manifest=manifest)
            self.assertEqual(info["total_records"], 110_000)
            self.assertEqual(info["added_records"], 100_000)
            self.assertEqual(info["data_sha256"],
                             hashlib.sha256(out.read_bytes()).hexdigest())
            self.assertTrue(manifest.is_file())
            records = load_records(out, allow_demo=True)
            train, dev = split_records(records, seed=42)
            self.assertEqual(len(train)+len(dev), 110_000)
            self.assertGreater(len(train), 50_000)
            self.assertGreater(len(dev), 5_000)
            self.assertFalse({r.topic_id for r in train} &
                             {r.topic_id for r in dev})
            del records, train, dev
            info2 = build(out, manifest=manifest)
            self.assertEqual(info["data_sha256"], info2["data_sha256"])


if __name__ == "__main__":
    unittest.main()
