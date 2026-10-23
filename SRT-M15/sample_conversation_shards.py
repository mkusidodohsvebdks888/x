"""Extract a small balanced train/dev subset from million-data gzip shards.

Provides a RAM-bounded *local smoke train* input for the pre-existing
train_bilingual_stages.py, not a streaming million-example model trainer.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import random


def collect(root, split, language, limit, seed):
    files = sorted((Path(root) / "shards" / split / language).glob("*.jsonl.gz"))
    if not files:
        raise ValueError(f"No data for {split}/{language}")
    rng = random.Random(seed)
    selected = []
    total = 0
    for file in files:
        with gzip.open(file, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                total += 1
                # Reservoir sampling: stable, no need to load the 1M corpus.
                if len(selected) < limit:
                    selected.append(row)
                else:
                    index = rng.randrange(total)
                    if index < limit:
                        selected[index] = row
    if len(selected) < limit:
        raise ValueError(f"Only {len(selected)} examples available in {split}/{language}; requested {limit}")
    for row in selected:
        row["split"] = split
        # Existing trainer only supports paired bilingual topic IDs.
        # For this independent-language smoke subset, omit topic_id.
        row.pop("topic_id", None)
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="local_data/conversation_million")
    p.add_argument("--output", default="local_data/conversation_smoke.jsonl")
    p.add_argument("--train-per-language", type=int, default=400)
    p.add_argument("--dev-per-language", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.train_per_language < 4 or args.dev_per_language < 1:
        p.error("train-per-language >= 4 and dev-per-language >= 1 are required")
    data = []
    for split, count in (("train", args.train_per_language), ("dev", args.dev_per_language)):
        for lang, offset in (("zh", 0), ("en", 1)):
            data.extend(collect(args.input, split, lang, count,
                                args.seed + offset + (0 if split == "train" else 100)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"path": str(output), "records": len(data),
                      "note": "Local smoke subset only; not full million-sample training."},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
