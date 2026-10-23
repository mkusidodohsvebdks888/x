"""Merge HUMAN-APPROVED Chinese/English SFT batches, preserving holdouts.

Never upgrades unreviewed teacher output. This is not a million-row streaming
trainer; inputs are intended to be modest local reviewed batches.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from bilingual_data import load_records, normalize_prompt, split_records


def merge(input_files, output):
    if not input_files:
        raise ValueError("Supply at least one reviewed input file")
    destination = Path(output)
    sources = [Path(p) for p in input_files]
    if any(p.resolve() == destination.resolve() for p in sources):
        raise ValueError("Do not overwrite source data")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Refusing to overwrite output: " + str(destination))
    merged = []
    ids = set()
    by_prompt = {}
    families = {}
    for source in sources:
        # Strict loader requires verified=true and explicit source.
        load_records(source, allow_demo=False)
        with source.open("r", encoding="utf-8") as reader:
            for line in reader:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("review_status") != "human-reviewed":
                    raise ValueError("Reviewed data required; do not merge auto-unverified drafts")
                if row.get("source", "").startswith("local-ollama:"):
                    raise ValueError("Unreviewed teacher data cannot enter strict merged SFT")
                if row.get("source", "").startswith("deepseek-web-interactive-session"):
                    raise ValueError("Unreviewed external answers cannot enter strict merged SFT")
                key = normalize_prompt(row["prompt"])
                if key in by_prompt:
                    raise ValueError(f"Duplicate prompt in {source}; earlier in {by_prompt[key]}")
                by_prompt[key] = str(source)
                family = row.get("family_id")
                split = row.get("split", "auto")
                if split not in ("train", "dev", "auto"):
                    raise ValueError("Unsupported split")
                if isinstance(family, str) and family:
                    if family in families and families[family] != split:
                        raise ValueError(f"Family {family} split conflict: {families[family]} vs {split}")
                    families[family] = split
                merged.append(row)
    if not merged:
        raise ValueError("No reviewed data")
    dest_dir = destination.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                      for row in merged)
    fd, temporary = tempfile.mkstemp(dir=dest_dir, prefix=".merged-", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as writer:
            writer.write(payload)
        validated = load_records(temporary, allow_demo=False)
        train, dev = split_records(validated)
        # Existing language trainer always requires both languages.
        for lang in ("zh", "en"):
            if not any(r.language == lang for r in train):
                raise ValueError(f"No {lang} training examples")
            if not any(r.language == lang for r in dev):
                raise ValueError(f"No {lang} heldout examples")
        if destination.exists():
            raise ValueError("Output file was created concurrently")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    manifest = {
        "format": "srt-m15-human-reviewed-sft-merged-v1",
        "inputs": [str(p) for p in sources], "records": len(merged),
        "languages": dict(Counter(r.language for r in validated)),
        "train": dict(Counter(r.language for r in train)),
        "dev": dict(Counter(r.language for r in dev)),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": "Review decisions made by humans; model fluency is not guaranteed.",
    }
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, action="append", required=True,
                   help="Pass once per approved JSONL batch")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        print(json.dumps(merge(args.input, args.output), ensure_ascii=False, indent=2))
    except (ValueError, OSError) as exc:
        p.exit(2, f"Cannot merge reviewed data: {exc}\n")


if __name__ == "__main__":
    main()
