"""Stream a target of 1,000,000 *source-backed* SRT-M15 dialogue examples.

This is a data ingestion/quality pipeline, NOT a training loop or a promise
of one million verified examples. No dataset is downloaded without an explicit
--hf-source. Never publish raw data unless its individual license allows it.

Run small local tests first:
  python build_conversation_million.py --input examples.jsonl::zh --target 10 --zh-target 10
HF datasets streaming requires: pip install datasets
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import unicodedata

EMAIL = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
CN_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ROLES = {"user": "用户", "human": "用户", "prompter": "用户",
         "assistant": "助手", "gpt": "助手", "bot": "助手"}


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_source(spec, kind):
    # file::language or dataset_id::config::split::language
    parts = spec.split("::")
    if kind == "file" and len(parts) == 2 and parts[1] in ("zh", "en"):
        return parts
    if kind == "hf" and len(parts) == 4 and parts[3] in ("zh", "en"):
        return parts
    raise ValueError("Use --input path.jsonl::zh or --hf-source owner/repo::config::split::en")


def stream_rows(kind, spec):
    if kind == "file":
        path, language = parse_source(spec, kind)
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as reader:
            for line in reader:
                if line.strip():
                    try:
                        yield json.loads(line), language
                    except json.JSONDecodeError:
                        yield None, language
    else:
        dataset, config, split, language = parse_source(spec, kind)
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("HF streaming requires pip install datasets") from exc
        config = None if config in ("-", "default") else config
        for row in load_dataset(dataset, name=config, split=split, streaming=True):
            yield row, language


def extract_pairs(row):
    """Return (conversation root, prompt, response) for 1+ assistant turns.

    Root-based split avoids cross-turn leakage when one dialogue has many
    assistant answers. Does not invent replies or blindly accept model output.
    """
    if not isinstance(row, dict):
        return
    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, list):
        history = []
        root = None
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = ROLES.get(str(message.get("role") or message.get("from") or "").lower())
            content = message.get("content") or message.get("value")
            if isinstance(content, list):
                content = " ".join(x.get("text", "") for x in content
                                   if isinstance(x, dict) and x.get("type") == "text")
            if role is None or not isinstance(content, str) or not content.strip():
                continue
            content = content.strip()
            if role == "用户" and root is None:
                root = content
            if role == "助手" and root and history and history[-1][0] == "用户":
                prompt = "\n".join(f"{r}：{t}" for r, t in history[-8:])
                yield root, prompt, content
            history.append((role, content))
        return
    instruction = row.get("instruction")
    if isinstance(instruction, str):
        extra = row.get("input") or ""
        prompt = instruction.strip() + ("\n" + extra.strip() if isinstance(extra, str) and extra.strip() else "")
        response = row.get("output") or row.get("response")
    else:
        prompt = row.get("prompt")
        response = row.get("response") or row.get("output")
        if not isinstance(prompt, str) and isinstance(row.get("inputs"), str):
            prompt, response = row.get("inputs"), row.get("targets")
        if not isinstance(prompt, str) and isinstance(row.get("input"), str):
            prompt, response = row.get("input"), row.get("target")
    if isinstance(prompt, str) and isinstance(response, str):
        yield prompt, prompt, response


class Shards:
    def __init__(self, output, shard_size):
        self.root = Path(output)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.counts = Counter()
        self.files = {}
        self.paths = []
    def add(self, split, language, row):
        key = split, language
        n = self.counts[key]
        if n % self.shard_size == 0:
            old = self.files.pop(key, None)
            if old:
                old.close()
            path = self.root / split / language / f"part-{n // self.shard_size:05d}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.paths.append(path)
            self.files[key] = gzip.open(path, "wt", encoding="utf-8")
        self.files[key].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.counts[key] += 1
    def close(self):
        for handle in self.files.values():
            handle.close()
        self.files.clear()


def split_for(root, seed):
    # One deterministic split for all turns with the same first user prompt.
    n = int(digest(str(seed) + ":" + canonical(root))[:10], 16) % 10000
    return "test" if n < 500 else ("dev" if n < 1000 else "train")


def run(args):
    if args.target < 1 or not 0 <= args.zh_target <= args.target:
        raise ValueError("target >= 1 and zh-target between 0 and target")
    if args.shard_size < 1:
        raise ValueError("shard-size must be positive")
    sources = [("file", v) for v in args.input] + [("hf", v) for v in args.hf_source]
    if not sources:
        raise ValueError("Provide at least one local --input or explicit --hf-source")
    if args.hf_source and not args.accept_source_terms:
        raise ValueError("Review source licenses, provenance and access terms; pass --accept-source-terms")
    # Sources may have different license/research-only restrictions. A row is
    # never automatically marked human-reviewed or independently verified.
    quota = {"zh": args.zh_target, "en": args.target - args.zh_target}
    accepted = Counter()
    errors = Counter()
    source_stats = {}
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty to avoid mixing dataset runs")
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output / "dedup.sqlite3")
    db.execute("CREATE TABLE seen (hash TEXT PRIMARY KEY)")
    db.execute("PRAGMA journal_mode=WAL")
    shards = Shards(output / "shards", args.shard_size)
    try:
        for kind, spec in sources:
            language = parse_source(spec, kind)[-1]
            source_id = spec.split("::")[0]
            count = Counter()
            for row, _ in stream_rows(kind, spec):
                if accepted[language] >= quota[language]:
                    break
                count["read"] += 1
                if row is None:
                    count["invalid_json"] += 1
                    continue
                if isinstance(row, dict):
                    explicit_language = str(row.get("language_code") or row.get("language") or "").lower()
                    if explicit_language and (
                        (explicit_language.startswith("zh") and language != "zh") or
                        (explicit_language.startswith("en") and language != "en") or
                        (explicit_language in ("fr", "de", "es", "ru", "ja", "ar", "pt", "ko"))
                    ):
                        count["other_language"] += 1
                        continue
                for root, prompt, answer in extract_pairs(row):
                    if accepted[language] >= quota[language]:
                        break
                    prompt, answer = prompt.strip(), answer.strip()
                    if (not 4 <= len(prompt) <= args.max_prompt_chars or
                            not 2 <= len(answer) <= args.max_response_chars):
                        count["invalid_length"] += 1
                        continue
                    if EMAIL.search(prompt + answer) or CN_PHONE.search(prompt + answer):
                        count["possible_pii"] += 1
                        continue
                    key = digest(canonical(prompt))
                    if not canonical(prompt):
                        continue
                    inserted = db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (key,)).rowcount
                    if not inserted:
                        count["duplicate"] += 1
                        continue
                    split = split_for(root, args.seed)
                    topic = digest(canonical(root))[:20]
                    output_row = {
                        "language": language, "prompt": prompt, "response": answer,
                        "topic_id": topic, "split": split, "source": source_id,
                        "synthetic": bool(row.get("synthetic", False)),
                        "verified": False, "review_status": "unreviewed",
                    }
                    shards.add(split, language, output_row)
                    accepted[language] += 1
                    count["accepted"] += 1
                if count["read"] % 5000 == 0:
                    db.commit()
                if accepted["zh"] + accepted["en"] == args.target:
                    break
            source_stats[kind + ":" + spec] = dict(count)
            print(json.dumps({"source": spec, "accepted_total": sum(accepted.values()),
                              "accepted_zh": accepted["zh"], "accepted_en": accepted["en"]},
                             ensure_ascii=False), flush=True)
            if sum(accepted.values()) == args.target:
                break
    finally:
        shards.close()
        db.commit()
        db.close()
    report = {
        "requested": args.target, "accepted": sum(accepted.values()),
        "language_targets": quota, "language_accepted": dict(accepted),
        "missing": {k: quota[k] - accepted[k] for k in quota},
        "split_counts": {f"{s}/{lang}": n for (s, lang), n in shards.counts.items()},
        "source_stats": source_stats, "source_terms_acknowledged": bool(args.accept_source_terms),
        "verified_by_humans": False,
        "warning": "Source datasets may themselves be synthetic or contain mistakes. "
                   "Filtering cannot certify factual accuracy, privacy, licensing, "
                   "training success, or fluent conversation. Inspect every source.",
        "format": "srt-m15-conversation-million-v1",
    }
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["accepted"] == report["requested"] else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", default=[], help="local JSONL(.gz)::zh|en")
    p.add_argument("--hf-source", action="append", default=[],
                   help="HF dataset::config::split::zh|en (uses streaming)")
    p.add_argument("--accept-source-terms", action="store_true",
                   help="Confirm you reviewed licenses/conditions for all HF sources")
    p.add_argument("--target", type=int, default=1000000)
    p.add_argument("--zh-target", type=int, default=500000)
    p.add_argument("--shard-size", type=int, default=10000)
    p.add_argument("--max-prompt-chars", type=int, default=2048)
    p.add_argument("--max-response-chars", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="local_data/conversation_million")
    args = p.parse_args()
    try:
        sys.exit(run(args))
    except (ValueError, RuntimeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
