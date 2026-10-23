"""Find weak SRT-M15 answers using a LOCAL, manually prepared evaluation set.

Marks possible learning gaps, never treats string overlap as fact verification.
Outputs questions only (not reference answers), for optional manual web queries.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata


def tokens(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    # Mixed Chinese+English: compare individual CJK characters or word tokens.
    return re.findall(r"[\u3400-\u9fff]|[a-z]+|\d+", text)


def overlap_f1(prediction, expected):
    a, b = Counter(tokens(prediction)), Counter(tokens(expected))
    if not a or not b:
        return 0.0
    shared = sum((a & b).values())
    return (2 * shared) / (sum(a.values()) + sum(b.values()))


def evaluate(*, checkpoint, cases, output, threshold=0.4, limit=100):
    if not 0 <= threshold <= 1 or not 1 <= limit <= 500:
        raise ValueError("Threshold must be 0..1 and limit 1..500")
    from chat_staged import load_checkpoint, reply
    model, vocab = load_checkpoint(checkpoint, "cpu")
    seen = set()
    misses = []
    counts = Counter()
    for line_num, line in enumerate(Path(cases).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or item.get("language") not in ("zh", "en"):
            raise ValueError(f"Evaluation line {line_num}: expected language zh/en")
        prompt, reference = item.get("prompt"), item.get("reference")
        if (not isinstance(prompt, str) or not isinstance(reference, str)
                or not prompt.strip() or not reference.strip()):
            raise ValueError(f"Evaluation line {line_num}: prompt/reference required")
        if len(prompt) > 350 or len(reference) > 2048:
            raise ValueError(f"Evaluation line {line_num}: text too long")
        language = item["language"]
        counts[f"evaluated_{language}"] += 1
        answer, error = reply(model, vocab, [], prompt, limit=80)
        score = overlap_f1(answer or "", reference)
        if error or not answer or score < threshold:
            key = (language, prompt.strip())
            if key not in seen and len(misses) < limit:
                seen.add(key)
                # We intentionally do NOT write the reference into a file
                # that may later be transmitted to DeepSeek's website.
                misses.append({
                    "language": language, "prompt": prompt.strip(),
                    "heuristic_score": round(score, 4),
                    "reason": "encoding-or-generation-error" if error else "low-reference-overlap",
                    "review_status": "candidate-only",
                    "origin": "local-staged-evaluation",
                })
                counts[f"potential_gaps_{language}"] += 1
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("Output cannot be a symlink")
    with destination.open("w", encoding="utf-8") as handle:
        for gap in misses:
            handle.write(json.dumps(gap, ensure_ascii=False) + "\n")
    return {"candidate_questions": len(misses), "metrics": dict(counts),
            "output": str(destination), "warning": "Lexical overlap is a rough triage signal, not a measure of correctness."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/local_dialogues/latest.pt")
    p.add_argument("--cases", required=True, help="JSONL with language,prompt,reference; not sent anywhere")
    p.add_argument("--output", default="local_data/deepseek_web_questions/local_gaps.jsonl")
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args()
    try:
        summary = evaluate(checkpoint=args.checkpoint, cases=args.cases,
                           output=args.output, threshold=args.threshold, limit=args.limit)
    except (OSError, ValueError, RuntimeError) as exc:
        p.exit(2, f"Local evaluation failed: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
