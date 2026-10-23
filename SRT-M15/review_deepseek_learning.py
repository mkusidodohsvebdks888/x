"""Human review and promotion gate for SRT-M15 DeepSeek Web drafts.

Reading a DeepSeek response is NEVER sufficient to label it verified.
Approvals are explicit, revision-bound and tied to an evidence note and a
rights attestation. No model training is executed here.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from bilingual_data import load_records, normalize_prompt, split_records

DEFAULT_DRAFTS = Path("local_data/deepseek_web_questions/drafts.jsonl")
DEFAULT_DECISIONS = Path("local_data/deepseek_web_questions/reviews.jsonl")
EMAIL = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
MAX_ANSWER = 2048


def read_jsonl(path):
    file = Path(path)
    if not file.exists():
        raise ValueError(f"File does not exist: {file}")
    rows = []
    for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file}:{number} invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{file}:{number} must be an object")
        rows.append(row)
    return rows


def draft_fingerprint(draft):
    fields = ("id", "language", "prompt", "response", "source", "answer_truncated")
    payload = json.dumps({k: draft.get(k) for k in fields},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def draft_index(path):
    indexed = {}
    for row in read_jsonl(path):
        identity = row.get("id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("Each draft must have a nonempty id")
        if identity in indexed:
            raise ValueError(f"Duplicate draft id: {identity}")
        indexed[identity] = row
    return indexed


def review_index(path):
    if not Path(path).exists():
        return {}
    decisions = {}
    for row in read_jsonl(path):
        identity = row.get("id")
        if not isinstance(identity, str) or row.get("decision") not in ("approve", "reject"):
            raise ValueError("Invalid review entry")
        decisions[identity] = row
    return decisions


def candidate_issues(draft, *, edited_answer=None):
    prompt = draft.get("prompt")
    answer = edited_answer if edited_answer is not None else draft.get("response")
    issues = []
    if draft.get("language") not in ("zh", "en"):
        issues.append("invalid-language")
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 1024:
        issues.append("invalid-prompt-length")
    if not isinstance(answer, str) or not 1 <= len(answer.strip()) <= MAX_ANSWER:
        issues.append("invalid-answer-length")
    if draft.get("answer_truncated"):
        issues.append("web-answer-truncated")
    if isinstance(prompt, str) and isinstance(answer, str):
        if EMAIL.search(prompt + answer) or PHONE.search(prompt + answer):
            issues.append("possible-personal-information")
        if normalize_prompt(prompt) == normalize_prompt(answer):
            issues.append("answer-repeats-question")
    return issues


def add_review(*, drafts, reviews, identity, decision, evidence="", rights_confirmed=False,
               answer_file=None, reason=""):
    indexed = draft_index(drafts)
    if identity not in indexed:
        raise ValueError(f"Unknown draft id: {identity}")
    draft = indexed[identity]
    if decision not in ("approve", "reject"):
        raise ValueError("Decision must be approve or reject")
    edited = None
    if decision == "approve":
        if not rights_confirmed:
            raise ValueError("Approving requires --rights-confirmed after checking source usage rights")
        if not isinstance(evidence, str) or len(evidence.strip()) < 8:
            raise ValueError("Approving requires meaningful --evidence describing independent review")
        if answer_file is not None:
            edited = Path(answer_file).read_text(encoding="utf-8").strip()
        issues = candidate_issues(draft, edited_answer=edited)
        # A truncated source is not silently promoted even after editing:
        # reviewer must re-collect the complete original source if needed.
        if issues:
            raise ValueError("Candidate blocked: " + ", ".join(issues))
    else:
        if not reason.strip():
            raise ValueError("Rejecting requires a reason")
    row = {
        "id": identity, "draft_fingerprint": draft_fingerprint(draft),
        "decision": decision, "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": evidence.strip() if decision == "approve" else "",
        "rights_confirmed": bool(rights_confirmed and decision == "approve"),
        "reason": reason.strip() if decision == "reject" else "",
        "edited_answer": edited,
    }
    dest = Path(reviews)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        raise ValueError("Review log cannot be a symlink")
    with dest.open("a", encoding="utf-8") as writer:
        writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        writer.flush()
        os.fsync(writer.fileno())
    return row


def get_approved_rows(drafts, reviews):
    indexed = draft_index(drafts)
    decisions = review_index(reviews)
    approved, rejected = [], Counter()
    seen = set()
    for identity, draft in indexed.items():
        review = decisions.get(identity)
        if review is None:
            rejected["not-reviewed"] += 1
            continue
        if review.get("decision") != "approve":
            rejected["rejected"] += 1
            continue
        if review.get("draft_fingerprint") != draft_fingerprint(draft):
            rejected["draft-changed"] += 1
            continue
        if not review.get("rights_confirmed") or len(str(review.get("evidence") or "").strip()) < 8:
            rejected["missing-human-review"] += 1
            continue
        edited = review.get("edited_answer")
        if edited is not None and not isinstance(edited, str):
            rejected["invalid-edit"] += 1
            continue
        issues = candidate_issues(draft, edited_answer=edited)
        if issues:
            rejected["issues:" + ",".join(issues)] += 1
            continue
        key = normalize_prompt(draft["prompt"])
        if key in seen:
            rejected["duplicate-question"] += 1
            continue
        seen.add(key)
        # Preserve provenance for DeepSeek web drafts and local Qwen
        # teacher demonstrations; never misattribute Qwen data to DeepSeek.
        origin = str(draft.get("source", ""))
        if origin == "deepseek-web-interactive-session":
            attribution = f"human-reviewed-deepseek-web:{identity}"
        elif origin.startswith("local-ollama:") and len(origin) <= 180:
            attribution = f"human-reviewed-{origin}:{identity}"
        else:
            rejected["unknown-source"] += 1
            continue
        approved.append({
            "language": draft["language"], "prompt": draft["prompt"].strip(),
            "response": (edited if edited is not None else draft["response"]).strip(),
            "source": attribution,
            "verified": True, "review_status": "human-reviewed",
            "review_evidence": review["evidence"], "synthetic": True,
            "split": "auto",
        })
    return approved, dict(rejected)


def export_dataset(*, drafts, reviews, output):
    rows, skipped = get_approved_rows(drafts, reviews)
    if not rows:
        raise ValueError("No human-approved training rows yet")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("Output cannot be a symlink")
    # Write an isolated temporary file and pass it through the ACTUAL SRT
    # bilingual loader/split to prevent accidental bypass of training rules.
    fd, temporary = tempfile.mkstemp(prefix=".reviewed-", suffix=".jsonl", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as writer:
            for row in rows:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        data = load_records(temporary, allow_demo=False)
        train, dev = split_records(data)
        if not train or not dev:
            raise ValueError("Both training and holdout must contain records")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"exported": len(rows), "train": len(train), "dev": len(dev),
            "languages": dict(Counter(row["language"] for row in rows)),
            "skipped": skipped, "path": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", default=str(DEFAULT_DRAFTS))
    parser.add_argument("--reviews", default=str(DEFAULT_DECISIONS))
    actions = parser.add_subparsers(dest="action", required=True)
    scan = actions.add_parser("scan", help="List drafts and automated quality warnings")
    scan.add_argument("--limit", type=int, default=30)
    approve = actions.add_parser("approve", help="Explicit manual approval; never automatic")
    approve.add_argument("id")
    approve.add_argument("--evidence", required=True,
                         help="Explain fact checking and content/provenance review")
    approve.add_argument("--rights-confirmed", action="store_true")
    approve.add_argument("--answer-file", help="Optional UTF-8 edited answer file")
    reject = actions.add_parser("reject")
    reject.add_argument("id")
    reject.add_argument("--reason", required=True)
    export = actions.add_parser("export")
    export.add_argument("--output", default="local_data/deepseek_web_questions/reviewed_sft.jsonl")
    args = parser.parse_args()
    try:
        if args.action == "scan":
            if not 1 <= args.limit <= 500:
                raise ValueError("scan --limit must be 1..500")
            drafts = draft_index(args.drafts)
            reviews = review_index(args.reviews)
            for identity, draft in list(drafts.items())[:args.limit]:
                flag = ",".join(candidate_issues(draft)) or "no-obvious-format-warning"
                print(f"{identity} | {draft.get('language')} | {reviews.get(identity, {}).get('decision', 'pending')} | {flag}")
                print(f"Q: {str(draft.get('prompt',''))[:180]}")
                print(f"A: {str(draft.get('response',''))[:260]}")
            print(f"Total drafts: {len(drafts)}; only format checks, NOT factual verification.")
        elif args.action in ("approve", "reject"):
            result = add_review(drafts=args.drafts, reviews=args.reviews,
                                identity=args.id, decision=args.action,
                                evidence=getattr(args, "evidence", ""),
                                rights_confirmed=getattr(args, "rights_confirmed", False),
                                answer_file=getattr(args, "answer_file", None),
                                reason=getattr(args, "reason", ""))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(export_dataset(drafts=args.drafts, reviews=args.reviews,
                                            output=args.output),
                             ensure_ascii=False, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Review failed: {exc}\n")


if __name__ == "__main__":
    main()
