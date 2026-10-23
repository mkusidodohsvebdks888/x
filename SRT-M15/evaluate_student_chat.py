"""Record original SRT-M15 answers for INDEPENDENT MANUAL conversation review.

Each case has language, prompt, criterion. This report marks output/pending
review, NOT pass/fail correctness. No student weights are modified.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


def evaluate(*, checkpoint, cases, output, max_tokens=48, device="auto"):
    if not 1 <= max_tokens <= 128:
        raise ValueError("max_tokens must be 1..128")
    import torch
    from chat_staged import load_checkpoint, reply
    run_device = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device)
    if run_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    model, vocab = load_checkpoint(checkpoint, run_device)
    raw_cases = Path(cases).read_text(encoding="utf-8").splitlines()
    rows = []
    for line_no, line in enumerate(raw_cases, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid evaluation JSON at line {line_no}") from exc
        if (not isinstance(item, dict) or item.get("language") not in ("zh", "en")
                or not isinstance(item.get("prompt"), str)
                or not 1 <= len(item["prompt"].strip()) <= 400
                or not isinstance(item.get("check"), str)
                or not item["check"].strip()):
            raise ValueError(f"Invalid evaluation scenario at line {line_no}")
        answer, error = reply(model, vocab, [], item["prompt"].strip(), limit=max_tokens)
        rows.append({
            "id": line_no, "language": item["language"],
            "prompt": item["prompt"].strip(),
            "criterion": item["check"].strip(),
            "generated_answer": answer or "",
            "generation_error": error or "",
            "nonempty_output": bool(answer and answer.strip()),
            "human_judgement": "pending",
        })
    if not rows:
        raise ValueError("No evaluation scenarios")
    result = {
        "checkpoint": str(checkpoint), "cases": len(rows),
        "nonempty_output": sum(x["nonempty_output"] for x in rows),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": run_device,
        "warning": "Generation availability is not conversational correctness. Review each case manually.",
        "results": rows,
    }
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        raise ValueError("Refusing to overwrite existing manual evaluation: " + str(dest))
    fd, temp = tempfile.mkstemp(dir=dest.parent, prefix=".acceptance-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as writer:
            json.dump(result, writer, ensure_ascii=False, indent=2)
        if dest.exists():
            raise ValueError("Evaluation output exists")
        os.replace(temp, dest)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/qwen_student/best.pt")
    p.add_argument("--cases", default="examples/conversation_acceptance_v1.jsonl")
    p.add_argument("--output", default="local_data/evaluation/student_acceptance.json")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()
    try:
        report = evaluate(
            checkpoint=args.checkpoint, cases=args.cases, output=args.output,
            max_tokens=args.max_tokens, device=args.device,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        p.exit(2, f"Evaluation failed: {exc}\n")
    print(f"Saved {report['cases']} responses, {report['nonempty_output']} nonempty.")
    print("Open the JSON report and review every criterion manually; no fluent-chat claim.")


if __name__ == "__main__":
    main()
