"""SRT-M15 Windows/local Qwen teacher -> review -> GPU student workflow.

All phases are explicit. A model-produced answer is NEVER auto-approved.
This orchestrator starts subprocesses synchronously and never runs in background.

Examples:
  python srt_full_pipeline.py doctor
  python srt_full_pipeline.py generate --pairs 40 --offset 0
  python srt_full_pipeline.py review --drafts local_data/qwen_teacher/batch_000000_000040.jsonl
  python srt_full_pipeline.py export --drafts ... --reviews ... --output ...
  python srt_full_pipeline.py train --data local_data/qwen_teacher/reviewed_sft.jsonl
  python srt_full_pipeline.py student --checkpoint checkpoints/qwen_student/best.pt
  python srt_full_pipeline.py chat
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.error import URLError
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable


def invoke(*items):
    command = [PYTHON, *map(str, items)]
    result = subprocess.run(command, cwd=HERE, check=False)
    if result.returncode:
        raise RuntimeError("Step failed with exit code " + str(result.returncode))
    return result.returncode


def doctor():
    print("SRT-M15 preflight (read-only)", flush=True)
    print("Python:", sys.version.split()[0])
    try:
        import torch
        available = torch.cuda.is_available()
        print("PyTorch:", torch.__version__, "| CUDA:", available)
        if available:
            print("GPU:", torch.cuda.get_device_name(0))
            print("GPU VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    except ImportError:
        print("PyTorch: not installed; install requirements.txt and CUDA-compatible torch")
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
            response_data = json.load(response)
        names = [row.get("name", "") for row in response_data.get("models", [])
                 if isinstance(row, dict)]
        print("Ollama: running locally")
        print("Available models:", ", ".join(names) if names else "(none)")
        if not names:
            print("Download a model: ollama pull qwen3:4b-instruct")
    except (OSError, URLError, ValueError) as exc:
        print("Ollama: not reachable on localhost (", type(exc).__name__, ")")
        print("Install/start Ollama and download: ollama pull qwen3:4b-instruct")
    print("This check never downloads models or modifies training weights.")


def batch_filename(offset, pairs):
    return HERE / "local_data/qwen_teacher" / f"batch_{offset:06d}_{offset+pairs:06d}.jsonl"


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("doctor", help="Check Python, CUDA GPU, and local Ollama")
    gen = sub.add_parser("generate", help="Collect a bounded bilingual teacher batch")
    gen.add_argument("--pairs", type=int, default=40)
    gen.add_argument("--offset", type=int, default=0)
    gen.add_argument("--model", default="qwen3:4b-instruct")
    gen.add_argument("--output", type=Path)
    gen.add_argument("--preview", action="store_true", help="Dry-run without teacher calls")
    scan = sub.add_parser("review", help="Display Qwen drafts for human review")
    scan.add_argument("--drafts", type=Path, required=True)
    scan.add_argument("--reviews", type=Path)
    scan.add_argument("--limit", type=int, default=30)
    exp = sub.add_parser("export", help="Export ONLY manually approved samples")
    exp.add_argument("--drafts", type=Path, required=True)
    exp.add_argument("--reviews", type=Path, required=True)
    exp.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("merge", help="Combine independently reviewed batches")
    merge.add_argument("--input", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train", help="Train SRT-M15 on verified SFT JSONL")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("checkpoints/qwen_student"))
    train.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    train.add_argument("--steps", type=int, default=60,
                       help="Steps per stage (1..1000)")
    train.add_argument("--stages", type=int, default=2)
    train.add_argument("--context", type=int, default=128)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--dim", type=int, default=32)
    train.add_argument("--resume", action="store_true",
                       help="Continue from output/latest.pt with SAME verified data/hyperparameters")
    demo = sub.add_parser("demo-train", help="UNREVIEWED toy experiment; never a production trained model")
    demo.add_argument("--data", type=Path, required=True)
    demo.add_argument("--output", type=Path, default=Path("checkpoints/qwen_demo_student"))
    demo.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    demo.add_argument("--steps", type=int, default=20)
    demo.add_argument("--context", type=int, default=128)
    student = sub.add_parser("student", help="Chat using ORIGINAL trained SRT-M15 weights")
    student.add_argument("--checkpoint", type=Path, default=Path("checkpoints/qwen_student/best.pt"))
    student.add_argument("--prompt")
    chat = sub.add_parser("chat", help="Chat immediately through PRETRAINED local Qwen")
    chat.add_argument("--model", default="qwen3:4b-instruct")
    return p.parse_args()


def run(args):
    if args.action == "doctor":
        return doctor()
    if args.action == "generate":
        if not 5 <= args.pairs <= 200 or args.offset < 0:
            raise ValueError("pairs must be 5..200; offset must be >= 0")
        output = args.output or batch_filename(args.offset, args.pairs)
        argv = ["qwen_teacher_data.py", "--pairs", args.pairs,
                "--offset", args.offset, "--model", args.model,
                "--output", output]
        if args.preview:
            argv.append("--dry-run")
        return invoke(*argv)
    if args.action == "review":
        reviews = args.reviews or args.drafts.with_name("reviews.jsonl")
        return invoke("review_deepseek_learning.py", "--drafts", args.drafts,
                      "--reviews", reviews, "scan", "--limit", args.limit)
    if args.action == "export":
        return invoke("review_deepseek_learning.py", "--drafts", args.drafts,
                      "--reviews", args.reviews, "export", "--output", args.output)
    if args.action == "merge":
        return invoke("merge_reviewed_sft.py", "--output", args.output,
                      *sum((["--input", path] for path in args.input), []))
    if args.action == "train":
        data = Path(args.data)
        if not data.is_file():
            raise ValueError("Verified training dataset not found: " + str(data))
        # Verify before subprocess starts: no --allow-demo-data here.
        from bilingual_data import load_records, split_records
        records = load_records(data, allow_demo=False)
        split_records(records)
        if not 1 <= args.steps <= 1000 or not 1 <= args.stages <= 10:
            raise ValueError("steps must be 1..1000 and stages 1..10")
        cmd = ["train_bilingual_stages.py", "--data", data,
               "--device", args.device, "--steps-per-stage", args.steps,
               "--stages", args.stages, "--context", args.context,
               "--batch-size", args.batch_size, "--dim", args.dim,
               "--heads", 4, "--experts", 2, "--reasoning-steps", 1,
               "--max-eval-pairs", 16, "--sampling", "cycle", "--output", args.output]
        checkpoint = Path(args.output) / "latest.pt"
        if args.resume:
            if not checkpoint.is_file():
                raise ValueError("--resume requested but checkpoint missing: " + str(checkpoint))
            cmd.extend(["--resume", checkpoint])
        print("Training original SRT-M15 weights on HUMAN-APPROVED data only.", flush=True)
        return invoke(*cmd)
    if args.action == "demo-train":
        if not 1 <= args.steps <= 1000:
            raise ValueError("steps must be 1..1000")
        print("UNREVIEWED DATA: experimental smoke test only; not certified learning.", flush=True)
        return invoke("train_bilingual_stages.py", "--data", args.data,
                      "--allow-demo-data", "--device", args.device,
                      "--steps-per-stage", args.steps, "--stages", 1,
                      "--context", args.context, "--batch-size", 2, "--dim", 16,
                      "--heads", 4, "--experts", 2, "--reasoning-steps", 1,
                      "--max-eval-pairs", 8, "--output", args.output)
    if args.action == "student":
        cp = Path(args.checkpoint)
        if not cp.is_file():
            latest = cp.with_name("latest.pt")
            if latest.is_file() and cp.name == "best.pt":
                cp = latest
            else:
                raise ValueError("Student checkpoint missing: " + str(cp))
        cmd = ["chat_staged.py", "--checkpoint", cp]
        if args.prompt:
            cmd.extend(["--prompt", args.prompt])
        return invoke(*cmd)
    if args.action == "chat":
        return invoke("start_local.py", "--chat-backend", "ollama", "--ollama-model", args.model)
    raise ValueError("Unknown subcommand")


if __name__ == "__main__":
    try:
        run(parse())
    except (ValueError, RuntimeError, OSError) as exc:
        sys.exit("SRT-M15 pipeline: " + str(exc))
