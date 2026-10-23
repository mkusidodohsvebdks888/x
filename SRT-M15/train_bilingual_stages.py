"""Bounded multi-stage, resumable training of *original* SRT-M15 weights.

Each stage runs genuine optimizer updates, evaluates held-out Chinese/English
separately, and atomically saves both latest (resumable) and best (promoted)
weights. This tiny original architecture is NOT broadly fluent.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import uuid

import torch

from bilingual_data import load_records, split_records
from srt_m15 import ModelConfig
from srt_m15.common_mechanisms import (
    character_vocabulary, evaluate_language, instruction_example,
    make_windows, masked_token_loss,
)
from srt_m15.language import SRTM15LanguageModel, CharVocabulary


FORMAT = "srt-m15-bilingual-staged-v1"
LANGUAGES = ("zh", "en")


def _atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _configuration(args, digest):
    return {
        "data_sha256": digest, "seed": args.seed, "context": args.context,
        "dim": args.dim, "heads": args.heads, "experts": args.experts,
        "reasoning_steps": args.reasoning_steps,
        "batch_size": args.batch_size, "steps_per_stage": args.steps_per_stage,
        "learning_rate": args.lr, "dev_fraction": args.dev_fraction,
        "allow_demo_data": bool(args.allow_demo_data),
    }


def _macro_loss(metrics):
    return sum(metrics[lang]["token_loss"] for lang in LANGUAGES) / len(LANGUAGES)


def can_promote(candidate, best, *, max_relative_regression=.1):
    """A heuristic small-test gate, NOT general-language certification."""
    if not all(
        math.isfinite(candidate[lang]["token_loss"])
        and candidate[lang]["token_loss"] <= best[lang]["token_loss"] * (1 + max_relative_regression)
        for lang in LANGUAGES
    ):
        return False
    return _macro_loss(candidate) < _macro_loss(best)


def _measure(model, heldout, *, device):
    result = {}
    for language in LANGUAGES:
        result[language] = evaluate_language(model, heldout[language], device=device)
    return result


def run(args):
    if not 1 <= args.stages <= 10 or not 1 <= args.steps_per_stage <= 1000:
        raise ValueError("stages must be 1..10, steps-per-stage must be 1..1000")
    if not 2 <= args.batch_size <= 64 or not 8 <= args.context <= 512:
        raise ValueError("batch-size must be 2..64 and context must be 8..512")
    if not (0 < args.lr <= .01 and 0 < args.dev_fraction < 1):
        raise ValueError("invalid learning rate or dev fraction")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_path = Path(args.data)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    records = load_records(data_path, allow_demo=args.allow_demo_data)
    train, dev = split_records(records, seed=args.seed, dev_fraction=args.dev_fraction)
    train_examples = [instruction_example(r.prompt, r.response) for r in train]
    dev_examples = [instruction_example(r.prompt, r.response) for r in dev]
    vocab = character_vocabulary(train_examples, dev_examples)
    windows_by_language = {
        lang: make_windows([instruction_example(r.prompt, r.response)
                            for r in train if r.language == lang], vocab, args.context)
        for lang in LANGUAGES
    }
    heldout = {
        lang: make_windows([instruction_example(r.prompt, r.response)
                            for r in dev if r.language == lang], vocab, args.context)
        for lang in LANGUAGES
    }
    cfg = ModelConfig(vocab_size=len(vocab.chars), num_classes=len(vocab.chars),
                      max_length=args.context, dim=args.dim, heads=args.heads,
                      num_experts=args.experts, reasoning_steps=args.reasoning_steps)
    model = SRTM15LanguageModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    rng = random.Random(args.seed)
    settings = _configuration(args, digest)
    output = Path(args.output)
    start_stage, total_steps = 0, 0
    history = []
    best = None
    best_stage = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state.get("format") != FORMAT or state.get("settings") != settings:
            raise ValueError("resume checkpoint/data/hyperparameters mismatch")
        if state.get("config") != asdict(cfg) or state.get("vocab") != vocab.chars:
            raise ValueError("resume model configuration/vocabulary mismatch")
        if int(state.get("stage", -1)) < 1 or int(state.get("total_steps", -1)) < 1:
            raise ValueError("invalid resume progress")
        if int(state["total_steps"]) != int(state["stage"]) * args.steps_per_stage:
            raise ValueError("inconsistent resume steps")
        if not isinstance(state.get("history"), list):
            raise ValueError("missing resume history")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        # torch.load(map_location='cpu') leaves optimizer buffers on CPU.
        # Relocate them when continuing on GPU.
        for entry in opt.state.values():
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    entry[key] = value.to(device)
        rng.setstate(state["rng_state"])
        history = state["history"]
        start_stage, total_steps = int(state["stage"]), int(state["total_steps"])
        best, best_stage = state["best_dev"], int(state["best_stage"])
        if len(history) != start_stage:
            raise ValueError("resume history length differs from completed stages")
        if output.resolve() == Path(args.resume).resolve():
            raise ValueError("output must be a directory, not the resume checkpoint file")
        # When resuming from a previous GitHub Actions artifact, the new output
        # directory is empty. Carry forward its previously promoted best model
        # instead of accidentally losing it if later stages fail promotion.
        earlier_best = Path(args.resume).with_name("best.pt")
        dest_best = output / "best.pt"
        if (best_stage > 0 and earlier_best.is_file()
                and earlier_best.resolve() != dest_best.resolve()
                and not dest_best.is_file()):
            prior = torch.load(earlier_best, map_location="cpu", weights_only=True)
            if (prior.get("format") != FORMAT or prior.get("settings") != settings
                    or int(prior.get("stage", -1)) != best_stage
                    or prior.get("config") != asdict(cfg)
                    or prior.get("vocab") != vocab.chars):
                raise ValueError("previous best checkpoint is incompatible")
            dest_best.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(earlier_best, dest_best)

    def report():
        return {
            "format": FORMAT, "model": "original SRT-M15 random-initialized toy",
            "device": device, "parameters": sum(p.numel() for p in model.parameters()),
            "stage": len(history),
            "total_steps": total_steps, "best_stage": best_stage,
            "train_examples": {lang: sum(r.language == lang for r in train) for lang in LANGUAGES},
            "dev_examples": {lang: sum(r.language == lang for r in dev) for lang in LANGUAGES},
            "settings": settings, "history": history,
            "warning": "Token-level tests do not prove broadly fluent Chinese/English dialogue.",
        }

    baseline = _measure(model, heldout, device=device)
    if best is None:
        best = baseline
    print(json.dumps({"event": "start", "stage": start_stage,
                      "total_steps": total_steps, "baseline": baseline},
                     ensure_ascii=False), flush=True)

    for stage in range(start_stage + 1, start_stage + args.stages + 1):
        model.train()
        losses = []
        for _ in range(args.steps_per_stage):
            # Reserve at least one window for each language in every optimizer step.
            zhn = args.batch_size // 2
            enn = args.batch_size - zhn
            samples = rng.choices(windows_by_language["zh"], k=zhn)
            samples += rng.choices(windows_by_language["en"], k=enn)
            rng.shuffle(samples)
            batch_x = torch.stack([p[0] for p in samples]).to(device)
            batch_y = torch.stack([p[1] for p in samples]).to(device)
            logits = model(batch_x)["logits"]
            loss = masked_token_loss(logits, batch_y)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("non-finite training loss; latest accepted files retained")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            total_steps += 1

        candidate = _measure(model, heldout, device=device)
        promote = can_promote(candidate, best)
        if promote:
            best, best_stage = candidate, stage
        item = {
            "stage": stage, "step": total_steps,
            "average_train_batch_loss": sum(losses) / len(losses),
            "dev": candidate, "promoted": promote,
        }
        history.append(item)
        checkpoint = {
            "format": FORMAT, "settings": settings, "config": asdict(cfg),
            "vocab": vocab.chars, "model": {k: v.detach().cpu().clone()
                                           for k, v in model.state_dict().items()},
            "optimizer": opt.state_dict(), "rng_state": rng.getstate(),
            "stage": stage, "total_steps": total_steps, "history": history,
            "best_dev": best, "best_stage": best_stage,
        }
        # Always keep latest for genuine continuation, even if it is weaker.
        _atomic_torch(output / "latest.pt", checkpoint)
        if promote:
            _atomic_torch(output / "best.pt", checkpoint)
        _atomic_json(output / "history.json", {
            **report(), "stage": stage, "total_steps": total_steps,
        })
        print(json.dumps({"event": "stage_complete", **item}, ensure_ascii=False),
              flush=True)
    return report()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="examples/bilingual_toy_sft.jsonl")
    p.add_argument("--allow-demo-data", action="store_true")
    p.add_argument("--output", default="checkpoints/bilingual_stages")
    p.add_argument("--resume", default="", help="trusted latest.pt from previous stage/run")
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--steps-per-stage", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--context", type=int, default=32)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--experts", type=int, default=2)
    p.add_argument("--reasoning-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--dev-fraction", type=float, default=.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
