"""Train SRT-M15 with the mechanisms shared by many autoregressive LLMs.

Pretraining (next-character prediction):
    python train_common.py --mode pretrain --data examples/common_text_corpus.txt

Instruction tuning (assistant-response-only supervised loss):
    python train_common.py --mode sft --data examples/common_instruction_examples.jsonl

All weights begin RANDOM unless --init-checkpoint is explicitly supplied.
This is a deliberately small educational trainer, not ChatGPT/DeepSeek parity.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import torch

from srt_m15 import ModelConfig
from srt_m15.language import SRTM15LanguageModel
from srt_m15.common_mechanisms import (
    LanguageExample, character_vocabulary, instruction_example, make_windows,
    split_examples, train_language_steps, evaluate_language, sample_tokens,
)


def load_examples(mode: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    if mode == "pretrain":
        # Each blank-line separated block is one independent example/document.
        # Split documents before extracting training windows to avoid overlap.
        blocks = path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n\n")
        examples = [LanguageExample(x.strip() + "\n") for x in blocks if len(x.strip()) >= 2]
    elif mode == "sft":
        examples = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                examples.append(instruction_example(obj["prompt"], obj["response"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid instruction JSONL row {number}: {exc}") from exc
    else:
        raise ValueError("mode must be pretrain or sft")
    if len(examples) < 2:
        raise ValueError("supply at least two independent documents or instruction examples")
    if len({e.text for e in examples}) != len(examples):
        raise ValueError("duplicated examples would contaminate train/validation split")
    return examples


def run(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but no CUDA device is available")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    examples = load_examples(args.mode, Path(args.data))
    train_examples, valid_examples = split_examples(
        examples, validation_fraction=args.validation_fraction, seed=args.seed)
    vocab = character_vocabulary(train_examples, valid_examples)
    train_windows = make_windows(train_examples, vocab, args.context)
    validation_windows = make_windows(valid_examples, vocab, args.context)

    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["vocab"] != vocab.chars:
            raise ValueError("initial checkpoint vocabulary does not match the new dataset")
        cfg = ModelConfig(**checkpoint["config"])
        if cfg.max_length != args.context:
            raise ValueError("checkpoint context does not match --context")
        model = SRTM15LanguageModel(cfg)
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        cfg = ModelConfig(
            vocab_size=len(vocab.chars), num_classes=len(vocab.chars),
            dim=args.dim, max_length=args.context, heads=args.heads,
            num_experts=args.experts, reasoning_steps=args.reasoning_steps)
        model = SRTM15LanguageModel(cfg)

    params = sum(p.numel() for p in model.parameters())
    print(json.dumps({
        "device": device, "parameters": params, "mode": args.mode,
        "train_examples": len(train_examples), "validation_examples": len(valid_examples),
        "train_windows": len(train_windows), "validation_windows": len(validation_windows),
        "vocabulary_size": len(vocab.chars)
    }, ensure_ascii=False))
    history = train_language_steps(
        model, train_windows, validation_windows, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        device=device, log_every=args.log_every)
    for entry in history:
        print(json.dumps(entry, ensure_ascii=False))
    final_eval = evaluate_language(model, validation_windows,
                                   batch_size=args.batch_size, device=device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Serialization is intentionally compatible with the existing generate.py.
    checkpoint = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": asdict(cfg), "vocab": vocab.chars,
        "metadata": {
            "trainer": "common_mechanisms", "mode": args.mode,
            "source": str(args.data), "seed": args.seed,
            "steps_this_run": args.steps, "validation_fraction": args.validation_fraction,
            "training_examples": len(train_examples),
            "validation_examples": len(valid_examples),
            "validation": final_eval, "history": history,
        },
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    print("Saved:", str(output))

    if args.prompt:
        unknown = sorted(set(args.prompt) - set(vocab.chars))
        if unknown:
            raise ValueError(f"prompt has unseen characters: {unknown!r}")
        model.to(device).eval()
        prompt_ids = torch.tensor([vocab.encode(args.prompt)], dtype=torch.long, device=device)
        generated = sample_tokens(model, prompt_ids, max_new_tokens=args.new_tokens,
                                  temperature=args.temperature, top_k=args.top_k,
                                  top_p=args.top_p)
        print("Generated:", vocab.decode(generated[0].tolist()))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("pretrain", "sft"), required=True)
    p.add_argument("--data", required=True, help="UTF-8 text file or JSONL instruction pairs")
    p.add_argument("--output", default="checkpoints/srt_m15_common.pt")
    p.add_argument("--init-checkpoint", default="", help="optional compatible starting weights")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--context", type=int, default=24)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--experts", type=int, default=3)
    p.add_argument("--reasoning-steps", type=int, default=1)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--prompt", default="", help="optional generation after training")
    p.add_argument("--new-tokens", type=int, default=20)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
