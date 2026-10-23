"""Bounded bilingual LoRA tuning on an existing open-weight language model.

Requires CUDA GPU and explicitly reviewed JSONL (unless --allow-demo-data).
This produces a LoRA adapter, NOT weights for the from-scratch SRTM15 network.
No automatic GPU rental, no model hub upload, and no unattended online writes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from bilingual_data import load_records, split_records, language_counts
from bilingual_runtime import BASE_MODEL, SYSTEM_PROMPT


def format_example(tokenizer, example, context_limit):
    """Only the assistant response receives training labels."""
    import torch

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example.prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    full = prefix + example.response + tokenizer.eos_token
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if len(ids) > context_limit:
        raise ValueError(f"{example.language} example exceeds max_seq_len")
    if not (0 < len(prefix_ids) < len(ids)):
        raise ValueError("assistant target is empty or tokenization boundary invalid")
    labels = [-100] * len(prefix_ids) + ids[len(prefix_ids):]
    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "language": example.language}


def batches(examples, *, batch_size, pad_id, device):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    for at in range(0, len(examples), batch_size):
        part = examples[at: at + batch_size]
        ids = pad_sequence([x["input_ids"] for x in part],
                           batch_first=True, padding_value=pad_id)
        labels = pad_sequence([x["labels"] for x in part],
                              batch_first=True, padding_value=-100)
        mask = (torch.arange(ids.shape[1])[None, :] <
                torch.tensor([len(x["input_ids"]) for x in part])[:, None]).long()
        yield {"input_ids": ids.to(device), "labels": labels.to(device),
               "attention_mask": mask.to(device)}


def evaluate(model, samples, pad_id, device):
    import math
    import torch

    model.eval()
    result = {}
    with torch.inference_mode():
        for language in ("zh", "en"):
            subset = [x for x in samples if x["language"] == language]
            loss_sum, tokens = 0., 0
            for batch in batches(subset, batch_size=1, pad_id=pad_id, device=device):
                count = int((batch["labels"] != -100).sum().item())
                value = float(model(**batch).loss.item())
                if not math.isfinite(value):
                    raise RuntimeError("Non-finite validation loss")
                loss_sum += value * count
                tokens += count
            if not tokens:
                raise ValueError(f"No held-out supervised tokens for {language}")
            loss = loss_sum / tokens
            result[language] = {"loss": loss, "perplexity": math.exp(min(80, loss)),
                                "tokens": tokens, "examples": len(subset)}
    return result


def eligible(before, after, *, max_relative_regression=.03):
    """A tiny dev set is a weak signal; never assert general fluency from this."""
    for lang in ("zh", "en"):
        if after[lang]["loss"] > before[lang]["loss"] * (1 + max_relative_regression):
            return False, f"{lang} regression on held-out data"
    total_before = sum(before[lang]["loss"] * before[lang]["tokens"]
                       for lang in ("zh", "en"))
    total_after = sum(after[lang]["loss"] * after[lang]["tokens"]
                      for lang in ("zh", "en"))
    return ((total_after < total_before), "improved" if total_after < total_before
            else "no held-out loss improvement")


def run(args):
    if not 1 <= args.max_steps <= 2000 or not 1 <= args.batch_size <= 16:
        raise ValueError("max_steps must be 1..2000 and batch_size 1..16")
    if not 128 <= args.max_seq_len <= 2048:
        raise ValueError("max_seq_len must be 128..2048")
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for this LoRA trainer; use Colab with a GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_records(args.data, allow_demo=args.allow_demo_data)
    train, dev = split_records(records, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    train_examples = [format_example(tokenizer, x, args.max_seq_len) for x in train]
    dev_examples = [format_example(tokenizer, x, args.max_seq_len) for x in dev]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=False,
    ).to("cuda")
    model.config.use_cache = False
    before = evaluate(model, dev_examples, pad_id, "cuda")
    lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                      lora_dropout=.05, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    for step in range(1, args.max_steps + 1):
        model.train()
        part = random.choices(train_examples, k=args.batch_size)
        data = next(batches(part, batch_size=args.batch_size,
                            pad_id=pad_id, device="cuda"))
        loss = model(**data).loss
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            print(f"step={step}/{args.max_steps} train_loss={loss.item():.4f}", flush=True)

    after = evaluate(model, dev_examples, pad_id, "cuda")
    accepted, reason = eligible(before, after)
    report = {
        "format": "srt-m15-bilingual-lora-v1",
        "pretrained_base": args.model,
        "training_origin": "demo" if args.allow_demo_data else "reviewed",
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "seed": args.seed, "steps": args.max_steps,
        "train_counts": language_counts(train), "dev_counts": language_counts(dev),
        "baseline_dev": before, "candidate_dev": after,
        "accepted": accepted, "reason": reason,
        "warning": "Dev loss improvement on a small set does not prove broad bilingual fluency."
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if accepted:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
        print(f"Accepted LoRA adapter: {output}")
    else:
        print("Candidate rejected. Base model remains unchanged.")
    return accepted


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--data", required=True)
    p.add_argument("--allow-demo-data", action="store_true",
                   help="Use unreviewed, educational examples ONLY for pipeline testing")
    p.add_argument("--output", default="checkpoints/bilingual_lora")
    p.add_argument("--report", default="checkpoints/bilingual_lora_report.json")
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
