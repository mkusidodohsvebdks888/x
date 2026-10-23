"""SRT-M15 supervised continual QA learning with replay and heldout gating.

Usage (from SRT-M15 directory, after supervised QA pretraining):
    python continual_qa.py --data learning_data/verified_qa.jsonl --steps 300

This is NOT self-supervised fact acquisition: only human/source-verified labels
are permitted as training examples. Promotion is conditional on heldout scores.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from learning_pipeline import read_verified_jsonl, split_verified, normalized_question, promotion_gate
from srt_m15 import ModelConfig
from srt_m15.language import CharVocabulary, SRTM15LanguageModel
from train_qa import DEV, answer, collate, load, make_example, training_pairs


def extend_vocab(model, vocab, cfg, samples, device):
    """Expand input/output vocabulary without resetting already learned tensors.

    Only tensors with a dimension equal to the old vocab size may be resized.
    Fail closed if an unexpected architecture change makes mapping ambiguous.
    """
    new_chars = sorted({char for q, a in samples for char in (f"问：{q}\n答：{a}\n")
                        if char not in vocab.stoi})
    if not new_chars:
        return model, vocab, cfg, []
    old_count = len(vocab.chars)
    new_vocab = CharVocabulary(list(vocab.chars) + new_chars)
    new_cfg = ModelConfig(**{
        **asdict(cfg), "vocab_size": len(new_vocab.chars),
        "num_classes": len(new_vocab.chars)
    })
    bigger = SRTM15LanguageModel(new_cfg).to(device)
    src, dst = model.state_dict(), bigger.state_dict()
    if src.keys() != dst.keys():
        raise RuntimeError("模型结构不兼容，拒绝不完整的词表迁移")
    resized = 0
    with torch.no_grad():
        for key, new_tensor in dst.items():
            old_tensor = src[key]
            if old_tensor.shape == new_tensor.shape:
                new_tensor.copy_(old_tensor)
                continue
            old_dims = tuple(old_tensor.shape)
            new_dims = tuple(new_tensor.shape)
            changed = [i for i, (o, n) in enumerate(zip(old_dims, new_dims)) if o != n]
            if (len(old_dims) != len(new_dims) or len(changed) != 1 or
                    old_dims[changed[0]] != old_count or
                    new_dims[changed[0]] != len(new_vocab.chars)):
                raise RuntimeError(f"不能安全迁移参数 {key}: {old_dims} -> {new_dims}")
            slices = tuple(slice(0, n) for n in old_dims)
            new_tensor[slices].copy_(old_tensor)
            resized += 1
    if resized == 0:
        raise RuntimeError("词表增长但没有对应扩容的参数，拒绝训练")
    bigger.load_state_dict(dst)
    return bigger, new_vocab, new_cfg, new_chars


def eval_loss(model, vocab, cases, device, context):
    if not cases:
        return None
    total, tokens = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for question, expected in cases:
            x, y = collate([make_example(vocab, question, expected, context)],
                           vocab.stoi["□"], device)
            logits = model(x)["logits"].reshape(-1, len(vocab.chars))
            total += F.cross_entropy(logits, y.reshape(-1), ignore_index=-100,
                                     reduction="sum").item()
            tokens += (y != -100).sum().item()
    return total / max(tokens, 1)


def assess(model, vocab, old_cases, new_cases, device, context):
    model.eval()
    scores = {}
    for prefix, cases in (("old", old_cases), ("new", new_cases)):
        correct = 0
        examples = []
        for question, expected in cases:
            prediction = answer(model, vocab, question, device,
                                max_new_tokens=min(80, max(len(expected) + 2, 16)))
            correct += prediction == expected
            examples.append({"question": question, "expected": expected,
                             "predicted": prediction, "match": prediction == expected})
        scores[f"{prefix}_exact"] = correct
        scores[f"{prefix}_total"] = len(cases)
        scores[f"{prefix}_loss"] = eval_loss(model, vocab, cases, device, context)
        scores[f"{prefix}_examples"] = examples
    return scores


def _old_cases():
    # Keep a fixed comparison suite across successive update sessions.
    # Reported preservation applies only to this suite, never to all knowledge.
    return list(DEV)


def train(args):
    if not Path(args.data).is_file():
        raise FileNotFoundError(f"未找到已审核数据：{args.data}")
    if args.steps < 1 or args.batch_size < 2 or args.eval_every < 1:
        raise ValueError("steps/eval_every >= 1, batch_size >= 2")
    if not 0 < args.replay_ratio < 1:
        raise ValueError("replay-ratio 必须介于 0 和 1 之间")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    base = Path(args.base)
    if not base.is_file() and base.name == "srt_m15_qa_continual.pt":
        base = base.with_name("srt_m15_qa.pt")
    if not base.is_file():
        raise FileNotFoundError("先运行 python train_qa.py，生成监督问答模型权重")
    model, vocab, cfg, state = load(base, device)
    if not str(state.get("qa_training", "")).startswith("masked_answer"):
        raise ValueError("当前检查点不是答案监督训练权重；请先运行 train_qa.py")
    verified = read_verified_jsonl(args.data)
    if not verified:
        raise ValueError("数据集中没有 verified=true 的已审核样本")
    new_train, new_dev = split_verified(verified, seed=args.seed)
    if not new_train or not new_dev:
        raise ValueError("至少准备同一答案的两个不同问题，分别用于训练和留出验证；未通过时不训练")

    old_train = list(training_pairs())
    # Replay past accepted updates when available; don't train on heldout pairs.
    for pair in state.get("replay_pairs", []):
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            old_train.append((pair[0], pair[1]))
    old_dev = _old_cases()
    # Re-test past accepted holdouts: replaying the trained samples alone does
    # not establish that earlier generalization has survived.
    old_dev.extend((q, a) for q, a in state.get("heldout_pairs", [])
                   if isinstance(q, str) and isinstance(a, str))
    old_dev = list({normalized_question(q): (q, a) for q, a in old_dev}.values())
    train_pairs = [(p.question, p.answer) for p in new_train]
    dev_pairs = [(p.question, p.answer) for p in new_dev]
    blocked = {normalized_question(q) for q, _ in old_dev + dev_pairs}
    leaked = [(q, a) for q, a in old_train + train_pairs if normalized_question(q) in blocked]
    if leaked:
        raise ValueError("训练数据与独立验证问题有交叉，请从新数据中移除重叠样本：" + str(leaked[:3]))
    # Re-use vocabulary from the previous model, then expand only missing rows.
    model, vocab, cfg, added = extend_vocab(
        model, vocab, cfg, old_train + old_dev + train_pairs + dev_pairs, device)
    replay_examples = [make_example(vocab, q, a, cfg.max_length) for q, a in old_train]
    fresh_examples = [make_example(vocab, q, a, cfg.max_length) for q, a in train_pairs]
    baseline = assess(model, vocab, old_dev, dev_pairs, device, cfg.max_length)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = None
    best_score = None
    report = {"base_checkpoint": str(base), "data": str(args.data),
              "sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
              "added_characters": added, "replay_count": len(old_train),
              "new_train_count": len(new_train), "new_dev_count": len(new_dev),
              "baseline": baseline, "best": None, "promoted": False}
    print(f"device={device} replay={len(replay_examples)} new_train={len(fresh_examples)} "
          f"heldout={len(dev_pairs)} vocab_added={len(added)}", flush=True)
    print("训练前：old exact", baseline["old_exact"], "/", baseline["old_total"],
          "new exact", baseline["new_exact"], "/", baseline["new_total"], flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        replay_n = min(args.batch_size - 1, max(1, round(args.batch_size * args.replay_ratio)))
        samples = (random.choices(replay_examples, k=replay_n) +
                   random.choices(fresh_examples, k=args.batch_size - replay_n))
        random.shuffle(samples)
        x, y = collate(samples, vocab.stoi["□"], device)
        logits = model(x)["logits"]
        loss = F.cross_entropy(logits.reshape(-1, len(vocab.chars)),
                               y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            candidate = assess(model, vocab, old_dev, dev_pairs, device, cfg.max_length)
            ok, reason = promotion_gate(baseline, candidate)
            print(f"step={step} train_loss={loss.item():.4f} "
                  f"old={candidate['old_exact']}/{candidate['old_total']} "
                  f"new_holdout={candidate['new_exact']}/{candidate['new_total']} "
                  f"gate={'PASS' if ok else 'REJECT'} {reason}", flush=True)
            # Don't pick a worse candidate merely because it was evaluated later.
            quality = (candidate["new_exact"], -candidate["new_loss"],
                       candidate["old_exact"], -candidate["old_loss"])
            if ok and (best_score is None or quality > best_score):
                best_score = quality
                best = {name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()}
                report["best"] = {"step": step, "metrics": candidate, "reason": reason}
    output = Path(args.output)
    report["promoted"] = best is not None
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if best is None:
        print(f"没有权重通过验证，原权重未修改。报告：{report_path}", flush=True)
        return False
    all_pairs = list({normalized_question(q): (q, a)
                      for q, a in old_train + train_pairs}.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    history = list(state.get("learning_history", []))
    history.append({"training_data_sha256": report["sha256"],
                    "step": report["best"]["step"],
                    "new_heldout_correct": report["best"]["metrics"]["new_exact"],
                    "old_correct": report["best"]["metrics"]["old_exact"]})
    torch.save({"model": best, "config": asdict(cfg), "vocab": vocab.chars,
                "qa_training": "masked_answer_v2_continual",
                "replay_pairs": [list(pair) for pair in all_pairs],
                "heldout_pairs": [list(pair) for pair in old_dev + dev_pairs],
                "learning_history": history,
                "evaluation": report["best"]["metrics"]}, temp)
    os.replace(temp, output)
    print(f"验证通过，已保存：{output}；报告：{report_path}", flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRT-M15 verified continual learning")
    parser.add_argument("--data", default="learning_data/verified_qa.jsonl")
    parser.add_argument("--base", default="checkpoints/srt_m15_qa_continual.pt")
    parser.add_argument("--output", default="checkpoints/srt_m15_qa_continual.pt")
    parser.add_argument("--report", default="checkpoints/learning_report.json")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--replay-ratio", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    train(args)