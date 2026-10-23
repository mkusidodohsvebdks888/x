"""Train an SRT-M15 toy arithmetic model FROM RANDOM INITIALIZATION.

Example: python train.py --steps 350 --device cpu
"""
import argparse
import json
import random
from pathlib import Path
from dataclasses import asdict
import torch
import torch.nn.functional as F
from srt_m15 import ArithmeticTokenizer, ModelConfig, SRTM15
from srt_m15.tasks import make_split
from srt_m15.learning import accuracy
from srt_m15.model import learning_relation_loss


def run(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    train_pairs, test_pairs = make_split(args.seed)
    distributed = args.task == "distributed"
    cfg = ModelConfig(vocab_size=ArithmeticTokenizer.VOCAB_SIZE,
                      num_classes=ArithmeticTokenizer.NUM_CLASSES,
                      dim=args.dim, reasoning_steps=args.reasoning_steps)
    model = SRTM15(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.005)
    print(f"device={device}  parameters={sum(p.numel() for p in model.parameters()):,}")
    metrics = []
    for step in range(args.steps + 1):
        model.train()
        samples = random.choices(train_pairs, k=args.batch_size)
        inputs, targets = ArithmeticTokenizer.batch(samples, device, distributed=distributed)
        result = model(inputs)
        ce = F.cross_entropy(result["receiver_logits"] if distributed else result["logits"], targets)
        auxiliary = F.cross_entropy(result["expert_logits"].reshape(-1, cfg.num_classes),
                                    targets[:, None].expand(-1, cfg.num_experts).reshape(-1))
        relation = learning_relation_loss(result["learning_relation_logits"], result["expert_logits"], targets)
        if distributed:
            # Expert A knows only operand a, B knows only operand b, C knows neither.
            # Supervising A/B on the sum would be contradictory; train C as receiver.
            loss = ce + args.comm_penalty * result["communication_cost"]
        else:
            loss = ce + 0.12 * auxiliary + 0.04 * relation + args.comm_penalty * result["communication_cost"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps:
            train_acc = accuracy(model, train_pairs, device, distributed=distributed)
            heldout_acc = accuracy(model, test_pairs, device, distributed=distributed)
            row = {"step": step, "loss": round(float(loss.item()), 4),
                   "train_accuracy": round(train_acc, 4),
                   "heldout_accuracy": round(heldout_acc, 4)}
            metrics.append(row)
            print(json.dumps(row, ensure_ascii=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.cpu().state_dict(), "config": asdict(cfg),
                "seed": args.seed, "train_pairs": train_pairs,
                "test_pairs": test_pairs, "task": args.task}, output)
    (output.parent / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--task", choices=["distributed", "addition"], default="distributed",
                        help="distributed splits operands and reads answer only from expert C")
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--reasoning-steps", type=int, default=3)
    parser.add_argument("--comm-penalty", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="checkpoints/srt_m15_toy.pt")
    run(parser.parse_args())
