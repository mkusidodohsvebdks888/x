"""Small scratch-initialized character language model (no pretrained weights).

Run: python train_lm.py --steps 200
"""
import argparse
from pathlib import Path
import random
from dataclasses import asdict
import torch
import torch.nn.functional as F
from srt_m15 import ModelConfig
from srt_m15.language import SRTM15LanguageModel, CharVocabulary

CORPUS = (
    "你好，我是正在学习语言规律的小型实验模型。\n"
    "神经网络可以在隐藏状态中进行多轮计算。\n"
    "我们研究神经网络之间的信息传递和动态记忆。\n"
    "计算需要可靠的验证，学习需要高质量数据。\n"
    "这是一个从随机参数开始训练的实验，不是成熟的聊天系统。\n"
) * 75


def run(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    corpus = Path(args.corpus).read_text(encoding="utf-8") if args.corpus else CORPUS
    vocab = CharVocabulary.from_text(corpus)
    corpus_ids = torch.tensor(vocab.encode(corpus), dtype=torch.long, device=device)
    cfg = ModelConfig(vocab_size=len(vocab.chars), num_classes=len(vocab.chars),
                      dim=args.dim, max_length=args.context, reasoning_steps=args.reasoning_steps)
    model = SRTM15LanguageModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"device={device}, vocab={len(vocab.chars)}, params={sum(p.numel() for p in model.parameters()):,}")
    for step in range(args.steps + 1):
        starts = torch.randint(0, len(corpus_ids) - args.context - 1, (args.batch_size,), device=device)
        tokens = torch.stack([corpus_ids[i:i+args.context] for i in starts])
        targets = torch.stack([corpus_ids[i+1:i+args.context+1] for i in starts])
        result = model(tokens)
        loss = F.cross_entropy(result["logits"].reshape(-1, len(vocab.chars)), targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps:
            print(f"step={step}, loss={loss.item():.4f}")
    output = Path(args.output)
    output.parent.mkdir(exist_ok=True, parents=True)
    torch.save({"model": model.cpu().state_dict(), "config": asdict(cfg),
                "vocab": vocab.chars}, output)
    model.eval()
    prompt = "你好"
    if all(ch in vocab.stoi for ch in prompt):
        ids = torch.tensor([vocab.encode(prompt)], dtype=torch.long)
        print("example:", vocab.decode(model.generate(ids, max_new_tokens=40)[0].tolist()))
    print("saved:", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context", type=int, default=24)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--reasoning-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--corpus", type=str, default="", help="optional UTF-8 file for training data")
    p.add_argument("--output", default="checkpoints/srt_m15_lm.pt")
    run(p.parse_args())
