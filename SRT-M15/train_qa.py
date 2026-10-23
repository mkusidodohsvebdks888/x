"""Supervised character-level QA training for the SRT-M15 research model.

No external LLM, pretrained weights, or hard-coded identity reply at inference.
Question/prompt characters are context; only answer tokens contribute to loss.
The tiny model can still fail, especially on paraphrases not seen in training.
"""
import argparse
import random
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from srt_m15 import ModelConfig
from srt_m15.language import CharVocabulary, SRTM15LanguageModel


IDENTITY_ANSWER = "我是SRT-M15，一个从零训练的小型Transformer实验模型。"
IDENTITY_TRAIN = [
    "你是谁", "你是谁呀", "你叫什么", "你叫什么名字",
    "你的名字是什么", "你的身份是什么", "你是什么模型",
    "你是哪个模型", "你是什么AI", "请介绍一下你自己",
    "介绍一下你自己", "自我介绍", "你是什么",
]
OTHER_TRAIN = [
    ("你好", "你好！"),
    ("早上好", "早上好！"),
    ("你会什么", "我可以尝试字符级问答和简单加法。"),
    ("你会算数学吗", "我正在学习简单计算。"),
    ("什么是模型", "模型是学习数据规律的计算系统。"),
    ("什么是训练", "训练通过调整参数来减小误差。"),
    ("什么是推理", "推理是利用模型参数预测结果。"),
    ("什么是人工智能", "人工智能研究如何让机器完成智能任务。"),
    ("什么是记忆", "记忆用于保存和使用信息。"),
    ("你能聊天吗", "我能尝试简单问答，但能力有限。"),
    ("你是ChatGPT吗", "不是，我是SRT-M15实验模型。"),
    ("你是GPT吗", "不是，我是SRT-M15实验模型。"),
    ("为什么会回答错", "可能因为训练不足或输入超出能力范围。"),
]
# Development examples are excluded from training. Sharing the identity *answer*
# tests unseen question phrasing, not ability to discover facts on its own.
DEV = [
    ("你到底是谁", IDENTITY_ANSWER),
    ("能介绍你自己吗", IDENTITY_ANSWER),
    ("你自己的名字是什么", IDENTITY_ANSWER),
    ("你好呀", "你好！"),
    ("你能做加法吗", "我正在学习简单计算。"),
]


def training_pairs():
    return [(q, IDENTITY_ANSWER) for q in IDENTITY_TRAIN] + OTHER_TRAIN


def make_vocab():
    # Include development characters for evaluation without unknown-token errors.
    text = "".join("问：" + q + "\n答：" + a + "\n" for q, a in training_pairs() + DEV)
    return CharVocabulary.from_text(text + "□")


def make_example(vocab, question, answer, context):
    prompt = f"问：{question}\n答："
    whole = prompt + answer + "\n"
    if len(whole) - 1 > context:
        raise ValueError(
            f"问答长度 {len(whole) - 1} 超过 --context {context}，"
            "请增大 context 或缩短示例。"
        )
    indices = vocab.encode(whole)
    inputs = torch.tensor(indices[:-1], dtype=torch.long)
    labels = torch.tensor(indices[1:], dtype=torch.long)
    # Label position i predicts character i+1; first answer label at len(prompt)-1.
    labels[:len(prompt) - 1] = -100
    return inputs, labels


def collate(examples, pad_id, device):
    length = max(x.numel() for x, _ in examples)
    x = torch.full((len(examples), length), pad_id, dtype=torch.long)
    y = torch.full((len(examples), length), -100, dtype=torch.long)
    for idx, (tokens, targets) in enumerate(examples):
        x[idx, :len(tokens)] = tokens
        y[idx, :len(targets)] = targets
    return x.to(device), y.to(device)


@torch.inference_mode()
def answer(model, vocab, question, device, max_new_tokens=80):
    """Greedy model decoding until newline: never substitute a canned answer."""
    model.eval()
    prompt = f"问：{question}\n答："
    unknown = [c for c in prompt if c not in vocab.stoi]
    if unknown:
        return f"[词表缺少: {unknown[0]!r}]"
    seq = torch.tensor([vocab.encode(prompt)], dtype=torch.long, device=device)
    response = []
    for _ in range(max_new_tokens):
        logits = model(seq[:, -model.cfg.max_length:])["logits"][:, -1, :]
        next_id = int(logits.argmax(-1).item())
        character = vocab.chars[next_id]
        if character == "\n":
            break
        response.append(character)
        next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
        seq = torch.cat((seq, next_tensor), dim=1)
    return "".join(response)


@torch.inference_mode()
def evaluate(model, vocab, dev_examples, pad_id, device, context):
    model.eval()
    total_loss, count = 0.0, 0
    for question, reply in DEV:
        x, y = collate([make_example(vocab, question, reply, context)],
                       pad_id, device)
        logits = model(x)["logits"].reshape(-1, len(vocab.chars))
        total_loss += F.cross_entropy(logits, y.reshape(-1),
                                      ignore_index=-100, reduction="sum").item()
        count += int((y != -100).sum().item())
    cases = [
        ("你是谁", IDENTITY_ANSWER),
        ("你叫什么名字", IDENTITY_ANSWER),
        DEV[0],
        DEV[1],
        ("你是ChatGPT吗", "不是，我是SRT-M15实验模型。"),
    ]
    outputs = [(q, answer(model, vocab, q, device), expected)
               for q, expected in cases]
    identity_correct = sum(actual == expected for _, actual, expected in outputs[:4])
    dev_correct = sum(actual == expected for _, actual, expected in outputs[2:4])
    return {
        "dev_answer_loss": round(total_loss / max(count, 1), 5),
        "identity_exact": identity_correct,
        "identity_total": 4,
        "unseen_identity_exact": dev_correct,
        "unseen_identity_total": 2,
        "examples": outputs,
    }


def save(model, vocab, cfg, path, step, evaluation):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().clone()
             for name, tensor in model.state_dict().items()}
    torch.save({
        "model": state,
        "config": asdict(cfg),
        "vocab": vocab.chars,
        "qa_training": "masked_answer_v1",
        "step": step,
        "evaluation": evaluation,
    }, path)


def load(path, device):
    data = torch.load(path, map_location="cpu", weights_only=True)
    vocab = CharVocabulary(data["vocab"])
    cfg = ModelConfig(**data["config"])
    model = SRTM15LanguageModel(cfg)
    model.load_state_dict(data["model"])
    return model.to(device).eval(), vocab, cfg, data


def run(args):
    if args.steps <= 0 or args.batch_size <= 0 or args.eval_every <= 0 or args.log_every <= 0:
        raise ValueError("训练步数、batch size 和日志间隔必须大于 0")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    output = Path(args.output)

    if args.eval_only:
        model, vocab, cfg, data = load(output, device)
        print("已加载检查点：", output, "step=", data.get("step"),
              "训练方式=", data.get("qa_training", "旧式字符续写"))
        if data.get("qa_training") != "masked_answer_v1":
            print("警告：这不是新的答案监督训练检查点。")
        for question in ["你是谁", "你叫什么名字", "你到底是谁", "你是ChatGPT吗", "你好"]:
            print(f"问：{question}\n模型答：{answer(model, vocab, question, device)}")
        return

    vocab = make_vocab()
    cfg = ModelConfig(
        vocab_size=len(vocab.chars),
        num_classes=len(vocab.chars),
        dim=args.dim,
        max_length=args.context,
        reasoning_steps=args.reasoning_steps,
    )
    model = SRTM15LanguageModel(cfg).to(device)
    pad_id = vocab.stoi["□"]
    train_examples = [make_example(vocab, q, a, args.context)
                      for q, a in training_pairs()]
    dev_examples = [make_example(vocab, q, a, args.context) for q, a in DEV]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"device={device} vocab={len(vocab.chars)} "
          f"params={sum(p.numel() for p in model.parameters()):,} "
          f"train_qa={len(train_examples)} dev_qa={len(dev_examples)}")
    best_score = None

    for step in range(1, args.steps + 1):
        model.train()
        batch = random.choices(train_examples, k=args.batch_size)
        x, y = collate(batch, pad_id, device)
        logits = model(x)["logits"]
        loss = F.cross_entropy(logits.reshape(-1, len(vocab.chars)),
                               y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.log_every == 0 or step == 1:
            print(f"step={step}/{args.steps} train_answer_loss={loss.item():.4f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, vocab, dev_examples, pad_id, device, args.context)
            # Prioritize correct identity generation; break ties using dev answer loss.
            score = (metrics["identity_exact"], metrics["unseen_identity_exact"],
                     -metrics["dev_answer_loss"])
            print(f"eval step={step} identity={metrics['identity_exact']}/4 "
                  f"unseen={metrics['unseen_identity_exact']}/2 "
                  f"dev_loss={metrics['dev_answer_loss']:.4f}", flush=True)
            for q, actual, expected in metrics["examples"][:3]:
                print(f"  {q} -> {actual!r} ({'✓' if actual == expected else '未匹配'})", flush=True)
            if best_score is None or score > best_score:
                best_score = score
                save(model, vocab, cfg, output, step, metrics)
                print("  已保存当前表现最佳的实验检查点：", output, flush=True)

    print("训练结束。模型权重已保存；准确率取决于实际评测，不能保证所有提问都正确。")
    model, vocab, cfg, data = load(output, device)
    print(f"采用第 {data.get('step')} 步保存的权重")
    for question in ["你是谁", "你叫什么名字", "你到底是谁", "你是ChatGPT吗"]:
        print(f"问：{question}\n模型答：{answer(model, vocab, question, device)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SRT-M15 supervised QA training and evaluation")
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context", type=int, default=96)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--reasoning-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--output", type=str, default="checkpoints/srt_m15_qa.pt")
    p.add_argument("--eval-only", action="store_true")
    run(p.parse_args())
