"""Shared, executable LLM mechanisms for the SRT-M15 research prototype.

Uses the EXISTING SRTM15LanguageModel for embeddings, causal attention,
reasoning, and token logits. Adds reproducible preparation, teacher-forced
next-token / supervised-instruction training, held-out evaluation, and sampling.
This module is an educational pipeline, not a production-scale LLM trainer.
"""
from dataclasses import dataclass
import math
import random

import torch
import torch.nn.functional as F

from .language import CharVocabulary


IGNORE_INDEX = -100


@dataclass(frozen=True)
class LanguageExample:
    text: str
    target_start: int = 1

    def __post_init__(self):
        if len(self.text) < 2:
            raise ValueError("an example must contain at least two characters")
        if not (1 <= self.target_start < len(self.text)):
            raise ValueError("target_start must identify a supervised target character")


def instruction_example(prompt: str, response: str) -> LanguageExample:
    """Supervise only the assistant response, not the user's question."""
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise TypeError("prompt and response must be strings")
    if not prompt.strip() or not response.strip():
        raise ValueError("prompt and response must be nonempty")
    prefix = "用户：" + prompt.strip() + "\n助手："
    return LanguageExample(prefix + response.strip() + "\n", len(prefix))


def split_examples(examples, *, validation_fraction=0.2, seed=42):
    """Split by example BEFORE making overlapping windows, avoiding label leakage."""
    examples = list(examples)
    if len(examples) < 2:
        raise ValueError("at least two independent examples are needed for a holdout")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    indexes = list(range(len(examples)))
    random.Random(seed).shuffle(indexes)
    n_valid = min(len(examples) - 1, max(1, round(len(examples) * validation_fraction)))
    return ([examples[i] for i in indexes[n_valid:]],
            [examples[i] for i in indexes[:n_valid]])


def character_vocabulary(train_examples, validation_examples=()):
    """Fit a character inventory; validation labels are never used for updates.

    In this simple char-level prototype, both splits supply *symbol inventory*
    so out-of-vocabulary validation characters do not crash evaluation.
    This differs from strict train-only tokenizer training.
    """
    all_examples = list(train_examples) + list(validation_examples)
    if not all_examples:
        raise ValueError("examples cannot be empty")
    return CharVocabulary.from_text("".join(ex.text for ex in all_examples))


def make_windows(examples, vocab: CharVocabulary, context: int):
    """Return (input_ids, labels) pairs, both [context], labels padded with -100.

    Each example is kept separate; no window crosses the train/validation split.
    Padding is RIGHT-only, so the causal attention cannot see padding when
    predicting real tokens. Instruction prompt tokens are loss-masked.
    """
    if context < 2:
        raise ValueError("context must be at least 2")
    result = []
    for example in examples:
        ids = vocab.encode(example.text)
        for start in range(0, len(ids) - 1, context):
            usable = min(context, len(ids) - 1 - start)
            input_ids = ids[start:start + usable] + [0] * (context - usable)
            labels = [
                ids[start + i + 1] if start + i + 1 >= example.target_start else IGNORE_INDEX
                for i in range(usable)
            ] + [IGNORE_INDEX] * (context - usable)
            if any(label != IGNORE_INDEX for label in labels):
                result.append((torch.tensor(input_ids, dtype=torch.long),
                               torch.tensor(labels, dtype=torch.long)))
    if not result:
        raise ValueError("no supervised tokens: check context and examples")
    return result


def masked_token_loss(logits, labels):
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits must be [batch, context, vocabulary], labels [batch, context]")
    if not bool((labels != IGNORE_INDEX).any().item()):
        raise ValueError("cannot compute loss on an all-masked batch")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           labels.reshape(-1), ignore_index=IGNORE_INDEX)


@torch.no_grad()
def evaluate_language(model, windows, *, batch_size=8, device="cpu"):
    """Compute held-out token loss, perplexity, and next-token accuracy."""
    if not windows or batch_size < 1:
        raise ValueError("evaluation requires windows and a positive batch size")
    was_training = model.training
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    try:
        for i in range(0, len(windows), batch_size):
            part = windows[i:i + batch_size]
            x = torch.stack([p[0] for p in part]).to(device)
            y = torch.stack([p[1] for p in part]).to(device)
            logits = model(x)["logits"]
            valid = y != IGNORE_INDEX
            count = int(valid.sum().item())
            total_loss += float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                ignore_index=IGNORE_INDEX, reduction="sum").item())
            correct += int(((logits.argmax(-1) == y) & valid).sum().item())
            total += count
    finally:
        model.train(was_training)
    if total == 0:
        raise ValueError("evaluation contains no supervised tokens")
    mean_loss = total_loss / total
    return {"token_loss": mean_loss,
            "perplexity": math.exp(min(mean_loss, 80.0)),
            "token_accuracy": correct / total,
            "supervised_tokens": total}


def train_language_steps(model, train_windows, validation_windows, *,
                         steps=100, batch_size=8, lr=0.001, seed=42,
                         device="cpu", log_every=20, max_grad_norm=1.0):
    """AdamW + cross entropy + backprop + clipping + independent evaluation.

    Evaluation never modifies weights. This function does NOT implement RLHF,
    DPO, or any claim of autonomous learning.
    """
    if steps < 1 or batch_size < 1 or lr <= 0 or log_every < 1:
        raise ValueError("steps, batch_size, log_every, lr must be positive")
    if not train_windows or not validation_windows:
        raise ValueError("training and validation windows are both required")
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    for step in range(1, steps + 1):
        model.train()
        batch = rng.choices(train_windows, k=batch_size)
        x = torch.stack([v[0] for v in batch]).to(device)
        y = torch.stack([v[1] for v in batch]).to(device)
        logits = model(x)["logits"]
        loss = masked_token_loss(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            metrics = evaluate_language(model, validation_windows,
                                        batch_size=batch_size, device=device)
            history.append({"step": step, "train_batch_loss": float(loss.item()),
                            "validation": metrics})
    return history


@torch.no_grad()
def sample_tokens(model, prompt_ids, *, max_new_tokens=30, temperature=1.0,
                  top_k=0, top_p=1.0, generator=None):
    """Temperature, top-k, nucleus sampling; no gradient updates at inference."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[1] < 1:
        raise ValueError("prompt_ids must be nonempty [batch, sequence]")
    if temperature <= 0 or top_k < 0 or not 0 < top_p <= 1 or max_new_tokens < 0:
        raise ValueError("invalid sampling parameters")
    model.eval()
    output = prompt_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(output[:, -model.cfg.max_length:])["logits"][:, -1, :] / temperature
        if top_k:
            k = min(top_k, logits.shape[-1])
            threshold = logits.topk(k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < threshold, -float("inf"))
        if top_p < 1.0:
            sorted_logits, sorted_index = logits.sort(dim=-1, descending=True)
            cumprob = sorted_logits.softmax(dim=-1).cumsum(-1)
            remove = (cumprob - sorted_logits.softmax(dim=-1)) >= top_p
            sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
            logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_index, sorted_logits)
        next_ids = torch.multinomial(logits.softmax(dim=-1), 1, generator=generator)
        output = torch.cat((output, next_ids), dim=1)
    return output
