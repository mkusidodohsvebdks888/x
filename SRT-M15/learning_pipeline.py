"""Verified QA data ingestion, leakage-safe splitting, and learning promotion rules.

This file is standard-library-only; it never treats model-generated text as truth.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import random
import unicodedata


@dataclass(frozen=True)
class VerifiedQA:
    question: str
    answer: str
    source: str
    split: str = "auto"


def normalized_question(text: str) -> str:
    """Merge trivial punctuation, case and whitespace variants for leak detection."""
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def read_verified_jsonl(path: str | Path) -> list[VerifiedQA]:
    """Only import explicitly verified samples. Unverified rows are skipped.

    Repeated normalized questions must not have contradictory answers or splits.
    """
    path = Path(path)
    result: dict[str, VerifiedQA] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: record must be an object")
            # No implicit trust: assistant-generated or proposed QA needs review.
            if record.get("verified") is not True:
                continue
            question, answer, source = (
                record.get("question"), record.get("answer"), record.get("source")
            )
            if not all(isinstance(x, str) for x in (question, answer, source)):
                raise ValueError(f"{path}:{line_no}: question, answer, source must be strings")
            question, answer, source = question.strip(), answer.strip(), source.strip()
            if not (question and answer and source):
                raise ValueError(f"{path}:{line_no}: required text must not be empty")
            if len(question) > 180 or len(answer) > 400:
                raise ValueError(f"{path}:{line_no}: sample exceeds configured limits")
            split = record.get("split", "auto")
            if split not in ("auto", "train", "dev"):
                raise ValueError(f"{path}:{line_no}: split must be auto/train/dev")
            key = normalized_question(question)
            if not key:
                raise ValueError(f"{path}:{line_no}: question has no meaningful characters")
            value = VerifiedQA(question, answer, source, split)
            previous = result.get(key)
            if previous:
                if previous.answer != answer:
                    raise ValueError(f"{path}:{line_no}: conflicting answer for {question!r}")
                if previous.split != split and "auto" not in (previous.split, split):
                    raise ValueError(f"{path}:{line_no}: train/dev leakage for {question!r}")
                if previous.split == "auto" and split != "auto":
                    result[key] = value
            else:
                result[key] = value
    return list(result.values())


def split_verified(pairs: list[VerifiedQA], *, seed: int = 42,
                   dev_fraction: float = 0.25) -> tuple[list[VerifiedQA], list[VerifiedQA]]:
    """Hold out unseen *questions*, never copies of a train question.

    For auto-split examples, reserve paraphrases from answer groups that have
    at least two different questions. An answer observed only once remains train;
    do not pretend that this provides a valid generalization test.
    """
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between zero and one")
    by_key: dict[str, VerifiedQA] = {}
    for pair in pairs:
        key = normalized_question(pair.question)
        if key in by_key and by_key[key] != pair:
            raise ValueError(f"duplicate/leaked question: {pair.question}")
        by_key[key] = pair
    train = [p for p in by_key.values() if p.split == "train"]
    dev = [p for p in by_key.values() if p.split == "dev"]
    auto = [p for p in by_key.values() if p.split == "auto"]
    groups: dict[str, list[VerifiedQA]] = {}
    for pair in auto:
        groups.setdefault(pair.answer, []).append(pair)
    rng = random.Random(seed)
    for answer in sorted(groups):
        items = sorted(groups[answer], key=lambda p: normalized_question(p.question))
        rng.shuffle(items)
        # Explicit train examples for the same answer can anchor auto dev cases.
        existing_train = any(p.answer == answer for p in train)
        if len(items) >= 2:
            ndev = max(1, int(round(len(items) * dev_fraction)))
            if not existing_train:
                ndev = min(ndev, len(items) - 1)
        elif existing_train:
            ndev = 1
        else:
            ndev = 0
        dev.extend(items[:ndev])
        train.extend(items[ndev:])
    train_keys = {normalized_question(p.question) for p in train}
    dev_keys = {normalized_question(p.question) for p in dev}
    if train_keys & dev_keys:
        raise ValueError("train/dev overlap detected")
    return train, dev


def promotion_gate(baseline: dict, candidate: dict, *, old_loss_tolerance: float = 0.05) -> tuple[bool, str]:
    """Require preserved old-task exact score and demonstrable heldout progress.

    Loss is token-averaged. There is no claim that this guarantees broad safety
    or generalization beyond the supplied benchmark.
    """
    required = ("old_exact", "old_loss", "new_exact", "new_loss", "new_total")
    for key in required:
        if key not in baseline or key not in candidate:
            raise ValueError(f"missing metric: {key}")
    if baseline["new_total"] <= 0 or candidate["new_total"] <= 0:
        return False, "没有独立的新增知识验证集，不允许自动更新正式权重"
    if candidate["old_exact"] < baseline["old_exact"]:
        return False, "旧任务正确数量下降，拒绝更新（防遗忘）"
    if candidate["old_loss"] > baseline["old_loss"] + old_loss_tolerance:
        return False, "旧任务损失增加过多，拒绝更新（防遗忘）"
    if candidate["new_exact"] <= baseline["new_exact"]:
        return False, "新任务留出集没有新增正确答案，拒绝将损失微降误当成掌握知识"
    return True, "旧任务未显著退化且留出集有所改进，可保存候选权重"