"""Validate and stratify Chinese/English SFT JSONL before neural training.

Only explicit verified=True records with nonempty sources are accepted in normal
mode. Bundled toy examples can be used solely with allow_demo=True.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import random
import unicodedata


LANGUAGES = ("zh", "en")


@dataclass(frozen=True)
class BilingualRecord:
    language: str
    prompt: str
    response: str
    source: str
    split: str = "auto"
    topic_id: str = ""


def normalize_prompt(value):
    return "".join(ch for ch in unicodedata.normalize("NFKC", value).casefold()
                   if ch.isalnum())


def load_records(path, *, allow_demo=False):
    records = []
    seen = {}
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"row {index}: invalid JSON") from exc
        if not isinstance(item, dict):
            raise ValueError(f"row {index}: expected object")
        lang, prompt, response = (item.get(k) for k in ("language", "prompt", "response"))
        if lang not in LANGUAGES or not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError(f"row {index}: language must be zh/en and texts must be strings")
        prompt, response = prompt.strip(), response.strip()
        if not (1 <= len(prompt) <= 1024 and 1 <= len(response) <= 2048):
            raise ValueError(f"row {index}: empty or excessive prompt/response")
        split = item.get("split", "auto")
        if split not in ("auto", "train", "dev"):
            raise ValueError(f"row {index}: invalid split")
        if not allow_demo:
            if item.get("verified") is not True:
                raise ValueError(f"row {index}: verified=true is required")
            if not isinstance(item.get("source"), str) or not item["source"].strip():
                raise ValueError(f"row {index}: source is required")
        key = normalize_prompt(prompt)
        if len(key) < 1 or key in seen:
            raise ValueError(f"row {index}: empty/duplicate prompt (previous row {seen.get(key)})")
        seen[key] = index
        topic_id = item.get("topic_id", "")
        if not isinstance(topic_id, str) or len(topic_id) > 100:
            raise ValueError(f"row {index}: invalid topic_id")
        records.append(BilingualRecord(lang, prompt, response,
                                      item.get("source") or "unverified-demo",
                                      split, topic_id.strip()))
    if not records:
        raise ValueError("no examples")
    return records


def split_records(records, *, seed=42, dev_fraction=0.2):
    """Each language must independently have train and dev records."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    train, dev = [], []
    rng = random.Random(seed)
    # For parallel Chinese/English topics, keep *both translations* entirely
    # in the same split. Splitting translated equivalents across train/dev
    # would inflate bilingual validation scores via answer/topic leakage.
    if any(r.topic_id for r in records):
        if any(not r.topic_id for r in records):
            raise ValueError("mixed paired/unpaired records: all need topic_id")
        groups = {}
        for record in records:
            groups.setdefault(record.topic_id, []).append(record)
        if len(groups) < 4:
            raise ValueError("at least four paired topics are required")
        for topic_id, group in groups.items():
            if len(group) != 2 or {r.language for r in group} != set(LANGUAGES):
                raise ValueError(f"paired topic {topic_id} must contain one zh and one en")
            if group[0].split != group[1].split:
                raise ValueError(f"paired topic {topic_id} has conflicting train/dev split")
        fixed_train = [k for k, v in groups.items() if v[0].split == "train"]
        fixed_dev = [k for k, v in groups.items() if v[0].split == "dev"]
        auto = sorted(k for k, v in groups.items() if v[0].split == "auto")
        rng.shuffle(auto)
        target_dev = max(1, round(len(groups) * dev_fraction))
        dev_count = max(0, target_dev - len(fixed_dev))
        dev_keys = set(fixed_dev + auto[:dev_count])
        train_keys = set(fixed_train + auto[dev_count:])
        if not dev_keys or not train_keys:
            raise ValueError("paired data need both train and dev topics")
        train = [r for k in sorted(train_keys) for r in groups[k]]
        dev = [r for k in sorted(dev_keys) for r in groups[k]]
        return train, dev
    for lang in LANGUAGES:
        group = [r for r in records if r.language == lang]
        if len(group) < 4:
            raise ValueError(f"at least four records for {lang} are required")
        fixed_train = [r for r in group if r.split == "train"]
        fixed_dev = [r for r in group if r.split == "dev"]
        auto = [r for r in group if r.split == "auto"]
        rng.shuffle(auto)
        target_dev = max(1, round(len(group) * dev_fraction))
        count_dev = max(0, target_dev - len(fixed_dev))
        by_lang_dev = fixed_dev + auto[:count_dev]
        by_lang_train = fixed_train + auto[count_dev:]
        if not by_lang_train or not by_lang_dev:
            raise ValueError(f"{lang} needs independent train and dev data")
        train.extend(by_lang_train)
        dev.extend(by_lang_dev)
    train_ids = {normalize_prompt(r.prompt) for r in train}
    dev_ids = {normalize_prompt(r.prompt) for r in dev}
    if train_ids & dev_ids:
        raise ValueError("train/dev prompt leakage")
    return train, dev


def language_counts(records):
    return {lang: sum(r.language == lang for r in records) for lang in LANGUAGES}
