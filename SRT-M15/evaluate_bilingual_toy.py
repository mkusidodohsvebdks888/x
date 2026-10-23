"""Evaluate the original from-scratch SRT-M15 toy checkpoint per language.

Metrics are HELD-OUT next-character loss/accuracy, not fluent dialogue scores.
Requires same seed and validation_fraction as train_common.py.
"""
import argparse
import json
from pathlib import Path
import torch

from bilingual_data import load_records, language_counts
from srt_m15 import ModelConfig
from srt_m15.language import SRTM15LanguageModel, CharVocabulary
from srt_m15.common_mechanisms import (
    instruction_example, split_examples, make_windows, evaluate_language,
)


def run(checkpoint_path, data_path, *, seed=42, fraction=0.2, output=None):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state.get("metadata", {}).get("mode") != "sft":
        raise ValueError("SFT checkpoint required")
    cfg = ModelConfig(**state["config"])
    model = SRTM15LanguageModel(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    vocab = CharVocabulary(state["vocab"])

    samples = load_records(data_path, allow_demo=True)
    examples = [instruction_example(s.prompt, s.response) for s in samples]
    _, holdout = split_examples(examples, validation_fraction=fraction, seed=seed)
    text_to_language = {instruction_example(x.prompt, x.response).text: x.language
                        for x in samples}
    report = {
        "model": "SRT-M15 from-scratch toy, not a broadly fluent language model",
        "seed": seed, "validation_fraction": fraction, "metrics": {}
    }
    for language in ("zh", "en"):
        chosen = [x for x in holdout if text_to_language[x.text] == language]
        if not chosen:
            raise ValueError(f"no held-out examples in {language}")
        windows = make_windows(chosen, vocab, cfg.max_length)
        metrics = evaluate_language(model, windows, device="cpu")
        report["metrics"][language] = {"examples": len(chosen), **metrics}
    report["total_examples"] = len(samples)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/bilingual_toy_sft.pt")
    p.add_argument("--data", default="examples/bilingual_toy_sft.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fraction", type=float, default=0.2)
    p.add_argument("--output", default="checkpoints/bilingual_toy_eval.json")
    a = p.parse_args()
    run(a.checkpoint, a.data, seed=a.seed, fraction=a.fraction, output=a.output)
