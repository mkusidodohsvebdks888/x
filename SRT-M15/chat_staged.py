"""Interactive local test of a staged SRT-M15 language checkpoint.

This script does not train; use train_bilingual_stages.py first.
Generated replies from this small character-level model may be incorrect.
"""
import argparse
from pathlib import Path

import torch

from srt_m15 import ModelConfig
from srt_m15.language import CharVocabulary, SRTM15LanguageModel


def load_checkpoint(path, device):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "srt-m15-bilingual-staged-v1":
        raise ValueError("Expected train_bilingual_stages.py checkpoint")
    vocabulary = CharVocabulary(checkpoint["vocab"])
    model = SRTM15LanguageModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, vocabulary


@torch.no_grad()
def reply(model, vocab, turns, user_text, limit=100):
    # Shared supervised trainer uses Chinese role markers for both languages.
    text = "".join(f"用户：{u}\n助手：{a}\n" for u, a in turns)
    text += f"用户：{user_text}\n助手："
    try:
        tokens = vocab.encode(text)
    except ValueError as exc:
        return None, str(exc)
    if len(tokens) >= model.cfg.max_length:
        # The tiny research model has a finite attention window.
        tokens = tokens[-model.cfg.max_length:]
    sequence = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)
    chars = []
    for _ in range(limit):
        prediction = model(sequence[:, -model.cfg.max_length:])["logits"][0, -1, :]
        token = int(prediction.argmax().item())
        character = vocab.decode([token])
        if character == "\n":
            break
        chars.append(character)
        next_token = torch.tensor([[token]], dtype=torch.long, device=sequence.device)
        sequence = torch.cat((sequence, next_token), 1)
    return "".join(chars).strip(), None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/local_dialogues/latest.pt")
    p.add_argument("--prompt", default=None, help="Run once; omit for interactive chat")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()
    if not 1 <= args.max_new_tokens <= 256:
        p.error("--max-new-tokens must be 1..256")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is not available")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model, vocab = load_checkpoint(args.checkpoint, device)
    print("SRT-M15 experimental local chat; /reset clears history; /exit quits.")
    print("Unseen characters will be rejected rather than silently replaced.")
    turns = []
    while True:
        if args.prompt is not None:
            question = args.prompt
        else:
            try:
                question = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if question == "/exit":
            break
        if question == "/reset":
            turns.clear()
            if args.prompt is not None:
                break
            print("History reset")
            continue
        if not question:
            if args.prompt is not None:
                break
            continue
        answer, err = reply(model, vocab, turns, question, limit=args.max_new_tokens)
        if err:
            print("Input rejected:", err)
        else:
            print("SRT-M15>", answer if answer else "[empty reply]")
            if answer:
                turns.append((question, answer))
                turns = turns[-4:]
        if args.prompt is not None:
            break


if __name__ == "__main__":
    main()
