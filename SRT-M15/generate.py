"""Generate character text from a trained SRT-M15LM checkpoint."""
import argparse
import torch
from srt_m15 import ModelConfig
from srt_m15.language import SRTM15LanguageModel, CharVocabulary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/srt_m15_lm.pt")
    p.add_argument("--prompt", default="你好")
    p.add_argument("--new-tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    a = p.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    c = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    vocab = CharVocabulary(c["vocab"])
    model = SRTM15LanguageModel(ModelConfig(**c["config"]))
    model.load_state_dict(c["model"])
    tokens = torch.tensor([vocab.encode(a.prompt)], dtype=torch.long)
    print(vocab.decode(model.generate(tokens, a.new_tokens, temperature=a.temperature)[0].tolist()))
