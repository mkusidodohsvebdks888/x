"""Autoregressive, character-level language modeling with the same SRT-M15 core.

Demonstrates a TRAINABLE text generation path, not production chat capabilities.
"""
import torch
from .model import SRTM15, ModelConfig


class CharVocabulary:
    def __init__(self, chars):
        self.chars = list(chars)
        if len(set(self.chars)) != len(self.chars):
            raise ValueError("duplicate characters in vocabulary")
        self.stoi = {s: i for i, s in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    def encode(self, text):
        unknown = [c for c in text if c not in self.stoi]
        if unknown:
            raise ValueError(f"Character not in training vocabulary: {unknown[0]!r}")
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.chars[int(i)] for i in ids)


class SRTM15LanguageModel(SRTM15):
    """Generate next-token distributions for EVERY position without lookahead.

    A single sequence is processed in parallel at each token position. The shared
    causal backbone ensures each position only accesses its previous context.
    The recurrent experts and communication exchange within each position only.
    """
    def forward(self, tokens, *, hard_routes=False, disable_communication=False):
        b, seq = tokens.shape
        hidden = self.backbone.forward_hidden(tokens)
        output = self.reason_from_features(hidden.reshape(b * seq, -1), None,
                                           hard_routes=hard_routes,
                                           disable_communication=disable_communication)
        output["logits"] = output["logits"].reshape(b, seq, -1)
        output["expert_logits"] = output["expert_logits"].reshape(b, seq, self.cfg.num_experts, -1)
        return output

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=40, *, temperature=0.8):
        if max_new_tokens < 0 or temperature <= 0:
            raise ValueError("max_new_tokens must be nonnegative; temperature positive")
        self.eval()
        seq = prompt_ids.clone()
        for _ in range(max_new_tokens):
            prediction = self(seq[:, -self.cfg.max_length:])["logits"][:, -1] / temperature
            next_id = torch.multinomial(torch.softmax(prediction, dim=-1), 1)
            seq = torch.cat([seq, next_id], 1)
        return seq
