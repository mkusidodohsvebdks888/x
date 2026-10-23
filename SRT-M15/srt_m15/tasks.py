"""Small deterministic toy domain (single-digit addition), for scientific sanity tests."""
import random
import torch


class ArithmeticTokenizer:
    BOS = 0
    PLUS = 1
    EOS = 2
    FIRST_DIGIT = 3
    MASK = 13
    VOCAB_SIZE = 14
    NUM_CLASSES = 19

    @classmethod
    def encode(cls, a: int, b: int):
        if not (0 <= a <= 9 and 0 <= b <= 9):
            raise ValueError("prototype supports only operands 0..9")
        return [cls.BOS, cls.FIRST_DIGIT + a, cls.PLUS, cls.FIRST_DIGIT + b, cls.EOS]

    @classmethod
    def encode_distributed(cls, a: int, b: int):
        cls.encode(a, b)  # validate range
        return [
            [cls.BOS, cls.FIRST_DIGIT + a, cls.PLUS, cls.MASK, cls.EOS],
            [cls.BOS, cls.MASK, cls.PLUS, cls.FIRST_DIGIT + b, cls.EOS],
            [cls.BOS, cls.MASK, cls.PLUS, cls.MASK, cls.EOS],
        ]

    @staticmethod
    def verify(a: int, b: int):
        return a + b  # deterministic external oracle for verified feedback

    @classmethod
    def batch(cls, pairs, device="cpu", distributed=False):
        encoder = cls.encode_distributed if distributed else cls.encode
        tokens = torch.tensor([encoder(a, b) for a, b in pairs], dtype=torch.long, device=device)
        labels = torch.tensor([cls.verify(a, b) for a, b in pairs], dtype=torch.long, device=device)
        return tokens, labels


def make_split(seed=42, fraction=0.8):
    pairs = [(a, b) for a in range(10) for b in range(10)]
    random.Random(seed).shuffle(pairs)
    n = int(len(pairs) * fraction)
    return pairs[:n], pairs[n:]
