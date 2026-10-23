"""Small, offline chat dispatcher for two separately trained SRT-M15 toy models.

The arithmetic network predicts single-digit addition; a deterministic verifier
is displayed separately. The character LM continues text and is NOT a general
question-answering model. Offline/local is the default. Optional external LLM
backends are explicit opt-ins and must never be mislabeled as local learning.
"""
import re
import os
import unicodedata
from pathlib import Path
from dataclasses import dataclass

import torch

from .model import ModelConfig, SRTM15
from .tasks import ArithmeticTokenizer
from .language import CharVocabulary, SRTM15LanguageModel


@dataclass
class ChatReply:
    mode: str
    text: str
    provider: str = "SRT-M15"
    observed_pair: tuple[int, int] | None = None
    prediction: int | None = None
    expected: int | None = None


def route_question(message):
    text = unicodedata.normalize("NFKC", message).strip()
    found = re.search(r"(?<!\d)(\d+)\s*(?:\+|\u52a0)\s*(\d+)(?!\d)", text)
    if found:
        before, after = text[:found.start()], text[found.end():]
        if (re.search(r"\d", before + after) or
                re.search(r"(?:\+|\-|\*|/|\u51cf|\u4e58|\u9664)\s*\d+", after)):
            return ("unsupported",)
        a, b = map(int, found.groups())
        return ("addition", a, b) if a <= 9 and b <= 9 else ("math_limit", a, b)
    if re.search(r"\d\s*(?:\-|\*|/|\u51cf|\u4e58|\u9664)\s*\d", text):
        return ("unsupported",)
    return ("text",)


def detect_text_language(message, requested='auto'):
    """A small deterministic routing hint, not language understanding."""
    if requested not in ('auto', 'zh', 'en'):
        raise ValueError('对话语言须为 auto、zh 或 en')
    if requested != 'auto':
        return requested
    # Mixed Chinese+English uses the Chinese dialogue prefix by default.
    if re.search(r'[\u3400-\u9fff]', message):
        return 'zh'
    return 'en' if re.search(r'[A-Za-z]', message) else 'zh'


def _is_identity_question(message):
    normalized = unicodedata.normalize("NFKC", message).casefold()
    normalized = re.sub(r"[\W_]+", "", normalized)
    return normalized in {
        "\u4f60\u662f\u8c01", "\u4f60\u662f\u8c01\u5440", "\u4f60\u662f\u8c01\u554a",
        "\u4f60\u53eb\u4ec0\u4e48", "\u4f60\u53eb\u4ec0\u4e48\u540d\u5b57",
        "\u4f60\u7684\u540d\u5b57", "\u4f60\u662f\u4ec0\u4e48\u6a21\u578b",
        "whoareyou", "whatisyourname", "introduceyourself",
    }


class SRTChatEngine:
    """Loads local weights once; external backends are explicit opt-in only."""

    VALID_BACKENDS = {"local", "web", "api", "auto"}

    def __init__(self, root=None, backend="local", external_backend=None):
        self.root = Path(root or ".").resolve()
        self.math_model = None
        self.math_distributed = False
        self.math_checkpoint_path = None
        self.language_model = None
        self.vocab = None
        self.language_type = "character"
        self.backend = str(backend).strip().lower()
        self.external_backend = str(
            external_backend or os.environ.get("SRT_EXTERNAL_BACKEND", "api")
        ).strip().lower()
        if self.backend not in self.VALID_BACKENDS:
            raise ValueError("backend must be local, web, api or auto")
        if self.external_backend not in {"api", "web"}:
            raise ValueError("external_backend must be api or web")
        self._web_client = None
        torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    def _load_math(self):
        if self.math_model is not None:
            return True
        adapted = self.root / "checkpoints/srt_m15_toy_adapted.pt"
        original = self.root / "checkpoints/srt_m15_toy.pt"
        checkpoint = adapted if adapted.is_file() else original
        if not checkpoint.is_file():
            return False
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = SRTM15(ModelConfig(**state["config"]))
        model.load_state_dict(state["model"])
        self.math_model = model.eval()
        self.math_distributed = state.get("task") == "distributed"
        self.math_checkpoint_path = checkpoint
        return True

    def _load_language(self):
        if self.language_model is not None:
            return True
        candidates = [self.root / "checkpoints/srt_m15_byte_qa.pt",
                      self.root / "checkpoints/srt_m15_qa.pt",
                      self.root / "checkpoints/srt_m15_lm.pt"]
        checkpoint = next((p for p in candidates if p.is_file()), None)
        if checkpoint is None:
            return False
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("tokenizer") == "utf8-bytes-v1":
            from .supervised_language import Utf8ByteVocabulary
            self.vocab = Utf8ByteVocabulary()
            self.language_type = "utf8-bytes-v1"
        else:
            self.vocab = CharVocabulary(state["vocab"])
            self.language_type = "character"
        model = SRTM15LanguageModel(ModelConfig(**state["config"]))
        model.load_state_dict(state["model"])
        self.language_model = model.eval()
        return True

    def clear_language_cache(self):
        self.language_model = None
        self.vocab = None
        self.language_type = "character"

    def status(self):
        return {
            "model": "SRT-M15",
            "backend": self.backend,
            "external_backend": self.external_backend,
            "math_checkpoint": (self.root / "checkpoints/srt_m15_toy.pt").is_file(),
            "language_checkpoint": any((self.root / "checkpoints" / p).is_file()
                                       for p in ("srt_m15_byte_qa.pt", "srt_m15_qa.pt", "srt_m15_lm.pt")),
            "adapted_checkpoint": (self.root / "checkpoints/srt_m15_toy_adapted.pt").is_file(),
            "byte_qa_checkpoint": (self.root / "checkpoints/srt_m15_byte_qa.pt").is_file(),
            "language_type": self.language_type,
            "capabilities": ["0..9 addition", "experimental Chinese/English UTF-8-byte continuation",
                             "optional DeepSeek Web on desktop", "optional DeepSeek API"],
        }

    def _external_reply(self, message, backend):
        if backend == "web":
            from .deepseek_web import DeepSeekWebClient, DeepSeekWebError
            if self._web_client is None:
                self._web_client = DeepSeekWebClient()
            try:
                return ChatReply("external", self._web_client.ask(message), provider="DeepSeek Web")
            except DeepSeekWebError as exc:
                return ChatReply("error", str(exc), provider="DeepSeek Web")
        if backend == "api":
            from .deepseek_api import DeepSeekAPIClient, DeepSeekAPIError
            try:
                return ChatReply("external", DeepSeekAPIClient().ask(message), provider="DeepSeek API")
            except DeepSeekAPIError as exc:
                return ChatReply("error", str(exc), provider="DeepSeek API")
        raise ValueError("unknown external backend")

    def close(self):
        if self._web_client is not None:
            self._web_client.close()
            self._web_client = None

    def reply(self, message, *, new_tokens=26, language='auto', backend=None):
        message = str(message).strip()
        if not message:
            return ChatReply("notice", "\u8bf7\u8f93\u5165\u95ee\u9898\uff0c\u4f8b\u5982 1+9=\u591a\u5c11 \u6216 \u4f60\u597d\u3002")
        if len(message) > 400:
            return ChatReply("limit", "\u8f93\u5165\u8fc7\u957f\uff0c\u8bf7\u7f29\u77ed\u5230 400 \u5b57\u7b26\u4ee5\u5185\u3002")
        raw_mode = message.casefold().startswith("/raw ")
        if raw_mode:
            message = message[5:].strip()
            if not message:
                return ChatReply("notice", "Please enter a question after /raw.")
        selected = str(backend or self.backend).strip().lower()
        if selected not in self.VALID_BACKENDS:
            raise ValueError("backend must be local, web, api or auto")
        route = route_question(message)
        if raw_mode:
            selected = "local"
        elif selected == "auto":
            selected = "local" if route[0] == "addition" or _is_identity_question(message) else self.external_backend
        if selected in {"web", "api"}:
            return self._external_reply(message, selected)
        text_language = detect_text_language(message, language)
        if route[0] == "unsupported":
            return ChatReply("limit", "\u76ee\u524d\u4ec5\u652f\u6301\u4e24\u4e2a 0\uff5e9 \u6574\u6570\u76f8\u52a0\uff0c\u4e0d\u652f\u6301\u8fde\u7eed\u8fd0\u7b97\u3002")
        if route[0] == "math_limit":
            return ChatReply("limit", "\u5df2\u8bc6\u522b\u52a0\u6cd5\uff0c\u4f46\u8bad\u7ec3\u6570\u5b66\u6a21\u578b\u7684\u64cd\u4f5c\u6570\u53ea\u6709 0\uff5e9\u3002")
        if route[0] == "addition":
            if not self._load_math():
                return ChatReply("notice", "\u672a\u627e\u5230\u6570\u5b66\u6743\u91cd\u6587\u4ef6\u3002")
            a, b = route[1:]
            x, _ = ArithmeticTokenizer.batch([(a, b)], distributed=self.math_distributed)
            with torch.inference_mode():
                out = self.math_model(x, hard_routes=True)
            logits = out["receiver_logits"] if self.math_distributed else out["logits"]
            predicted = int(logits.argmax(dim=-1).item())
            verified = ArithmeticTokenizer.verify(a, b)
            verdict = "\u9884\u6d4b\u6b63\u786e" if predicted == verified else "\u9884\u6d4b\u9519\u8bef\uff1a\u8bf7\u4ee5\u9a8c\u8bc1\u7b54\u6848\u4e3a\u51c6"
            return ChatReply("math", f"{a} + {b}\n\u795e\u7ecf\u7f51\u7edc\u9884\u6d4b\uff1a{predicted}\n\u9a8c\u8bc1\u5668\u7b54\u6848\uff1a{verified}\n{verdict}",
                             observed_pair=(a, b), prediction=predicted, expected=verified)
        if not self._load_language():
            return ChatReply("notice", "\u672a\u627e\u5230\u8bed\u8a00\u6a21\u578b\u6743\u91cd\u6587\u4ef6\u3002")
        if self.language_type == "utf8-bytes-v1":
            from .supervised_language import prompt_prefix
            prefix = prompt_prefix(message, text_language)
            ids = self.vocab.encode(prefix)[-self.language_model.cfg.max_length:]
            inputs = torch.tensor([ids], dtype=torch.long)
            with torch.inference_mode():
                generated = self.language_model.generate(inputs, max_new_tokens=min(max(new_tokens * 3, 1), 72),
                                                         temperature=0.8)
            output = self.vocab.decode(generated[0, inputs.shape[1]:].tolist())
            output = output.split("\n问：")[0].split("\nQ:")[0].split("\nA:")[0]
            return ChatReply("text", f"{text_language.upper()} 字节模型续写（不代表可靠回答）：\n" + output)
        missing = sorted({char for char in message if char not in self.vocab.stoi})
        warning = ""
        if missing:
            if "\u25a1" not in self.vocab.stoi:
                chars = " ".join(repr(c) for c in missing[:8])
                return ChatReply("vocab", "\u65e7\u6a21\u578b\u7684\u5b57\u7b26\u8bcd\u8868\u4e0d\u5305\u542b\uff1a" + chars + "\u3002\u8bf7\u6269\u5c55\u8bed\u6599\u5e76\u91cd\u65b0\u8bad\u7ec3\u3002")
            warning = "\u672a\u77e5\u5b57\u7b26\u5df2\u66ff\u6362\u4e3a\u25a1\uff0c\u8bed\u4e49\u53ef\u80fd\u53d7\u635f\u3002\n"
            message = "".join(c if c in self.vocab.stoi else "\u25a1" for c in message)
        question_prefix = "\u95ee\uff1a"
        answer_prefix = "\n\u7b54\uff1a"
        framed = question_prefix + message + answer_prefix
        if all(c in self.vocab.stoi for c in framed):
            prompt = framed
        else:
            prompt = message
        prompt = prompt[-self.language_model.cfg.max_length:]
        ids = torch.tensor([self.vocab.encode(prompt)], dtype=torch.long)
        with torch.inference_mode():
            generated = self.language_model.generate(ids, max_new_tokens=min(max(new_tokens, 1), 60))
        continuation = self.vocab.decode(generated[0, ids.shape[1]:].tolist())
        return ChatReply("text", warning + "\u5b57\u7b26\u6a21\u578b\u7eed\u5199\uff08\u4e0d\u4fdd\u8bc1\u95ee\u7b54\u6b63\u786e\uff09\uff1a\n" + continuation)
