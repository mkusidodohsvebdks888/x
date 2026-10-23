"""SRT-M15 local neural chat with explicit, opt-in external LLM backends.

Local answers come from the experimental SRT-M15 weights, not fixed identity
strings. DeepSeek Web/API replies are labeled as external and never silently
treated as learned SRT-M15 knowledge.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import re
import unicodedata

import torch
from .model import ModelConfig, SRTM15
from .tasks import ArithmeticTokenizer
from .language import CharVocabulary, SRTM15LanguageModel


@dataclass
class ChatReply:
    mode: str
    text: str
    provider: str = "SRT-M15"


def recognize_question(message):
    cleaned = unicodedata.normalize("NFKC", message).strip()
    match = re.search(r"(?<![\d-])(\d+)\s*(?:\+|加)\s*(\d+)(?!\d)", cleaned)
    if match:
        rest = cleaned[match.end():]
        if re.search(r"(?:[+*/\-]|加|减|乘|除)\s*\d+", rest):
            return ("unsupported_expression",)
        a, b = int(match.group(1)), int(match.group(2))
        if a <= 9 and b <= 9:
            return ("addition", a, b)
        return ("unsupported_math", a, b)
    if re.search(r"\d+\s*(?:[-*/]|减|乘|除)\s*\d+", cleaned):
        return ("unsupported_expression",)
    return ("text",)


def _is_identity_question(message):
    """Route identity to local weights in auto mode (not a scripted answer)."""
    normalized = unicodedata.normalize("NFKC", message).casefold()
    normalized = re.sub(r"[\W_]+", "", normalized)
    return normalized in {
        "你是谁", "你是谁呀", "你是谁啊", "你到底是谁", "你好你是谁",
        "你叫什么", "你叫什么名字", "你叫什么名", "你的名字",
        "你的名字是什么", "你的身份是什么", "你的身份",
        "你是什么", "你是什么模型", "你是哪个模型", "你是什么ai",
        "请介绍一下你自己", "介绍一下你自己", "自我介绍",
        "自我介绍一下", "你是gpt吗", "你是chatgpt吗",
        "whoareyou", "whatisyourname", "introduceyourself",
    }


class SRTChatEngine:
    VALID_BACKENDS = {"local", "web", "api", "auto"}

    def __init__(self, root=None, arithmetic_path=None, language_path=None,
                 backend="local", external_backend=None):
        self.root = Path(root or ".").resolve()
        self.arithmetic_path = (Path(arithmetic_path) if arithmetic_path
                                else self.root / "checkpoints/srt_m15_toy.pt")
        if language_path is None:
            enhanced = self.root / "checkpoints/srt_m15_qa.pt"
            language_path = enhanced if enhanced.is_file() else self.root / "checkpoints/srt_m15_lm.pt"
        self.language_path = Path(language_path)
        self.backend = str(backend).strip().lower()
        self.external_backend = str(
            external_backend or os.environ.get("SRT_EXTERNAL_BACKEND", "api")
        ).strip().lower()
        if self.backend not in self.VALID_BACKENDS:
            raise ValueError("backend 必须是 local / web / api / auto")
        if self.external_backend not in {"web", "api"}:
            raise ValueError("external_backend 只能是 web 或 api")
        self._math = None
        self._lm = None
        self._vocab = None
        self._web_client = None
        torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    def status(self):
        return {
            "model": "SRT-M15",
            "backend": self.backend,
            "external_backend": self.external_backend,
            "math_checkpoint": self.arithmetic_path.is_file(),
            "language_checkpoint": self.language_path.is_file(),
            "capabilities": ["0..9 addition", "trainable char-level QA",
                             "optional DeepSeek Web (desktop)", "optional DeepSeek API"],
        }

    def _math_model(self):
        if self._math is None:
            if not self.arithmetic_path.is_file():
                return None
            data = torch.load(self.arithmetic_path, map_location="cpu", weights_only=True)
            model = SRTM15(ModelConfig(**data["config"]))
            model.load_state_dict(data["model"])
            self._math = (model.eval(), data.get("task") == "distributed")
        return self._math

    def _language_model(self):
        if self._lm is None:
            if not self.language_path.is_file():
                return None
            data = torch.load(self.language_path, map_location="cpu", weights_only=True)
            vocab = CharVocabulary(data["vocab"])
            model = SRTM15LanguageModel(ModelConfig(**data["config"]))
            model.load_state_dict(data["model"])
            self._lm = model.eval()
            self._vocab = vocab
        return self._lm

    def _external_reply(self, question, backend):
        if backend == "web":
            from .deepseek_web import DeepSeekWebClient, DeepSeekWebError
            if self._web_client is None:
                self._web_client = DeepSeekWebClient()
            try:
                answer = self._web_client.ask(question)
                return ChatReply("external", answer, "DeepSeek Web")
            except DeepSeekWebError as exc:
                return ChatReply("error", f"网页版 DeepSeek 未成功回答：{exc}", "DeepSeek Web")
        if backend == "api":
            from .deepseek_api import DeepSeekAPIClient, DeepSeekAPIError
            try:
                answer = DeepSeekAPIClient().ask(question)
                return ChatReply("external", answer, "DeepSeek API")
            except DeepSeekAPIError as exc:
                return ChatReply("error", f"DeepSeek API 未成功回答：{exc}", "DeepSeek API")
        raise ValueError("未知外部模型方式")

    def close(self):
        if self._web_client is not None:
            self._web_client.close()
            self._web_client = None

    def reply(self, message, *, new_tokens=80, backend=None):
        message = str(message).strip()
        if len(message) > 400:
            return ChatReply("limit", "输入过长，请将问题缩短到400个字符以内。")
        if not message:
            return ChatReply("notice", "请输入问题，例如：你是谁，或者：1+9=多少。")

        # /raw explicitly tests SRT-M15 weights even if web/api/auto is selected.
        raw_mode = message.casefold().startswith("/raw ")
        if raw_mode:
            message = message[5:].strip()
            if not message:
                return ChatReply("notice", "请在 /raw 后输入问题，例如 /raw 你是谁。")

        selected = str(backend or self.backend).strip().lower()
        if selected not in self.VALID_BACKENDS:
            raise ValueError("backend 必须是 local / web / api / auto")
        if raw_mode:
            selected = "local"

        route = recognize_question(message)
        if selected == "auto":
            if route[0] == "addition" or _is_identity_question(message):
                selected = "local"
            else:
                selected = self.external_backend
        if selected in {"web", "api"}:
            return self._external_reply(message, selected)

        if route[0] == "unsupported_expression":
            return ChatReply("limit", "当前本地数学模型仅支持两个 0～9 整数的单次加法；可切换网页版或 API。")
        if route[0] == "unsupported_math":
            a, b = route[1:]
            return ChatReply("limit", f"识别为 {a}+{b}，但本地数学网络仅训练了 0～9 的两个整数相加。")
        if route[0] == "addition":
            loaded = self._math_model()
            if loaded is None:
                return ChatReply("notice", "没有数学模型权重，请先训练或恢复 SRT_M15_Results.zip。")
            model, distributed = loaded
            a, b = route[1:]
            tokens, _ = ArithmeticTokenizer.batch([(a, b)], distributed=distributed)
            with torch.inference_mode():
                result = model(tokens, hard_routes=True)
            logits = result["receiver_logits"] if distributed else result["logits"]
            predicted = int(logits.argmax(-1).item())
            verified = ArithmeticTokenizer.verify(a, b)
            verdict = "预测正确" if predicted == verified else "预测错误（研究实验模型）"
            return ChatReply("math", f"题目：{a}+{b}\n神经网络预测：{predicted}\n验证器答案：{verified}\n{verdict}")

        model = self._language_model()
        if model is None:
            return ChatReply(
                "notice",
                "没有本地语言模型权重。请在 Notebook 第 9 步运行 train_qa.py，"
                "或恢复之前训练好的模型。"
            )

        unknown = sorted({ch for ch in message if ch not in self._vocab.stoi})
        notice = ""
        if unknown:
            sample = " ".join(repr(ch) for ch in unknown[:8])
            if "□" not in self._vocab.stoi:
                return ChatReply("vocab", f"模型词表缺少：{sample}。请训练第 9 步问答模型。")
            message = "".join(ch if ch in self._vocab.stoi else "□" for ch in message)
            notice = f"未知字符 {sample} 被替换为 □，可能影响答案。\n"

        framed = f"问：{message}\n答："
        prompt = framed if all(ch in self._vocab.stoi for ch in "问：\n答") else message
        prompt = prompt[-model.cfg.max_length:]
        input_ids = torch.tensor([self._vocab.encode(prompt)], dtype=torch.long)

        # Same greedy strategy as train_qa.py; no hard-coded identity string.
        output_chars = []
        with torch.inference_mode():
            for _ in range(max(1, min(int(new_tokens), 80))):
                logits = model(input_ids[:, -model.cfg.max_length:])["logits"][:, -1]
                next_id = int(logits.argmax(-1).item())
                character = self._vocab.chars[next_id]
                if character == "\n":
                    break
                output_chars.append(character)
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_id]], dtype=torch.long)], dim=1
                )
        answer = "".join(output_chars)
        if not answer:
            answer = "（神经网络没有生成有效回答，请继续训练。）"
        return ChatReply("text", notice + answer)
