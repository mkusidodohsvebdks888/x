"""SRT-M15 chat routing: toy arithmetic or small character language model.

No external LLM, no general question answering. Existing language checkpoints
cannot represent characters absent from their training vocabulary.
"""
from dataclasses import dataclass
from pathlib import Path
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


class SRTChatEngine:
    def __init__(self, arithmetic_path="checkpoints/srt_m15_toy.pt",
                 language_path=None):
        self.arithmetic_path = Path(arithmetic_path)
        if language_path is None:
            enhanced = Path("checkpoints/srt_m15_qa.pt")
            language_path = enhanced if enhanced.is_file() else "checkpoints/srt_m15_lm.pt"
        self.language_path = Path(language_path)
        self._math = None
        self._lm = None
        self._vocab = None
        torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

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

    def reply(self, message, *, new_tokens=36):
        message = message.strip()
        if not message:
            return ChatReply("notice", "请输入问题，例如：1+9=多少，或：你好")
        route = recognize_question(message)
        if route[0] == "unsupported_expression":
            return ChatReply("limit", "当前数学模型仅支持两个 0～9 整数的单次加法；暂不支持连续运算、减法和乘除法。")
        if route[0] == "unsupported_math":
            a, b = route[1:]
            return ChatReply("limit", f"识别为 {a}+{b}，但当前数学网络仅训练了 0～9 的两个整数相加。")
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
            verdict = "预测正确" if predicted == verified else "预测错误（仍属于研究实验模型）"
            return ChatReply("math", f"题目：{a}+{b}\n神经网络预测：{predicted}\n验证器答案：{verified}\n{verdict}")
        model = self._language_model()
        if model is None:
            return ChatReply("notice", "没有语言模型权重，请先训练或恢复 SRT_M15_Results.zip。")
        unknown = sorted({ch for ch in message if ch not in self._vocab.stoi})
        notice = ""
        if unknown:
            sample = " ".join(repr(ch) for ch in unknown[:8])
            if "□" not in self._vocab.stoi:
                return ChatReply("vocab", f"旧语言模型词表中缺少：{sample}。请先运行可选问答语料训练，或者输入 1+9=多少 测试数学模型。")
            message = "".join(ch if ch in self._vocab.stoi else "□" for ch in message)
            notice = f"注意：未见字符 {sample} 已替换为 □，输入语义可能丢失。\\n"
        framed = f"问：{message}\n答："
        prompt = framed if all(ch in self._vocab.stoi for ch in "问：\n答") else message
        prompt = prompt[-model.cfg.max_length:]
        input_ids = torch.tensor([self._vocab.encode(prompt)], dtype=torch.long)
        with torch.inference_mode():
            output = model.generate(input_ids, max_new_tokens=min(max(int(new_tokens), 1), 80))
        generated = self._vocab.decode(output[0, input_ids.size(1):].tolist())
        return ChatReply("text", notice + "字符模型续写（不代表可靠回答）：\n" + generated)
