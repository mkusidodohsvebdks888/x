"""Incremental, bounded token decoding for SRT-M15 API.

The 'staged' checkpoint is the original small SRT network. Its character
generation is experimental and NOT a fluent general-purpose conversation model.
This module never sends user conversation history to another provider.
"""
import codecs
from dataclasses import dataclass
from pathlib import Path

import torch

from .language import CharVocabulary, SRTM15LanguageModel
from .model import ModelConfig


@dataclass
class LocalDecoder:
    model: object
    vocab: object
    tokenizer_type: str
    qa_training: str = ""
    provider: str = "SRT-M15 local"


def validate_messages(messages, *, max_total_chars=600):
    """Strict OpenAI-style stateless conversation: [system]? user,assistant,...,user.

    All turns are actually included, not silently dropped or treated as model
    memory. Context must fit the user-supplied local checkpoint.
    """
    if not messages or messages[-1].role != "user":
        raise ValueError("The final message must have role=user")
    systems = [i for i, msg in enumerate(messages) if msg.role == "system"]
    if systems not in ([], [0]):
        raise ValueError("Only one leading system message is accepted")
    conversation = messages[1:] if systems else messages
    for i, msg in enumerate(conversation):
        expected = "user" if i % 2 == 0 else "assistant"
        if msg.role != expected:
            raise ValueError("Messages must alternate user, assistant and end in user")
        if "\x00" in msg.content:
            raise ValueError("NUL in messages is not allowed")
    if sum(len(msg.content) for msg in messages) > max_total_chars:
        raise ValueError("Conversation exceeds 600 character budget; shorten history")
    return len(conversation) > 1 or bool(systems)


def as_prompt(messages, *, language="auto"):
    """Pack history into *text*, not model-native role embeddings.

    Local toy checkpoints have NOT been trained to obey system/assistant
    structured roles reliably; this is useful context formatting only.
    """
    has_context = validate_messages(messages)
    if not has_context:
        return messages[-1].content, False
    if language == "auto":
        from .chat import detect_text_language
        language = detect_text_language(messages[-1].content)
    if language not in ("zh", "en"):
        raise ValueError("language must be auto, zh or en")
    # Reuse the training corpus' question/answer delimiters as much as
    # possible. The model still has no trained instruction-role awareness.
    q, a = ("问：", "答：") if language == "zh" else ("Q: ", "A: ")
    turns = []
    for msg in messages:
        if msg.role == "system":
            turns.append(("说明：" if language == "zh" else "Instruction: ") + msg.content)
        else:
            turns.append((q if msg.role == "user" else a) + msg.content)
    turns.append(a)
    return "\n".join(turns), True


def from_chat_engine(engine):
    """Wrap the existing checkpoint; no destructive replacement of SRTChatEngine."""
    if not engine._load_language():
        raise FileNotFoundError("SRT-M15 local language checkpoint is not installed")
    return LocalDecoder(engine.language_model, engine.vocab, engine.language_type,
                        engine.qa_training, "SRT-M15 local")


def from_staged_checkpoint(checkpoint_path):
    """Load a checkpoint from train_bilingual_stages.py (trusted local file only)."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Staged SRT-M15 checkpoint not present")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "srt-m15-bilingual-staged-v1":
        raise ValueError("Unsupported staged SRT-M15 checkpoint format")
    config = ModelConfig(**state["config"])
    vocab = CharVocabulary(state["vocab"])
    if config.vocab_size != len(vocab.chars):
        raise ValueError("Staged checkpoint vocabulary size mismatch")
    model = SRTM15LanguageModel(config)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return LocalDecoder(model, vocab, "character", "", "SRT-M15 staged (local)")


def prepare_local(decoder, prompt, *, language="auto", has_context=False):
    """Validate tokenized input before HTTP 200 or the first SSE frame."""
    from .chat import detect_text_language

    model = decoder.model
    if decoder.tokenizer_type == "utf8-bytes-v1":
        from .supervised_language import prompt_prefix
        prefix = prompt if has_context else prompt_prefix(
            prompt, detect_text_language(prompt, language))
        # The byte checkpoint was trained with a short max context; unlike
        # older experimental CLI, API must never silently truncate history.
        tokens = decoder.vocab.encode(prefix)
        if len(tokens) >= model.cfg.max_length:
            raise ValueError("Conversation exceeds local model byte context window")
        return torch.tensor([tokens], dtype=torch.long), "bytes"

    if decoder.tokenizer_type != "character":
        raise ValueError("Unsupported tokenizer")
    if decoder.qa_training.startswith("masked_answer"):
        prefix = "问：" + prompt + "\n答："
    elif has_context:
        prefix = prompt
    else:
        prefix = "问：" + prompt + "\n答："
    if any(c not in decoder.vocab.stoi for c in prefix):
        # A user should receive an explicit client error rather than hallucination
        # from silently replacing unknown characters.
        raise ValueError("The trained character vocabulary cannot encode this input")
    tokens = decoder.vocab.encode(prefix)
    if len(tokens) >= model.cfg.max_length:
        raise ValueError("Conversation exceeds local model character context window")
    return torch.tensor([tokens], dtype=torch.long), "characters"


def iter_local_tokens(decoder, prompt, *, max_tokens=80, language="auto",
                      has_context=False):
    """Yield *newly inferred* UTF-8 text while sampling one token at a time."""
    if not 1 <= max_tokens <= 80:
        raise ValueError("max_tokens must be between 1 and 80")
    ids, mode = prepare_local(decoder, prompt, language=language, has_context=has_context)
    decoder.model.eval()
    incremental = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text_so_far = ""
    with torch.inference_mode():
        for _ in range(max_tokens):
            logits = decoder.model(ids[:, -decoder.model.cfg.max_length:])["logits"][:, -1]
            if decoder.qa_training.startswith("masked_answer"):
                next_id = int(logits.argmax(dim=-1).item())
            else:
                probabilities = torch.softmax(logits / 0.8, dim=-1)
                next_id = int(torch.multinomial(probabilities, num_samples=1).item())
            if mode == "bytes":
                piece = incremental.decode(bytes([next_id]), final=False)
            else:
                piece = decoder.vocab.chars[next_id]
            ids = torch.cat((ids, torch.tensor([[next_id]], dtype=torch.long)), dim=1)
            # Stop on a completed line / next QA marker, without exposing
            # a fabricated next dialogue turn as assistant output.
            if "\n" in piece:
                head = piece.split("\n", 1)[0]
                if head:
                    yield head
                break
            if piece:
                text_so_far += piece
                if len(text_so_far) > 500:
                    break
                yield piece
        if mode == "bytes":
            tail = incremental.decode(b"", final=True)
            if tail and "\n" not in tail:
                yield tail
