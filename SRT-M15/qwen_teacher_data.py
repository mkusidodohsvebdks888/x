"""Locally distill short Qwen dialogue answers into SRT-M15 experimental data.

Uses an already downloaded Ollama Qwen model on 127.0.0.1; no hosted API.
Outputs are SYNTHETIC / UNREVIEWED and are not automatically trained.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from bilingual_data import load_records, split_records, normalize_prompt
from srt_m15.ollama_local import OllamaLocalClient, OllamaLocalError

TOPICS = [
    ("日常问候", "everyday greetings"), ("自我介绍", "introducing yourself"),
    ("时间管理", "time management"), ("英语学习", "learning English"),
    ("阅读习惯", "reading habits"), ("团队合作", "teamwork"),
    ("Python 编程", "Python programming"), ("人工智能", "AI basics"),
    ("日常购物", "shopping"), ("旅行准备", "travel preparation"),
    ("礼貌沟通", "polite communication"), ("澄清问题", "clarifying a question"),
    ("信息总结", "summarizing information"), ("纠正错误", "correcting a mistake"),
    ("承认不确定性", "acknowledging uncertainty"), ("学习计划", "study planning"),
    ("解决问题", "problem-solving"), ("写作", "writing"),
    ("表达观点", "expressing an opinion"), ("记住上下文", "remembering context"),
]
TASKS = [
    ("请用两句话解释{topic}，就像正常聊天一样。",
     "Explain {topic} in two conversational sentences."),
    ("请给我一个关于{topic}的简单例子。",
     "Give me a simple example of {topic}."),
    ("我正在学习{topic}，先问我一个问题以便了解我的需求。",
     "I am learning {topic}. Ask one question to clarify my needs."),
    ("我不太理解{topic}，请换一种简单说法。",
     "I don't understand {topic}. Explain it more simply."),
    ("学习{topic}，你建议我先做哪两件事？",
     "What two steps should I take first to learn {topic}?"),
    ("关于{topic}，请说一个常见误区和纠正方法。",
     "Give a common misconception about {topic} and correct it."),
    ("如果你不确定我说的{topic}是什么意思，你会怎么追问？",
     "If you were unsure what I meant by {topic}, how would you ask?"),
    ("我之前说过想学{topic}，请简短概括我刚才的目标。",
     "Earlier I said I wanted to learn {topic}. Briefly summarize my goal."),
    ("我原本不想学习{topic}，现在改变主意了，请按最新决定回答。",
     "I previously didn't want to study {topic}, but changed my mind. Respond to my latest decision."),
    ("请给正在学习{topic}的人一句友好、具体的鼓励。",
     "Give a friendly and specific encouragement to someone studying {topic}."),
]
PRIVATE = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b|(?<!\d)1[3-9]\d{9}(?!\d)")
DEFAULT = Path("local_data/qwen_teacher/qwen_sft_unreviewed.jsonl")


def planned(index):
    topic_i = index % len(TOPICS)
    template_i = (index // len(TOPICS)) % len(TASKS)
    cycle = index // (len(TOPICS) * len(TASKS))
    zh, en = TASKS[template_i]
    topic_zh, topic_en = TOPICS[topic_i]
    zh = zh.format(topic=topic_zh)
    en = en.format(topic=topic_en)
    if cycle:
        zh += f" 请从第{cycle+2}种新角度回答。"
        en += f" Give another perspective, variation {cycle+2}."
    return topic_i, zh, en


def teacher_messages(language, prompt):
    intro = (
        "为实验性助手 SRT-M15 编写自然、准确的中文对话训练示例。"
        "仅回答用户问题，用两到三句话。信息不足时请澄清或说明不确定。"
        "不要称自己为 Qwen，也不要声称记住了用户未提供的信息。"
        if language == "zh" else
        "Write a clear, natural English answer of two or three short sentences "
        "as a teaching example for an experimental assistant named SRT-M15. "
        "Don't claim to be Qwen; don't fabricate missing earlier chat context."
    )
    return [{"role": "system", "content": intro}, {"role": "user", "content": prompt}]


def reject_reason(question, answer, limit):
    if not isinstance(answer, str):
        return "invalid-length"
    if normalize_prompt(question) == normalize_prompt(answer):
        return "echo"
    if not 5 <= len(answer.strip()) <= limit:
        return "invalid-length"
    if PRIVATE.search(question + answer):
        return "possible-private-data"
    if ("我是qwen" in answer.casefold().replace(" ", "")
            or "i am qwen" in answer.casefold()):
        return "wrong-model-identity"
    return ""


def make(client, pairs=20, output=DEFAULT, max_tokens=220, max_chars=1000, log=print):
    if not 5 <= pairs <= 200 or not 32 <= max_tokens <= 512 or not 40 <= max_chars <= 2048:
        raise ValueError("Limits: pairs=5..200, max_tokens=32..512, max_chars=40..2048")
    output = Path(output)
    manifest = output.with_suffix(".manifest.json")
    if output.exists() or manifest.exists():
        raise ValueError("Output already exists; choose a fresh --output path")
    rows, rejects = [], Counter()
    for index in range(pairs):
        topic, zh, en = planned(index)
        answers = []
        for language, prompt in (("zh", zh), ("en", en)):
            try:
                answer = client.chat(teacher_messages(language, prompt), max_tokens=max_tokens)
            except OllamaLocalError as exc:
                raise RuntimeError(f"Ollama unavailable at topic {index}: {exc}") from exc
            issue = reject_reason(prompt, answer, max_chars)
            if issue:
                log(f"Skipped topic {index}: {issue}")
                rejects[issue] += 1
                break
            answers.append((language, prompt, answer.strip()))
        if len(answers) != 2:
            continue
        topic_id = f"qwen-pair-{index:06d}"
        # All examples about this subject stay in the same split.
        split = "dev" if topic % 5 == 0 else "train"
        for language, question, answer in answers:
            rows.append({
                "id": topic_id + "-" + language,
                "language": language, "prompt": question, "response": answer,
                "topic_id": topic_id, "family_id": f"topic-{topic:02d}",
                "split": split, "verified": False, "synthetic": True,
                "training_approved": False, "review_status": "unreviewed",
                "source": "local-ollama:" + client.model,
                "teacher_model": client.model,
            })
        log(f"{len(rows)//2}/{pairs} bilingual pairs prepared")
    if len(rows) < 8 or {r["split"] for r in rows} != {"train", "dev"}:
        raise ValueError("Too few accepted train and dev examples")
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        data = load_records(temporary, allow_demo=True)
        train, dev = split_records(data)
        if {row.topic_id for row in train} & {row.topic_id for row in dev}:
            raise ValueError("Paired topics leaked into dev")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    summary = {
        "teacher_model": client.model,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_pairs": pairs, "accepted_pairs": len(rows)//2,
        "records": len(rows), "train_records": len(train), "dev_records": len(dev),
        "languages": dict(Counter(row["language"] for row in rows)),
        "skipped": dict(rejects),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "human_reviewed": False, "model_trained": False,
        "warning": "Qwen outputs are unverified teaching drafts, not original Qwen pretraining data.",
    }
    # Atomic per-file writes; refuse overwriting existing data.
    for path, payload in ((output, text), (manifest, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")):
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".qwen-out-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            if path.exists():
                raise ValueError("Concurrent output exists: " + str(path))
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--model", default="qwen3:4b-instruct")
    p.add_argument("--output", type=Path, default=DEFAULT)
    p.add_argument("--max-tokens", type=int, default=220)
    p.add_argument("--max-answer-chars", type=int, default=1000)
    args = p.parse_args()
    if not 5 <= args.pairs <= 200:
        p.error("pairs must be 5..200")
    if args.dry_run:
        for i in range(args.pairs):
            _, zh, en = planned(i)
            print(f"{i + 1}: ZH {zh}\n   EN {en}")
        print("Preview only: no Ollama calls or training.")
        return
    print("Collecting Qwen teacher demonstrations locally, not training model weights...")
    try:
        result = make(client=OllamaLocalClient(model=args.model),
                      pairs=args.pairs, output=args.output,
                      max_tokens=args.max_tokens, max_chars=args.max_answer_chars)
    except (ValueError, OSError, RuntimeError) as exc:
        p.exit(2, f"Teacher generation failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
