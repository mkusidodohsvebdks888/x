"""Offline bilingual multi-turn dialogue examples for SRT-M15 experiments.

Synthetic/unreviewed training material. Not a claim of fluent conversation.
Run: python generate_local_dialogues.py
Then train with train_bilingual_stages.py --allow-demo-data.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from bilingual_data import load_records, normalize_prompt, split_records

ROOT = Path(__file__).resolve().parent
NAMES = [("小林", "Lin"), ("小陈", "Chen"), ("小雨", "Yu"), ("小安", "An"),
         ("小方", "Fang"), ("小宁", "Ning"), ("小夏", "Xia"), ("小吴", "Wu")]
PLACES = [("图书馆", "library"), ("公园", "park"), ("教室", "classroom"),
          ("办公室", "office"), ("咖啡馆", "cafe"), ("书店", "bookstore")]
OBJECTS = [("笔记本", "notebook"), ("雨伞", "umbrella"), ("背包", "backpack"),
           ("水杯", "cup"), ("地图", "map"), ("耳机", "headphones"),
           ("外套", "coat"), ("书", "book")]


def pairs(i):
    name, en_name = NAMES[i % len(NAMES)]
    place, en_place = PLACES[(i // len(NAMES)) % len(PLACES)]
    obj, en_obj = OBJECTS[(i // (len(NAMES) * len(PLACES))) % len(OBJECTS)]
    # A situation stays intact across training/validation. Varying prompt
    # values is not equivalent to human-authored linguistic diversity.
    return [
        ("remember_name",
         f"用户：你好，我叫{name}。\n助手：你好！\n用户：我叫什么名字？",
         f"你刚才说你叫{name}。",
         f"User: Hello, my name is {en_name}.\nAssistant: Hello!\nUser: What is my name?",
         f"You said your name was {en_name}."),
        ("remember_place",
         f"用户：我准备去{place}。\n助手：知道了。\n用户：刚才我要去哪里？",
         f"你准备去{place}。",
         f"User: I plan to visit the {en_place}.\nAssistant: Got it.\nUser: Where was I going?",
         f"You planned to visit the {en_place}."),
        ("remember_object",
         f"用户：{name}把{obj}放在{place}。\n助手：收到。\n用户：谁把什么放在哪里？",
         f"{name}把{obj}放在{place}。",
         f"User: {en_name} put the {en_obj} in the {en_place}.\nAssistant: Noted.\nUser: Who put what where?",
         f"{en_name} put the {en_obj} in the {en_place}."),
        ("correction",
         f"用户：{name}要去{place}。\n助手：好的。\n用户：更正，{name}不去了。\n助手：明白。\n用户：现在还应该说{name}要去{place}吗？",
         f"不应该。你后来更正说{name}不去了。",
         f"User: {en_name} will visit the {en_place}.\nAssistant: Okay.\nUser: Correction: {en_name} will not go.\nAssistant: Understood.\nUser: Should I still say that {en_name} will visit the {en_place}?",
         f"No. You corrected that and said {en_name} would not go."),
        ("uncertainty",
         f"用户：我有一个{obj}。\n助手：好的。\n用户：你知道我的{obj}是什么颜色吗？",
         f"不知道，你没有告诉我{obj}的颜色。",
         f"User: I have a {en_obj}.\nAssistant: Okay.\nUser: Do you know what color my {en_obj} is?",
         "I don't know; you haven't told me its color."),
        ("clarification",
         f"用户：请帮{name}把{obj}送到一个地方。\n助手：好的。\n用户：你现在知道具体目的地吗？",
         "还不知道。请告诉我具体目的地。",
         f"User: Please help {en_name} deliver a {en_obj} somewhere.\nAssistant: Okay.\nUser: Do you know the exact destination yet?",
         "Not yet. Please tell me the exact destination."),
        ("brief_summary",
         f"用户：今天{name}带着{obj}去了{place}。\n助手：知道了。\n用户：请简短总结。",
         f"{name}带着{obj}去了{place}。",
         f"User: Today {en_name} took a {en_obj} to the {en_place}.\nAssistant: Noted.\nUser: Summarize briefly.",
         f"{en_name} took a {en_obj} to the {en_place}."),
        ("identity",
         f"用户：{name}提到了{obj}。\n助手：收到。\n用户：你是谁？",
         "我是 SRT-M15，一个正在开发中的实验性 AI 助手。",
         f"User: {en_name} mentioned a {en_obj}.\nAssistant: Noted.\nUser: Who are you?",
         "I'm SRT-M15, an experimental AI assistant under development."),
        ("unsupported_claim",
         f"用户：{name}的{obj}不见了。\n助手：我可以帮忙整理寻找步骤。\n用户：你能确定{obj}就在{place}吗？",
         f"不能，我没有证据确定{obj}在{place}。",
         f"User: {en_name} lost a {en_obj}.\nAssistant: I can help plan the search.\nUser: Are you certain the {en_obj} is at the {en_place}?",
         f"No. I have no evidence that the {en_obj} is at the {en_place}."),
        ("polite_request",
         f"用户：{name}想借{obj}，请写一句礼貌的请求。",
         f"你好，请问可以借用一下{obj}吗？谢谢！",
         f"User: {en_name} wants to borrow a {en_obj}. Write a polite request.",
         f"Hello, may I borrow the {en_obj}, please? Thank you!"),
    ]


def generate():
    rows, seen = [], set()
    counts = Counter()
    # 384 distinct name/place/object situations, 10 patterns each.
    for i in range(len(NAMES) * len(PLACES) * len(OBJECTS)):
        for j, (category, zq, za, eq, ea) in enumerate(pairs(i)):
            keys = [normalize_prompt(zq), normalize_prompt(eq)]
            if len(set(keys)) != 2 or any(k in seen or not k for k in keys):
                continue
            seen.update(keys)
            topic_id = f"dialogue-{i:04d}-{j:02d}"
            split = "dev" if i % 5 == 0 else "train"
            for lang, prompt, response in (("zh", zq, za), ("en", eq, ea)):
                rows.append({
                    "language": lang, "prompt": prompt, "response": response,
                    "topic_id": topic_id, "family_id": f"dialogue-situation-{i:04d}",
                    "category": category, "split": split,
                    "verified": False, "review_status": "unreviewed",
                    "synthetic": True, "source": "srt-m15-local-dialogue-v1"
                })
            counts[category] += 2
    if len(rows) < 2000:
        raise AssertionError("insufficient unique dialogue examples")
    return rows, dict(counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "examples/local_dialogue_synthetic_v1.jsonl")
    args = ap.parse_args()
    rows, categories = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    args.output.write_text(payload, encoding="utf-8")
    train, dev = split_records(load_records(args.output, allow_demo=True))
    assert not ({r.topic_id for r in train} & {r.topic_id for r in dev})
    print(json.dumps({
        "rows": len(rows), "train": len(train), "dev": len(dev),
        "categories": categories,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "warning": "Synthetic data only; does not indicate training or fluency."
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
