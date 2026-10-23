"""Generate exactly 110,000 bilingual synthetic training examples (10k + 100k).

The additional 100k rows are 50k paired task topics, not 100k unique human
conversations. Most are deterministic structured/math tasks, and ALL are
unreviewed synthetic experiments. Do not claim the resulting model is fluent.

Generation is offline, deterministic, and preserves the prior 10k rows.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

from bilingual_data import load_records, normalize_prompt, split_records
from generate_bilingual_10k import generate as generate_previous, LEXICON, _family_split

ROOT = Path(__file__).resolve().parent
SOURCE_VERSION = "srt-m15-v4-110k-20261023"

# Additional English-to-Chinese words used only for educational translation
# examples; isolated word translations are context dependent.
MORE_WORDS = """umbrella|雨伞
backpack|背包
wallet|钱包
classroom|教室
notebook|笔记本
headphones|耳机
mirror|镜子
shelf|架子
towel|毛巾
blanket|毯子
""".splitlines()

PEOPLE = [
    ("小明", "Ming"), ("小华", "Hua"), ("小丽", "Li"), ("小雨", "Yu"),
    ("小东", "Dong"), ("小芳", "Fang"), ("小峰", "Feng"), ("小宁", "Ning"),
    ("小林", "Lin"), ("小雪", "Xue"), ("小文", "Wen"), ("小亮", "Liang"),
    ("小梅", "Mei"), ("小涛", "Tao"), ("小月", "Yue"), ("小浩", "Hao"),
    ("小玉", "Yu'er"), ("小洁", "Jie"), ("小安", "An"), ("小杰", "Jie'er"),
    ("小琴", "Qin"), ("小贝", "Bei"), ("小森", "Sen"), ("小晨", "Chen"),
    ("小兰", "Lan"),
]
PLACES = [
    ("图书馆", "library"), ("教室", "classroom"), ("办公室", "office"),
    ("公园", "park"), ("车站", "station"), ("商店", "shop"),
    ("食堂", "cafeteria"), ("会议室", "meeting room"),
    ("花园", "garden"), ("体育馆", "gym"),
    ("候车室", "waiting room"), ("阅览室", "reading room"),
]
ACTIONS = [
    ("阅读", "reading"), ("学习", "studying"), ("整理资料", "organizing notes"),
    ("写报告", "writing a report"), ("讨论计划", "discussing a plan"),
    ("检查设备", "checking equipment"), ("准备展示", "preparing a presentation"),
    ("复习课程", "reviewing a course"), ("练习表达", "practicing speaking"),
    ("讨论问题", "discussing a problem"),
]
VOCAB_TEMPLATES = [
    ("“{en}”在日常中文里通常怎么说？", 'How do you usually say "{zh}" in English?'),
    ("请给出英语名词“{en}”的中文翻译。", 'Give an English noun for "{zh}".'),
    ("在词汇卡片上，“{en}”的中文对应是什么？", 'On a vocabulary card, what is the English for "{zh}"?'),
    ("读到“{en}”时应该联想到哪个中文词？", 'What English word commonly means "{zh}"?'),
    ("学习英文“{en}”，最常见的中文译法是什么？", 'When studying English, how might you translate "{zh}"?'),
    ("请把“{en}”翻译成普通中文。", 'Translate the Chinese word "{zh}" into plain English.'),
    ("英语单词“{en}”一般能译成什么？", 'What English term corresponds to "{zh}"?'),
    ("课堂上如何翻译词语“{en}”？", 'In class, how could you translate "{zh}"?'),
    ("“{en}”的一个常用中文意思是什么？", 'Name a common English translation of "{zh}".'),
    ("为单词表写出“{en}”的中文词。", 'Write an English word for the glossary entry "{zh}".'),
]
VOCAB_CONTEXTS = ["", "（用于初级英语词汇表）", "（请只写常见含义）"]


def _split(category: str, family: int) -> str:
    # Stable across runs: place a whole generator template/parameter family in
    # one split rather than putting nearly-identical sentences on both sides.
    return _family_split("v4_" + category, family)


def generate():
    records, prior_stats = generate_previous()
    if len(records) != 10000:
        raise ValueError("prior generator must produce exactly 10,000 records")
    all_topics = {row["topic_id"] for row in records}
    all_prompts = {normalize_prompt(row["prompt"]) for row in records}
    if len(all_topics) != 5000 or len(all_prompts) != 10000:
        raise ValueError("prior source contains duplicate topic/prompt")

    stats = Counter()
    def add(tag, category, family, zq, za, eq, ea):
        if tag in all_topics:
            raise ValueError(f"duplicate topic id {tag}")
        all_topics.add(tag)
        split = _split(category, family)
        for lang, question, answer in (("zh", zq, za), ("en", eq, ea)):
            key = normalize_prompt(question)
            if not key or key in all_prompts:
                raise ValueError(f"duplicate or empty question: {question}")
            if not question.strip() or not answer.strip():
                raise ValueError("missing prompt/answer")
            if len(question) > 1024 or len(answer) > 2048:
                raise ValueError("question/answer too long")
            all_prompts.add(key)
            records.append({
                "language": lang, "prompt": question, "response": answer,
                "topic_id": tag, "family_id": f"v4-{category}-{family:04d}",
                "category": category, "split": split, "verified": False,
                "review_status": "unreviewed", "synthetic": True,
                "source": SOURCE_VERSION,
            })
        stats[category] += 1

    # Deterministic, verifiably calculated short answers (34,000 paired topics).
    for i in range(7000):
        a, b = 4000+i, 19 + (i*43) % 1901
        add(f"v4-add-{i:05d}", "addition", i//140,
            f"{a} 加上 {b} 是多少？", f"{a} + {b} = {a+b}。",
            f"Find the sum of {a} and {b}.", f"{a} + {b} = {a+b}.")
    for i in range(6000):
        a, b = 20000+i, 41 + (i*37) % 1951
        add(f"v4-sub-{i:05d}", "subtraction", i//120,
            f"请算出 {a} 减去 {b} 的结果。", f"{a} - {b} = {a-b}。",
            f"Compute {a} minus {b}.", f"{a} - {b} = {a-b}.")
    for i in range(5000):
        a, b = 310+i, 2 + (i*17) % 99
        add(f"v4-mul-{i:05d}", "multiplication", i//100,
            f"计算乘法 {a} 乘以 {b}。", f"{a} × {b} = {a*b}。",
            f"Multiply {a} by {b}.", f"{a} × {b} = {a*b}.")
    for i in range(4000):
        q, d = 90+i, 2+(i*23)%89
        n = q*d
        add(f"v4-div-{i:05d}", "division", i//80,
            f"计算 {n} 除以 {d}。", f"{n} ÷ {d} = {q}。",
            f"Divide {n} by {d}.", f"{n} ÷ {d} = {q}.")
    for i in range(3000):
        amount, pct = (120+i)*100, 1+(i*7)%40
        answer = (120+i)*pct
        add(f"v4-pct-{i:05d}", "percentages", i//60,
            f"{amount} 的 {pct}% 是多少？", f"{amount} × {pct}% = {answer}。",
            f"What is {pct}% of {amount}?", f"{pct}% of {amount} is {answer}.")
    for i in range(3000):
        a, b = 1000 + (i*11)%4500, 1900 + (i*37)%4700
        if a == b:
            za, ea = "两个数相等。", "The numbers are equal."
        else:
            za, ea = f"{max(a,b)} 较大。", f"{max(a,b)} is larger."
        add(f"v4-compare-{i:05d}", "comparison", i//60,
            f"数字 {a} 和 {b} 哪个更大？", za,
            f"Which number is greater: {a} or {b}?", ea)
    for i in range(3000):
        first, delta = 100+i, 2+(i*19)%35
        seq = [first+j*delta for j in range(4)]
        add(f"v4-sequence-{i:05d}", "sequences", i//60,
            f"等差数列 {seq[0]}、{seq[1]}、{seq[2]} 的下一项是？",
            f"公差为 {delta}，下一项是 {seq[3]}。",
            f"What comes next in the arithmetic sequence {seq[0]}, {seq[1]}, {seq[2]}?",
            f"The step is {delta}, so the next term is {seq[3]}.")
    for i in range(3000):
        a, b, c = 800+i, 100+(i*17)%5500, 190+(i*71)%6300
        nums = sorted((a,b,c))
        sorted_text = ", ".join(str(v) for v in nums)
        add(f"v4-order-{i:05d}", "sorting", i//60,
            f"把 {a}、{b}、{c} 从小到大排列。",
            f"从小到大是：{sorted_text}。",
            f"Put {a}, {b}, and {c} in ascending order.",
            f"Ascending order: {sorted_text}.")

    # Six families of simple word problems, each with 1000 distinct quantities.
    for i in range(6000):
        mode = i//1000
        x, y, z = 80+i, 2+(i*7)%70, 1+(i*13)%20
        if mode == 0:
            zq = f"虚构场景：小明有 {x} 本书，又买了 {y} 本，一共有多少本？"
            eq = f"Fictional problem: Ming has {x} books and buys {y} more. How many altogether?"
            ans = x+y
        elif mode == 1:
            zq = f"虚构场景：仓库原有 {x} 箱货物，发走 {y} 箱，还剩多少箱？"
            eq = f"Fictional problem: A store has {x} boxes and ships {y}. How many remain?"
            ans = x-y
        elif mode == 2:
            zq = f"虚构场景：每排有 {y} 个座位，共有 {x} 排，座位总数是多少？"
            eq = f"Fictional problem: There are {x} rows with {y} seats each. How many seats?"
            ans = x*y
        elif mode == 3:
            zq = f"虚构场景：一天读 {y} 页书，连续读 {x} 天，一共读多少页？"
            eq = f"Fictional problem: Someone reads {y} pages daily for {x} days. How many pages?"
            ans = x*y
        elif mode == 4:
            zq = f"虚构场景：钱包有 {x} 元，花了 {y} 元，又收入 {z} 元，余额多少元？"
            eq = f"Fictional problem: You have {x} dollars, spend {y}, then receive {z}. Balance?"
            ans = x-y+z
        else:
            zq = f"虚构场景：每袋有 {y} 个苹果，一共装 {x} 袋，有多少个苹果？"
            eq = f"Fictional problem: There are {x} bags with {y} apples each. How many apples?"
            ans = x*y
        add(f"v4-wordproblem-{i:05d}", "practical_word_problems",
            (mode*20) + (i%1000)//50, zq, f"计算结果是 {ans}。",
            eq, f"The result is {ans}.")

    # Times, duration, and representation (4,000 paired topics).
    for i in range(4000):
        variant = i%3
        # Distinct time across variants: each has <= 1334 possibilities.
        serial = i//3
        hour, minute = serial//60, serial%60
        time24 = f"{hour:02d}:{minute:02d}"
        period = "AM" if hour < 12 else "PM"
        hour12 = hour % 12 or 12
        if variant == 0:
            zq = f"24 小时制 {time24} 对应的 12 小时制时间是什么？"
            eq = f"Convert {time24} from 24-hour time to 12-hour time."
            za = f"{hour12}:{minute:02d} {period}。"
            ea = f"{hour12}:{minute:02d} {period}."
        elif variant == 1:
            minutes = hour*60+minute
            zq = f"从午夜 00:00 到 {time24} 过去了多少分钟？"
            eq = f"How many minutes after midnight is {time24}?"
            za, ea = f"{minutes} 分钟。", f"{minutes} minutes."
        else:
            newtime = (datetime(2026, 1, 1, hour, minute) +
                       timedelta(minutes=45)).strftime("%H:%M")
            zq = f"24 小时制时间 {time24} 再过 45 分钟是几点？"
            eq = f"What time is 45 minutes after {time24} in 24-hour notation?"
            za, ea = f"{newtime}。", f"{newtime}."
        add(f"v4-time-{i:05d}", "time_reasoning", variant*25+serial//60,
            zq, za, eq, ea)

    # Bilingual basic vocabulary in 30 classroom/formality contexts. These
    # are paraphrased translations and NOT 3,000 genuinely distinct facts.
    words = [line.split("|") for line in LEXICON] + [
        line.split("|") for line in MORE_WORDS if line.strip()]
    if len(words) != 100 or any(len(row) != 2 for row in words):
        raise ValueError("lexicon must contain exactly 100 word pairs")
    for word_index, (en, zh) in enumerate(words):
        for variant in range(30):
            template_z, template_e = VOCAB_TEMPLATES[variant%10]
            suffix = VOCAB_CONTEXTS[variant//10]
            zq = template_z.format(en=en, zh=zh) + suffix
            eq = template_e.format(en=en, zh=zh)
            if variant//10 == 1:
                eq += " Please use a common meaning."
            elif variant//10 == 2:
                eq += " For a beginner's glossary."
            add(f"v4-word-{word_index:03d}-{variant:02d}",
                "common_vocabulary", word_index,
                zq, f"一个常见对应词是“{zh}”。",
                eq, f'A common translation is "{en}".')

    # 25 fictional people x 12 places x 10 activities = 3,000 unique
    # structured field extraction problems.
    for person_i, (cn_name, en_name) in enumerate(PEOPLE):
        for place_i, (cn_place, en_place) in enumerate(PLACES):
            for action_i, (cn_action, en_action) in enumerate(ACTIONS):
                n = person_i*120 + place_i*10 + action_i
                field = n%3
                zh_record = f"姓名：{cn_name}；地点：{cn_place}；活动：{cn_action}"
                en_record = f"Name: {en_name}; place: {en_place}; activity: {en_action}"
                if field == 0:
                    zq = f"从虚构记录“{zh_record}”中提取地点。"
                    eq = f"Extract the place from this fictional record: {en_record}."
                    za, ea = f"地点是{cn_place}。", f"The place is {en_place}."
                elif field == 1:
                    zq = f"读虚构记录“{zh_record}”，参与者是谁？"
                    eq = f"Who is named in this fictional record? {en_record}."
                    za, ea = f"参与者是{cn_name}。", f"The person is {en_name}."
                else:
                    zq = f"虚构记录“{zh_record}”中，活动是什么？"
                    eq = f"Identify the activity in this fictional record: {en_record}."
                    za, ea = f"活动是{cn_action}。", f"The activity is {en_action}."
                add(f"v4-extract-{n:05d}", "information_extraction",
                    person_i, zq, za, eq, ea)

    expected = {
        "addition": 7000, "subtraction": 6000, "multiplication": 5000,
        "division": 4000, "percentages": 3000, "comparison": 3000,
        "sequences": 3000, "sorting": 3000, "practical_word_problems": 6000,
        "time_reasoning": 4000, "common_vocabulary": 3000,
        "information_extraction": 3000,
    }
    if dict(stats) != expected:
        raise AssertionError(f"generation count mismatch: {dict(stats)}")
    if len(records) != 110000 or Counter(r["language"] for r in records) != {
        "zh": 55000, "en": 55000
    }:
        raise AssertionError(f"expected 110,000 rows, got {len(records)}")
    if len(all_topics) != 55000 or len(all_prompts) != 110000:
        raise AssertionError("duplicate topic or normalized prompt")
    return records, {"prior_10k": prior_stats, "new_paired_topics": dict(stats)}


def build(output: Path, *, manifest=None):
    rows, stats = generate()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    output.write_text(payload, encoding="utf-8")
    records = load_records(output, allow_demo=True)
    train, dev = split_records(records, seed=42)
    train_families = {
        row["family_id"] for row in rows if row.get("family_id") and
        row["split"] == "train"
    }
    dev_families = {
        row["family_id"] for row in rows if row.get("family_id") and
        row["split"] == "dev"
    }
    if train_families & dev_families:
        raise AssertionError("template family leakage detected")
    assert not ({r.topic_id for r in train} & {r.topic_id for r in dev})
    result = {
        "format": "srt-m15-synthetic-110k-v4",
        "generator": SOURCE_VERSION,
        "total_records": len(rows),
        "added_records": 100000,
        "total_paired_topics": 55000,
        "by_language": dict(Counter(r["language"] for r in rows)),
        "train_records": len(train), "dev_records": len(dev),
        "data_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "new_topics_by_category": stats["new_paired_topics"],
        "all_verified": False, "human_reviewed": False,
        "warning": "Synthetic template demonstrations are not independent natural-language data.",
    }
    if manifest:
        Path(manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(manifest).write_text(json.dumps(result, ensure_ascii=False, indent=2) +
                                  "\n", encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path,
                   default=ROOT / "examples/bilingual_synthetic_v4_110000.jsonl")
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "examples/bilingual_synthetic_v4_manifest.json")
    a = p.parse_args()
    print(json.dumps(build(a.output, manifest=a.manifest),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
