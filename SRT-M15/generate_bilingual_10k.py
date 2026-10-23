"""Reproducibly create exactly 10,000 bilingual synthetic SRT-M15 QA records.

Uses the existing 1,000-row corpus verbatim, then 4,500 new paired topics.
The generated examples are templated demonstrations, NOT reviewed facts or
proof of broad Chinese/English fluency. No web scraping, API calls, or secrets.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from bilingual_data import load_records, normalize_prompt, split_records

ROOT = Path(__file__).resolve().parent
SEED_DATA = ROOT / "examples/bilingual_synthetic_v2_1000.jsonl"

# Common everyday nouns. These are short hand-authored translations; a real
# language dataset needs independent linguistic review and diverse sources.
LEXICON = """cup|杯子
book|书
desk|书桌
light|灯
chair|椅子
door|门
street|街道
bus|公共汽车
train|火车
plane|飞机
boat|船
park|公园
shop|商店
bank|银行
school|学校
clinic|诊所
office|办公室
meeting|会议
message|消息
email|电子邮件
question|问题
answer|答案
story|故事
picture|图片
screen|屏幕
battery|电池
charger|充电器
clock|时钟
calendar|日历
morning|早晨
night|夜晚
breakfast|早餐
lunch|午餐
dinner|晚餐
water|水
tea|茶
coffee|咖啡
milk|牛奶
bread|面包
rice|米饭
fruit|水果
apple|苹果
banana|香蕉
orange|橙子
table|桌子
phone|电话
computer|计算机
mouse|鼠标
keyboard|键盘
file|文件
folder|文件夹
password|密码
account|账户
website|网站
network|网络
server|服务器
garden|花园
flower|花
tree|树
river|河流
lake|湖
mountain|山
cloud|云
sun|太阳
rain|雨
wind|风
snow|雪
spring|春天
summer|夏天
autumn|秋天
winter|冬天
friend|朋友
neighbor|邻居
teacher|老师
student|学生
parent|家长
child|孩子
doctor|医生
driver|司机
worker|工人
manager|经理
team|团队
project|项目
schedule|日程安排
budget|预算
price|价格
ticket|票
passport|护照
museum|博物馆
library|图书馆
""".strip().splitlines()

# Situation, English rendering, safe/general advice in both languages,
# and a polite help request in both languages.
SITUATIONS = """我的电脑启动很慢|my computer starts slowly|先记录启动时间并检查启动项。|Record the startup time and check startup applications.|能否帮我检查电脑的启动设置？|Could you help me check the startup settings?
我赶不上原定的车|I cannot catch my scheduled ride|先核对下一班交通方式并通知受影响的人。|Check the next available ride and inform anyone affected.|请问下一班车什么时候出发？|When does the next ride leave?
我忘记带笔记本|I forgot my notebook|先记录重要信息，之后补充到笔记本中。|Record important points temporarily and copy them into your notebook later.|请问可以借我一张纸吗？|Could I borrow a sheet of paper?
我找不到预约邮件|I cannot find my appointment email|检查收件箱、垃圾邮件和相关账户。|Check your inbox, spam folder, and the relevant account.|能否帮我重新发送预约确认？|Could you resend the appointment confirmation?
我迟到了十分钟|I am ten minutes late|主动道歉并说明自己预计到达的时间。|Apologize and state when you expect to arrive.|抱歉我迟到了，可以现在开始吗？|Sorry I'm late. Can we begin now?
我不理解作业要求|I do not understand the assignment|先列出不明确的部分，再向老师确认。|List the unclear parts and ask the teacher for clarification.|请问可以解释一下作业的评分标准吗？|Could you explain the assignment criteria?
我需要修改预约时间|I need to reschedule an appointment|查看可选时间并尽早联系对方。|Check available times and contact the other party early.|请问可以把预约改到下周吗？|Could we move the appointment to next week?
我把东西落在出租车上|I left something in a taxi|保存行程记录并尽快联系平台或司机。|Keep the trip details and contact the driver or service promptly.|能否帮我联系司机查询失物？|Could you help me contact the driver about a lost item?
我收到一封可疑邮件|I received a suspicious email|不要点击未知链接，先核实发件人。|Avoid unknown links and verify the sender.|能否帮我确认这封邮件是否来自官方？|Could you help verify whether this email is official?
我的快递迟迟没有送到|my package has not arrived|先查看物流状态并核对预计送达日期。|Check the tracking status and estimated delivery date.|请问可以帮我查询包裹的最新状态吗？|Could you check the latest tracking status?
我忘记了朋友的生日|I forgot a friend's birthday|真诚致歉并表达关心，不必编造借口。|Apologize sincerely and express care without inventing excuses.|我想补上一句生日祝福，可以吗？|May I send you a belated birthday wish?
我需要一个安静的学习环境|I need a quiet place to study|寻找较少干扰的空间并关闭无关通知。|Find a low-distraction location and silence unnecessary notifications.|这里有安静的自习区域吗？|Is there a quiet study area here?
我手机存储空间不足|my phone is low on storage|先查看占用情况，备份后再清理不需要的文件。|Check storage usage and back up files before deleting them.|请问可以帮我查看哪些文件占空间吗？|Could you help me see which files take up space?
我的水杯漏水了|my water bottle is leaking|先把水杯移开电子设备并清理漏出的水。|Move the bottle away from electronics and clean up spilled water.|请问能借我一些纸巾吗？|Could I borrow some tissues?
我听不清楚视频中的说话声|I cannot hear the speech in a video|检查音量、字幕和播放设备的设置。|Check the volume, captions, and playback device settings.|这段视频可以打开字幕吗？|Could you turn on captions for this video?
我在超市找不到某种商品|I cannot find a product at the supermarket|先查看分类指示，再询问工作人员。|Check aisle signs before asking a staff member.|请问这种商品放在哪个货架？|Which aisle has this item?
我需要向同事说明延期|I need to explain a delay to a coworker|说明原因、当前进度及新的时间安排。|Explain the cause, current status, and revised timeline.|我们可以重新确认交付时间吗？|Could we confirm a new delivery date?
我想向老师请教问题|I want to ask my teacher a question|先整理你已经尝试的方法，再提出具体疑问。|Summarize what you tried and ask a specific question.|请问您方便帮我看一下这道题吗？|Would you have a moment to look at this problem?
我的航班信息发生变化|my flight details have changed|通过官方渠道核对最新航班及登机信息。|Verify the updated flight and boarding details through official channels.|请问最新的登机时间是多少？|What is the updated boarding time?
我需要写一份活动通知|I need to write an event notice|写清活动主题、时间、地点及联系方式。|Include the event topic, date, location, and contact information.|能否帮我核对通知是否清楚？|Could you check whether the notice is clear?
我的电脑连接不上无线网络|my computer cannot connect to Wi-Fi|先确认网络名称、密码和其他设备是否能连接。|Check the network name, password, and whether other devices can connect.|请问可以帮我检查无线网络设置吗？|Could you help me check the Wi-Fi settings?
我看不清楚投影的文字|I cannot read the projected text|先调整座位，再考虑请求放大内容。|Try changing seats or asking to enlarge the content.|可以请您放大一下字体吗？|Could you make the text larger, please?
我想退掉不合适的衣服|I want to return clothes that do not fit|先查看退货条件，保留票据和商品标签。|Check the return policy and keep the receipt and tags.|请问这件衣服可以退换吗？|Can I return or exchange these clothes?
我错过了报名截止时间|I missed the registration deadline|查看是否还有补报途径，不要假定一定可以延期。|Check whether late registration is offered; do not assume it is.|请问现在还有补报名额吗？|Is late registration still available?
我的行李箱轮子坏了|my suitcase wheel is broken|检查是否还能安全搬运，必要时寻找维修或替代方式。|Check whether it is safe to move and consider repair or alternatives.|请问附近有行李箱维修服务吗？|Is there a luggage repair service nearby?
我想为朋友准备惊喜|I want to plan a surprise for a friend|先考虑对方的喜好和可接受的安排。|Consider your friend's preferences and comfort.|你可以帮我安排一个简单的庆祝活动吗？|Could you help me plan a simple celebration?
我遇到不熟悉的英文表达|I came across an unfamiliar English phrase|查看完整语境并核对可靠词典中的用法。|Read the context and check usage in a reliable dictionary.|请问这句话在这里是什么意思？|What does this phrase mean in this context?
我参加会议时没有听清决定|I missed a decision during the meeting|查看会议记录并向相关人员确认。|Review the meeting notes and confirm with those involved.|可以再说明一下刚才的决定吗？|Could you restate the decision we just made?
我忘记了账户登录密码|I forgot my account password|使用服务提供的正规找回流程，避免向他人透露验证码。|Use the official recovery process and avoid sharing verification codes.|请问在哪里可以重置密码？|Where can I reset my password?
我想确认考试地点|I need to confirm the exam location|查阅正式考试通知并提前规划路线。|Read the official exam notice and plan your route ahead of time.|请问可以告诉我考试的具体地址吗？|Could you confirm the exact exam address?
""".strip().splitlines()

ASK_ZH = (
    "请回答这个计算题：{x}。",
    "请计算并写出结果：{x}。",
)
ASK_EN = (
    "Solve this calculation: {x}.",
    "Calculate and provide the result: {x}.",
)
VOCAB_ZH = [
    "“{en}”用中文怎么说？", "请把英语词语“{en}”翻译成中文。",
    "日常英语中的“{en}”是什么意思？", "如何理解英文名词“{en}”？",
    "请给出“{en}”的常见中文对应词。", "学习单词“{en}”时常见的中文含义是什么？",
    "英语词汇“{en}”的中文解释是什么？", "帮我把“{en}”译成中文。",
    "“{en}”这个英文词通常指什么？", "请告诉我“{en}”的中文意思。",
]
VOCAB_EN = [
    'How do you say "{zh}" in English?', 'Translate "{zh}" into English.',
    'What is the usual English word for "{zh}"?', 'Give an English equivalent of "{zh}".',
    'Which English noun commonly means "{zh}"?', 'How would you translate "{zh}"?',
    'What English word can express "{zh}"?', 'Provide the English for "{zh}".',
    'What is a common translation of "{zh}"?', 'What do you call "{zh}" in English?',
]
SCENE_QZ = [
    "遇到“{scene}”时，该怎么说明？", "如何向他人解释“{scene}”？",
    "能帮我简要描述“{scene}”吗？", "请给出关于“{scene}”的一句说明。",
    "向同事说明“{scene}”的情况，该怎么说？", "我要描述“{scene}”，请给个表达。",
    "“{scene}”这种情况怎么简短表达？", "如何清楚描述“{scene}”的现状？",
    "描述“{scene}”时应说什么？", "请写出“{scene}”的简短说明。",
]
SCENE_QE = [
    "How can I describe that {scene}?", "Explain briefly that {scene}.",
    "What could I say when {scene}?", "Write a short explanation for when {scene}.",
    "How should I tell a colleague that {scene}?", "Help me explain that {scene}.",
    "What is a concise way to describe that {scene}?", "Describe the situation where {scene}.",
    "How could I communicate that {scene}?", "Please summarize what to say when {scene}.",
]
SCENE_ADVICE_ZH = [
    "当{scene}时，我应该先做什么？", "遇到{scene}可以采取什么第一步？",
    "关于{scene}，有什么基本建议？", "如何开始处理{scene}？",
    "假设{scene}，你建议怎么做？", "对{scene}这种情况该怎么办？",
    "如果{scene}，先采取什么行动？", "请简短建议如何应对{scene}。",
    "在{scene}时，有什么谨慎做法？", "解决{scene}时第一步是什么？",
]
SCENE_ADVICE_EN = [
    "What should I do first if {scene}?", "What is a first step when {scene}?",
    "Any basic advice if {scene}?", "How can I start handling it if {scene}?",
    "What do you suggest if {scene}?", "How should I respond when {scene}?",
    "What action comes first if {scene}?", "Give brief advice for when {scene}.",
    "What is a careful response if {scene}?", "What would the first step be if {scene}?",
]
SCENE_ASK_ZH = [
    "假设{scene}，如何礼貌求助？", "遇到{scene}应该怎么开口请求帮助？",
    "请为{scene}写一句求助语。", "我需要别人帮忙处理{scene}，怎么说？",
    "当{scene}时，能给一句请求吗？", "如何委婉地为{scene}求助？",
    "帮我写一句关于{scene}的礼貌提问。", "处理{scene}时怎么询问工作人员？",
    "遇到{scene}，请写一句请求帮助的话。", "请给出{scene}时可以说的一句礼貌请求。",
]
SCENE_ASK_EN = [
    "How can I politely ask for help if {scene}?", "How should I request help when {scene}?",
    "Write a help request for when {scene}.", "How do I ask someone to help if {scene}?",
    "Suggest a polite request if {scene}.", "How can I ask nicely when {scene}?",
    "Give a polite question for a situation where {scene}.", "How can I ask staff for help if {scene}?",
    "Write one request for help when {scene}.", "Give a courteous request if {scene}.",
]


def _family_split(category: str, family: int) -> str:
    # Entire source families are held out. Never let small paraphrase changes
    # from one family leak from training into validation.
    digest = hashlib.sha256(f"{category}:{family}:srt-m15-v3".encode()).digest()
    return "dev" if int.from_bytes(digest[:4], "big") % 5 == 0 else "train"


def generate(seed_file: Path = SEED_DATA):
    rows = [json.loads(line) for line in seed_file.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 1000 or Counter(r["language"] for r in rows) != {"zh": 500, "en": 500}:
        raise ValueError("expected exactly 1000 existing bilingual source records")
    topics = {r["topic_id"] for r in rows}
    prompts = {normalize_prompt(r["prompt"]) for r in rows}
    if len(topics) != 500 or len(prompts) != 1000:
        raise ValueError("source dataset has duplicate questions or topics")
    stats = Counter()
    def add(name, category, family, zh_q, zh_a, en_q, en_a):
        if name in topics:
            raise ValueError("duplicate topic ID: " + name)
        topics.add(name)
        split = _family_split(category, family)
        for lang, q, a in (("zh", zh_q, zh_a), ("en", en_q, en_a)):
            key = normalize_prompt(q)
            if not q or not a or key in prompts:
                raise ValueError("empty or duplicated prompt: " + q)
            prompts.add(key)
            if len(q) > 1024 or len(a) > 2048:
                raise ValueError("text exceeds the dataset format limits")
            rows.append({
                "language": lang, "prompt": q, "response": a,
                "topic_id": name, "category": category, "split": split,
                "verified": False, "review_status": "unreviewed",
                "source": "assistant-authored-synthetic-generator-v3",
                "synthetic": True, "family_id": f"{category}-{family:03d}",
            })
        stats[category] += 1

    for i in range(600):
        a, b = 130 + i, 11 + (i * 37) % 103
        expr_zh, expr_en = f"{a} 加 {b}", f"{a} plus {b}"
        add(f"v3-add-{i:04d}", "arithmetic_add", i % 30,
            f"计算：{expr_zh} 等于多少？", f"{a} + {b} = {a+b}。",
            f"What is {expr_en}?", f"{a} + {b} = {a+b}.")
    for i in range(600):
        a, b = 900 + i, 20 + (i * 31) % 121
        add(f"v3-sub-{i:04d}", "arithmetic_subtract", i % 30,
            f"{a} 减 {b} 的结果是多少？", f"{a} - {b} = {a-b}。",
            f"Calculate {a} minus {b}.", f"{a} - {b} = {a-b}.")
    for i in range(600):
        a, b = 10 + i, 2 + (i * 11) % 19
        add(f"v3-mul-{i:04d}", "arithmetic_multiply", i % 30,
            f"请计算 {a} 乘 {b}。", f"{a} × {b} = {a*b}。",
            f"What is {a} multiplied by {b}?", f"{a} × {b} = {a*b}.")
    for i in range(300):
        a, b = 10 + i, 12 + (i * 29) % 113
        answer_zh = f"{max(a,b)} 更大。" if a != b else "这两个数相等。"
        answer_en = f"{max(a,b)} is larger." if a != b else "They are equal."
        add(f"v3-compare-{i:04d}", "number_comparison", i % 30,
            f"比较数字 {a} 与 {b}，哪个更大？", answer_zh,
            f"Which is larger, {a} or {b}?", answer_en)
    for i in range(300):
        value = 200 + i
        parity_zh = "偶数" if value % 2 == 0 else "奇数"
        parity_en = "even" if value % 2 == 0 else "odd"
        add(f"v3-parity-{i:04d}", "number_parity", i % 30,
            f"整数 {value} 是奇数还是偶数？", f"{value} 是{parity_zh}。",
            f"Is the integer {value} odd or even?", f"{value} is {parity_en}.")
    for i in range(300):
        a, b, c = 50 + i, 12 + (i * 13) % 47, 160 + (i * 7) % 79
        sorted_nums = ", ".join(str(z) for z in sorted((a,b,c)))
        add(f"v3-sort-{i:04d}", "number_sorting", i % 30,
            f"请将 {a}、{b}、{c} 从小到大排列。", "从小到大为：" + sorted_nums + "。",
            f"Sort {a}, {b}, and {c} in ascending order.", "Ascending order: " + sorted_nums + ".")
    entries = [line.split("|") for line in LEXICON]
    if len(entries) != 90 or any(len(e) != 2 for e in entries):
        raise ValueError("expected 90 bilingual vocabulary entries")
    for i, (en, zh) in enumerate(entries):
        for variant in range(10):
            add(f"v3-lex-{i:03d}-{variant:02d}", "vocabulary",
                i, VOCAB_ZH[variant].format(en=en),
                f"“{en}”常见的中文对应词是“{zh}”。",
                VOCAB_EN[variant].format(zh=zh),
                f'A common English word for "{zh}" is "{en}".')
    scenes = [line.split("|") for line in SITUATIONS]
    if len(scenes) != 30 or any(len(r) != 6 for r in scenes):
        raise ValueError("expected 30 situation records")
    for i, (zh_scene, en_scene, zh_advice, en_advice, zh_ask, en_ask) in enumerate(scenes):
        for variant in range(10):
            add(f"v3-scene-explain-{i:03d}-{variant:02d}", "situation",
                i, SCENE_QZ[variant].format(scene=zh_scene),
                "可以说明当前情况并补充具体影响：" + zh_scene + "。",
                SCENE_QE[variant].format(scene=en_scene),
                "You can explain the issue and its impact: " + en_scene + ".")
            add(f"v3-scene-advice-{i:03d}-{variant:02d}", "situation",
                i, SCENE_ADVICE_ZH[variant].format(scene=zh_scene),
                zh_advice,
                SCENE_ADVICE_EN[variant].format(scene=en_scene),
                en_advice)
            add(f"v3-scene-ask-{i:03d}-{variant:02d}", "situation",
                i, SCENE_ASK_ZH[variant].format(scene=zh_scene),
                "可以礼貌地问：“" + zh_ask + "”",
                SCENE_ASK_EN[variant].format(scene=en_scene),
                'You could politely ask: "' + en_ask + '"')
    if len(rows) != 10000 or Counter(r["language"] for r in rows) != {"zh": 5000, "en": 5000}:
        raise AssertionError(f"wrong total/locale count: {len(rows)}")
    if len(topics) != 5000 or len(prompts) != 10000:
        raise AssertionError("not exactly 5000 unique paired topics and 10000 prompts")
    return rows, dict(stats)


def write_dataset(output: Path, *, seed_file=SEED_DATA, manifest=None):
    rows, categories = generate(seed_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                      for row in rows)
    output.write_text(payload, encoding="utf-8")
    # Use the same grouped split logic the trainer uses. Fail before publishing
    # if any group crosses train and dev.
    records = load_records(output, allow_demo=True)
    train, dev = split_records(records, seed=42)
    train_topics = {x.topic_id for x in train}
    if train_topics.intersection(x.topic_id for x in dev):
        raise AssertionError("paired topic crossed training/validation")
    data = {
        "source": str(seed_file),
        "dataset": str(output),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "total_records": len(rows),
        "total_paired_topics": len(rows) // 2,
        "by_language": dict(Counter(r["language"] for r in rows)),
        "added_paired_topics_by_category": categories,
        "train_records": len(train), "dev_records": len(dev),
        "verified": False,
        "warning": "Large synthetic template dataset; not a general language benchmark.",
    }
    if manifest:
        path = Path(manifest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-file", type=Path, default=SEED_DATA)
    p.add_argument("--output", type=Path, default=ROOT / "examples/bilingual_synthetic_v3_10000.jsonl")
    p.add_argument("--manifest", type=Path, default=ROOT / "examples/bilingual_synthetic_v3_manifest.json")
    args = p.parse_args()
    print(json.dumps(write_dataset(args.output, seed_file=args.seed_file,
                                  manifest=args.manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
