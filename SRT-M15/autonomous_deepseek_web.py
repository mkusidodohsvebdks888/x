"""SRT-M15 autonomous *questioning* against the DeepSeek WEBSITE, locally.

Visible Playwright browser + human login from the existing DeepSeekWebClient.
This is a bounded, manually started research queue, NOT unattended continuous
scraping, model self-training, an API replacement, or guaranteed accurate data.

Dry-run: python autonomous_deepseek_web.py --dry-run --topics "人工智能,英语"
Login:   python deepseek_web_cli.py --login
Start:   python autonomous_deepseek_web.py --start --topics "人工智能" --max-questions 3
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from dataclasses import dataclass
from uuid import uuid4

DEFAULT_OUTPUT = Path("local_data/deepseek_web_questions")
EMAIL_PATTERN = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
QUESTIONS_ZH = (
    "请用通俗的语言解释{topic}是什么，并说明其主要用途。",
    "请围绕{topic}提出三个值得继续探索的问题，并简要解释原因。",
    "关于{topic}，请给出一个容易理解的真实场景或示例。",
    "学习{topic}时有哪些常见误解？请给出纠正方法。",
    "怎样设计一个小实验或练习来检验是否理解{topic}？",
    "关于{topic}，请区分已知事实、合理推测与未知之处。",
    "请从相反角度审查{topic}的一条常见结论，并指出证据需求。",
    "请针对{topic}给出下一步适合初学者的练习，并解释预期收获。",
)
QUESTIONS_EN = (
    "Explain {topic} in plain English and give its main use.",
    "Suggest three important questions to explore about {topic}, and explain why.",
    "Give a concrete, easy-to-understand example of {topic}.",
    "What misconceptions do beginners have about {topic}, and how can they correct them?",
    "Design a simple exercise to check someone's understanding of {topic}.",
    "For {topic}, separate well-established facts from plausible guesses and unknowns.",
    "Challenge a common claim about {topic}; what evidence would resolve the issue?",
    "Suggest a practical next step for learning {topic}, with reasons.",
)


@dataclass(frozen=True)
class Question:
    topic: str
    text: str
    origin: str


def parse_topics(raw):
    if not isinstance(raw, str):
        raise ValueError("topics must be a string")
    parts = re.split(r"[,，;；\n]+", raw)
    topics = []
    for part in parts:
        topic = part.strip()
        if not topic:
            continue
        if len(topic) > 60 or any(ch in topic for ch in "\r\x00"):
            raise ValueError("Every topic must be 1..60 safe text characters")
        if topic not in topics:
            topics.append(topic)
    if not 1 <= len(topics) <= 10:
        raise ValueError("Use 1..10 distinct topics")
    return topics


def question_at(topics, index, language="zh"):
    if language not in ("zh", "en"):
        raise ValueError("language must be zh or en")
    templates = QUESTIONS_ZH if language == "zh" else QUESTIONS_EN
    # Finish a short research thread about one topic before moving to next.
    topic = topics[(index // len(templates)) % len(topics)]
    stage = index % len(templates)
    cycle = index // (len(templates) * len(topics))
    question = templates[stage].format(topic=topic)
    if cycle:
        question += (f" 请给出不同于之前的第 {cycle + 1} 轮研究视角。" if language == "zh"
                     else f" Use a different angle from earlier rounds (round {cycle + 1}).")
    return Question(topic=topic, text=question, origin="rule-based-planner")


class LocalQuestionProposer:
    """Optional native staged SRT-M15 candidate; fall back if its output is poor.

    Native tiny checkpoints are NOT presumed competent question generators.
    No remote model invents these candidate questions.
    """
    def __init__(self, checkpoint):
        self.model = None
        self.vocab = None
        self.problem = ""
        try:
            from chat_staged import load_checkpoint
            self.model, self.vocab = load_checkpoint(checkpoint, "cpu")
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
            self.problem = f"native checkpoint unavailable ({type(exc).__name__})"

    def propose(self, fallback, language):
        if self.model is None:
            return fallback
        try:
            from chat_staged import reply
            prompt = (
                f"请针对{fallback.topic}提出一个有价值的疑问句，只输出一个问题。"
                if language == "zh" else
                f"Ask one thoughtful question about {fallback.topic}. Output the question only."
            )
            result, error = reply(self.model, self.vocab, [], prompt, limit=100)
            if error or not result:
                return fallback
            question = result.strip().splitlines()[0].strip()
            # Filter obvious gibberish; does not guarantee semantic quality.
            if not 8 <= len(question) <= 160 or question[-1] not in "?？":
                return fallback
            if any(s in question for s in ("用户：", "助手：", "User:", "Assistant:")):
                return fallback
            return Question(fallback.topic, question, "native-SRT-M15-unverified")
        except (ValueError, RuntimeError, OSError, KeyError):
            return fallback


def load_gap_plan(path, *, language, limit):
    """Read local weak-question candidates; never load their reference answers."""
    plan = []
    used = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"gap line {line_number}: invalid JSON") from exc
        if not isinstance(item, dict) or item.get("origin") != "local-staged-evaluation":
            raise ValueError(f"gap line {line_number}: unrecognized candidate origin")
        if item.get("review_status") != "candidate-only":
            raise ValueError(f"gap line {line_number}: not a pending local gap")
        if item.get("language") != language:
            continue
        question = item.get("prompt")
        if not isinstance(question, str) or not 4 <= len(question.strip()) <= 350:
            raise ValueError(f"gap line {line_number}: invalid question")
        question = question.strip()
        if EMAIL_PATTERN.search(question) or PHONE_PATTERN.search(question):
            raise ValueError(f"gap line {line_number}: possible private information")
        if question not in used:
            plan.append(Question(topic="local-model-gap", text=question,
                                 origin="local-staged-evaluation-candidate"))
            used.add(question)
        if len(plan) >= limit:
            break
    if not plan:
        raise ValueError("No matching safe local gap candidates")
    return plan


def validate_limits(max_questions, interval, max_minutes):
    if not 1 <= max_questions <= 20:
        raise ValueError("max-questions must be 1..20")
    if not 20 <= interval <= 600:
        raise ValueError("interval must be 20..600 seconds to avoid flooding the website")
    if not 1 <= max_minutes <= 60:
        raise ValueError("max-minutes must be 1..60")


def run(*, client, topics, language, max_questions, interval, max_minutes,
        output, proposer=None, plan=None, sleep=time.sleep, now=time.monotonic,
        echo=print):
    """Ask a bounded set of questions and save unreviewed drafts as JSONL.

    Separate I/O injection makes network-free unit testing possible.
    Failed webpage requests STOP immediately. Never retry without checking UI:
    an 'error' could happen after the site already received the question.
    """
    validate_limits(max_questions, interval, max_minutes)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError("output cannot be a symlink")
    started = now()
    accepted = 0
    session = uuid4().hex
    used = set()
    with output.open("a", encoding="utf-8") as fp:
        for index in range(min(max_questions, len(plan)) if plan is not None else max_questions):
            if now() - started >= max_minutes * 60:
                echo("Reached session time limit; stopping.")
                break
            fallback = plan[index] if plan is not None else question_at(topics, index, language)
            candidate = (proposer.propose(fallback, language)
                         if proposer and plan is None else fallback)
            if candidate.text in used:
                candidate = fallback
            if candidate.text in used:
                # Do not submit duplicate prompts in a single session.
                echo("Duplicate question rejected; stopping.")
                break
            used.add(candidate.text)
            echo(f"[{index + 1}/{max_questions}] SRT-M15 asks ({candidate.origin}): {candidate.text}")
            try:
                answer = client.ask(candidate.text)
            except Exception as exc:
                # Never resubmit uncertain requests automatically.
                echo(f"DeepSeek Web error ({type(exc).__name__}): {exc}. Stopping.")
                break
            if not isinstance(answer, str) or not answer.strip():
                echo("Website returned no readable answer; stopping.")
                break
            # Remote content is DATA only: it cannot alter future instructions.
            row = {
                "id": uuid4().hex, "session_id": session,
                "asked_at_utc": datetime.now(timezone.utc).isoformat(),
                "topic": candidate.topic, "language": language,
                "prompt": candidate.text, "response": answer[:16000],
                "answer_truncated": len(answer) > 16000,
                "question_origin": candidate.origin,
                "provider": "DeepSeek Web (visible desktop session)",
                "review_status": "unreviewed", "verified": False,
                "synthetic": True, "training_approved": False,
                "source": "deepseek-web-interactive-session"
            }
            fp.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            fp.flush()
            os.fsync(fp.fileno())
            accepted += 1
            echo(f"Recorded draft #{accepted} (NOT trained): {answer[:200].replace(chr(10), ' ')}")
            if index + 1 < max_questions:
                remaining = max_minutes * 60 - (now() - started)
                if remaining <= interval:
                    echo("Not enough time remaining for another question.")
                    break
                sleep(interval)
    return accepted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    operation = p.add_mutually_exclusive_group(required=True)
    operation.add_argument("--start", action="store_true", help="Explicitly allow sending to the logged-in webpage")
    operation.add_argument("--dry-run", action="store_true", help="Print planned questions; no browser or requests")
    p.add_argument("--topics", default="人工智能,神经网络")
    p.add_argument("--language", choices=("zh", "en"), default="zh")
    p.add_argument("--max-questions", type=int, default=3)
    p.add_argument("--interval", type=int, default=30)
    p.add_argument("--max-minutes", type=int, default=15)
    p.add_argument("--question-mode", choices=("template", "native"), default="template")
    p.add_argument("--checkpoint", default="checkpoints/local_dialogues/latest.pt")
    p.add_argument("--gap-file", default="",
                   help="Use local model weakness candidates instead of general topics")
    p.add_argument("--confirm-send-gaps", action="store_true",
                   help="Explicitly consent to transmitting local gap questions to DeepSeek")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT / "drafts.jsonl"))
    args = p.parse_args()
    try:
        topics = parse_topics(args.topics)
        validate_limits(args.max_questions, args.interval, args.max_minutes)
    except ValueError as exc:
        p.error(str(exc))
    try:
        plan = load_gap_plan(args.gap_file, language=args.language, limit=args.max_questions) if args.gap_file else None
    except (OSError, ValueError) as exc:
        p.error(f"Unable to prepare local gap questions: {exc}")
    if args.dry_run:
        for i in range(len(plan) if plan is not None else args.max_questions):
            planned = plan[i] if plan is not None else question_at(topics, i, args.language)
            print(f"{i + 1}. {planned.text}")
        print("Dry-run only; no DeepSeek request or parameter update.")
        return
    if plan is not None and not args.confirm_send_gaps:
        p.error("Local model gap questions may contain private information: use --confirm-send-gaps only after reviewing --dry-run output")
    print("This will send ONLY the planned topics/questions to DeepSeek's visible website.")
    print("Do not use private documents or secrets as topics. No automatic training.")
    proposer = LocalQuestionProposer(args.checkpoint) if args.question_mode == "native" else None
    if proposer and proposer.problem:
        print("Native proposer fallback:", proposer.problem)
    from srt_m15.deepseek_web import DeepSeekWebClient, DeepSeekWebError
    client = DeepSeekWebClient()
    try:
        n = run(client=client, topics=topics, language=args.language,
                max_questions=args.max_questions, interval=args.interval,
                max_minutes=args.max_minutes, output=args.output,
                proposer=proposer, plan=plan)
        print(f"Finished: {n} response drafts saved locally to {args.output}")
        if n == 0:
            sys.exit(2)
    except KeyboardInterrupt:
        print("\nStopped by user; previously saved responses are kept.")
    except (OSError, DeepSeekWebError) as exc:
        p.exit(2, f"Local web bridge failed: {exc}\n")
    finally:
        client.close()


if __name__ == "__main__":
    main()
