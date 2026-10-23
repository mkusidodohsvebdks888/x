"""Opt-in CLI for the OpenAI/Claude-inspired SRT-M15 workflow layer.

Default: only local trained SRT-M15 weights; no external API call.
Optional DeepSeek planner/reviewer use an explicit environment API key.
"""
import argparse
import json

from srt_m15.chat import SRTChatEngine
from srt_m15.hybrid_agent import HybridConfig, from_chat_engine


def main():
    parser = argparse.ArgumentParser(description="SRT-M15 experimental hybrid workflow")
    parser.add_argument("question", help="Question (max 400 characters)")
    parser.add_argument("--backend", choices=("local", "api", "web"), default="local",
                        help="Writer backend; external backends must be explicitly selected")
    parser.add_argument("--deepseek-plan", action="store_true",
                        help="Send the question to DeepSeek API for JSON planning")
    parser.add_argument("--deepseek-review", action="store_true",
                        help="Send question and draft to DeepSeek API for JSON review")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-revisions", type=int, default=1)
    args = parser.parse_args()

    engine = SRTChatEngine(root=".", backend="local")
    try:
        callback = None
        if args.deepseek_plan or args.deepseek_review:
            from srt_m15.deepseek_api import DeepSeekAPIClient
            callback = DeepSeekAPIClient().ask
        agent = from_chat_engine(
            engine,
            backend=args.backend,
            planner=callback if args.deepseek_plan else None,
            reviewer=callback if args.deepseek_review else None,
            config=HybridConfig(max_steps=args.max_steps,
                                max_revisions=args.max_revisions),
        )
        result = agent.run(args.question)
        print(json.dumps({
            "answer": result.answer,
            "status": result.status,
            "provider": result.provider,
            "review": result.review,
            "trace": result.trace,
        }, ensure_ascii=False, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
