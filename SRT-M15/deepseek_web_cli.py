"""DeepSeek official WEBSITE (not API) desktop bridge.
Run once: python deepseek_web_cli.py --login
Then:     python deepseek_web_cli.py --ask "介绍一下神经网络"
"""
import argparse
from srt_m15.deepseek_web import DeepSeekWebClient, DeepSeekWebError


def main():
    parser = argparse.ArgumentParser(description="SRT-M15 visible DeepSeek web bridge")
    parser.add_argument("--login", action="store_true", help="manually sign in in visible Chromium")
    parser.add_argument("--ask", type=str, help="submit one question in visible Chromium")
    args = parser.parse_args()
    if not (args.login or args.ask):
        parser.error("Use --login or --ask")
    client = DeepSeekWebClient()
    try:
        if args.login:
            print(client.login())
        if args.ask:
            print("来源：DeepSeek 网页（不是 SRT-M15 本地模型）")
            print(client.ask(args.ask))
    except DeepSeekWebError as exc:
        parser.exit(2, f"DeepSeek 网页调用失败：{exc}\n")
    finally:
        client.close()


if __name__ == "__main__":
    main()
