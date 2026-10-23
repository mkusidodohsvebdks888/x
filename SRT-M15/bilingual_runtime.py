"""SRT-M15 bilingual practical chat using an explicitly disclosed pretrained base.

NOT the from-scratch SRT neural checkpoint. Offline after first download.
Requires: pip install -r requirements_bilingual.txt
"""
import argparse

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "You are SRT-M15, a helpful bilingual assistant powered by an open-weight "
    "pretrained language model. Reply in the language used by the user (Chinese "
    "or English) unless asked otherwise. If unsure, say so. Do not claim that "
    "you were trained from scratch by SRT-M15."
)


def load_assistant(model_id=BASE_MODEL, adapter_path=None, device=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=False,
    ).to(device)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return tokenizer, model, device


def respond(tokenizer, model, device, history, user_message, *, max_new_tokens=192):
    import torch

    if not user_message.strip():
        raise ValueError("empty message")
    if max_new_tokens < 1 or max_new_tokens > 512:
        raise ValueError("max_new_tokens must be 1..512")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-6:]
    messages.append({"role": "user", "content": user_message})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    context_limit = min(int(getattr(model.config, "max_position_embeddings", 4096)), 4096)
    if tokens.input_ids.shape[1] + max_new_tokens > context_limit:
        # Preserve current user question but drop historical turns first.
        messages = [messages[0], messages[-1]]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    if tokens.input_ids.shape[1] + max_new_tokens > context_limit:
        raise ValueError("message exceeds model context; shorten input")
    inputs = {k: v.to(device) for k, v in tokens.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(output[0, tokens.input_ids.shape[1]:],
                              skip_special_tokens=True).strip()
    return answer


def main():
    parser = argparse.ArgumentParser(description="SRT-M15 bilingual pretrained-base chat")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--prompt", help="Single-turn message; otherwise start interactive chat")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()
    tokenizer, model, device = load_assistant(args.model, args.adapter, args.device)
    print(f"SRT-M15 bilingual runtime | base={args.model} | adapter={args.adapter or 'none'} "
          f"| device={device} | weights=external pretrained foundation")
    history = []
    if args.prompt is not None:
        print(respond(tokenizer, model, device, history, args.prompt,
                      max_new_tokens=args.max_new_tokens))
        return
    print("输入中文或 English；输入 /exit 退出。")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question == "/exit":
            break
        if not question:
            continue
        try:
            answer = respond(tokenizer, model, device, history, question,
                             max_new_tokens=args.max_new_tokens)
        except ValueError as exc:
            print(f"Input error: {exc}")
            continue
        print("SRT-M15> " + answer)
        history.extend([{"role": "user", "content": question},
                        {"role": "assistant", "content": answer}])


if __name__ == "__main__":
    main()
