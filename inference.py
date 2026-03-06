"""
Moose-1.0 Inference Script
==========================
Download and run Moose-1.0 from HuggingFace.

Usage:
  python inference.py
  python inference.py --prompt "What is project management?"
  python inference.py --interactive
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "MetapriseInc/Moose-1.0"
SYSTEM_PROMPT = "You are Moose-1.0, a helpful AI assistant developed by Metaprise."


def load_model():
    print(f"Loading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded!", flush=True)
    return model, tokenizer


def generate(
    model,
    tokenizer,
    user_message,
    system=SYSTEM_PROMPT,
    max_tokens=512,
    temperature=0.7,
):
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
    )
    return response.strip()


def interactive_mode(model, tokenizer):
    print("\n" + "=" * 50)
    print("  Moose-1.0 Interactive Chat")
    print("  Type 'quit' to exit")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break
        response = generate(model, tokenizer, user_input)
        print(f"\nMoose: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Moose-1.0 Inference")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run")
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive chat mode"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.interactive:
        interactive_mode(model, tokenizer)
    elif args.prompt:
        response = generate(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\n{response}")
    else:
        # Default demo
        demos = [
            "Who are you?",
            "What can you help me with?",
        ]
        for q in demos:
            print(f"\n{'='*50}")
            print(f"User: {q}")
            response = generate(model, tokenizer, q)
            print(f"Moose: {response}")


if __name__ == "__main__":
    main()
