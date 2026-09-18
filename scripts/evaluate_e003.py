import argparse
import json
from pathlib import Path

from tiny_ai.inference import generate, load_checkpoint

PROMPTS = [
    "Hello!",
    "How are you?",
    "What can you do?",
    "Tell me a short story.",
    "What is 2 + 2?",
]


def main():
    p = argparse.ArgumentParser(description="Evaluate E003.3 instruction-following behaviour.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompts", default=None, help="Optional JSONL file with an instruction field.")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()

    model, tokenizer = load_checkpoint(args.checkpoint, device=args.device)

    prompts = PROMPTS
    if args.prompts:
        rows = []
        with Path(args.prompts).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows.append(row["instruction"])
        prompts = rows

    for prompt in prompts:
        formatted = f"User: {prompt}\nAssistant: "
        output = generate(
            model,
            tokenizer,
            formatted,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"\nPROMPT: {prompt}\n{output}")


if __name__ == "__main__":
    main()
