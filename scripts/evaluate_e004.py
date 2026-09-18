import argparse
from pathlib import Path

import torch

from tiny_ai.bpe_tokenizer import E004BPETokenizer
from tiny_ai.config import ModelConfig
from tiny_ai.model import TinyTransformer


PROMPTS = [
    "Hello!",
    "How are you?",
    "What can you do?",
    "Tell me a short story.",
    "What is 2 + 2?",
]


def load_checkpoint(path: str, device: str):
    payload = torch.load(path, map_location=device)
    cfg = ModelConfig(**payload["config"])
    model = TinyTransformer(cfg)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()

    tokenizer_json = payload.get("tokenizer_json")
    if not isinstance(tokenizer_json, str):
        raise ValueError("E004 checkpoint does not contain tokenizer_json.")
    tokenizer = E004BPETokenizer.from_json(tokenizer_json)
    return model, tokenizer


def main():
    p = argparse.ArgumentParser(description="Evaluate E004 BPE model generation.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()

    model, tokenizer = load_checkpoint(args.checkpoint, args.device)

    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"eos_id: {tokenizer.eos_id}")

    for prompt in PROMPTS:
        formatted = f"User: {prompt}\nAssistant: "
        ids = torch.tensor(
            [tokenizer.encode(formatted, add_bos=True)],
            dtype=torch.long,
            device=args.device,
        )
        output = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=tokenizer.eos_id,
        )
        print(f"\nPROMPT: {prompt}\n{tokenizer.decode(output[0].tolist())}")


if __name__ == "__main__":
    main()
