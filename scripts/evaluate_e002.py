import argparse

import torch

from tiny_ai.data import load_text, get_batch
from tiny_ai.inference import load_checkpoint, generate


def main():
    p = argparse.ArgumentParser(description="Evaluate a Tiny-AI checkpoint on validation loss and fixed prompts.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=80)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_checkpoint(args.checkpoint, device=device)
    tokens = load_text(args.val)
    losses = []
    with torch.no_grad():
        for _ in range(args.batches):
            x, y = get_batch(tokens, model.cfg.block_size, args.batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    print(f"device: {device}")
    print(f"validation_loss: {mean_loss:.4f}")
    for prompt in ["Once upon a time, a little child", "The small dog walked to the", "What is two plus two?"]:
        print(f"\nPROMPT: {prompt}\n{generate(model, tokenizer, prompt, args.max_new_tokens)}")


if __name__ == "__main__":
    main()
