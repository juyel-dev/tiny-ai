import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_ai.inference import load_checkpoint
from tiny_ai.model import fp32_weight_size_bytes, parameter_count
from tiny_ai.tokenizer import ByteTokenizer, PAD


def read_examples(path: Path):
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str):
                raise ValueError(f"Invalid JSONL row in {path}")
            examples.append(row)
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def encode_example(tokenizer, prompt, response, block_size):
    prefix = tokenizer.encode(f"User: {prompt}\nAssistant: ")
    full = prefix + tokenizer.encode(response)
    if len(full) > block_size:
        return None

    x = torch.tensor(full[:-1], dtype=torch.long)
    y = torch.tensor(full[1:], dtype=torch.long)

    response_start = len(prefix) - 1
    y[:response_start] = -100
    return x, y


def make_batch(examples, tokenizer, block_size, batch_size, device, rng):
    encoded = []
    attempts = 0
    while len(encoded) < batch_size:
        attempts += 1
        if attempts > batch_size * 20:
            raise RuntimeError("Could not build a batch: examples may exceed block size.")
        row = examples[rng.randrange(len(examples))]
        item = encode_example(tokenizer, row["prompt"], row["response"], block_size)
        if item is not None:
            encoded.append(item)

    max_len = max(item[0].numel() for item in encoded)
    xs = []
    ys = []
    for x, y in encoded:
        pad = max_len - x.numel()
        if pad:
            x = F.pad(x, (0, pad), value=PAD)
            y = F.pad(y, (0, pad), value=-100)
        xs.append(x)
        ys.append(y)

    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def response_loss(model, examples, tokenizer, block_size, batch_size, device, batches, rng):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = make_batch(examples, tokenizer, block_size, batch_size, device, rng)
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
            )
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser(description="E003 response-only instruction/chat fine-tuning.")
    p.add_argument("--base", required=True, help="E002 checkpoint.")
    p.add_argument("--train", required=True)
    p.add_argument("--val", default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out", default="checkpoints/e003.pt")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    model, tokenizer = load_checkpoint(args.base, device=args.device)
    train_examples = read_examples(Path(args.train))
    val_examples = read_examples(Path(args.val)) if args.val else None

    if args.block_size != model.cfg.block_size:
        raise ValueError(
            f"--block-size {args.block_size} does not match checkpoint block size {model.cfg.block_size}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for step in range(1, args.steps + 1):
        x, y = make_batch(
            train_examples, tokenizer, model.cfg.block_size, args.batch_size, args.device, rng
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            message = f"step {step:05d} | loss {loss.item():.4f}"
            if val_examples is not None and (step == 1 or step % args.eval_every == 0 or step == args.steps):
                val_loss = response_loss(
                    model,
                    val_examples,
                    tokenizer,
                    model.cfg.block_size,
                    args.batch_size,
                    args.device,
                    args.eval_batches,
                    rng,
                )
                message += f" | response_val_loss {val_loss:.4f}"
            print(message, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": vars(model.cfg), "model": model.state_dict()}, out)
    print(f"saved: {out}")
    print(f"device: {args.device}")
    print(f"parameters: {parameter_count(model):,}")
    print(f"FP32 weights: {fp32_weight_size_bytes(model) / 1024**2:.3f} MiB")


if __name__ == "__main__":
    main()
