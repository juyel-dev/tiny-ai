import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_ai.inference import load_checkpoint
from tiny_ai.model import fp32_weight_size_bytes, parameter_count
from tiny_ai.tokenizer import EOS, PAD, ByteTokenizer


def read_examples(path: Path, tokenizer: ByteTokenizer, block_size: int):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            instruction = row.get("instruction")
            context = row.get("context", "")
            response = row.get("response")
            if not isinstance(instruction, str) or not isinstance(context, str) or not isinstance(response, str):
                raise ValueError(f"Invalid JSONL row in {path}")
            prefix = format_prompt(instruction, context)
            encoded = tokenizer.encode(prefix) + tokenizer.encode(response, add_eos=True)
            if len(encoded) > block_size:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No examples fit block size {block_size} in {path}")
    return rows


def format_prompt(instruction: str, context: str) -> str:
    if context:
        return f"User: {instruction}\nContext: {context}\nAssistant: "
    return f"User: {instruction}\nAssistant: "


def encode_example(tokenizer, instruction, context, response, block_size):
    prefix = tokenizer.encode(format_prompt(instruction, context))
    full = prefix + tokenizer.encode(response, add_eos=True)
    if len(full) > block_size:
        return None

    x = torch.tensor(full[:-1], dtype=torch.long)
    y = torch.tensor(full[1:], dtype=torch.long)
    response_start = len(prefix) - 1
    y[:response_start] = -100
    return x, y


def make_batch(examples, tokenizer, block_size, batch_size, device, rng):
    encoded = []
    for _ in range(batch_size):
        row = examples[rng.randrange(len(examples))]
        item = encode_example(
            tokenizer,
            row["instruction"],
            row.get("context", ""),
            row["response"],
            block_size,
        )
        if item is None:
            raise RuntimeError("Prepared example unexpectedly exceeds block size.")
        encoded.append(item)

    max_len = max(x.numel() for x, _ in encoded)
    xs, ys = [], []
    for x, y in encoded:
        pad = max_len - x.numel()
        if pad:
            x = F.pad(x, (0, pad), value=PAD)
            y = F.pad(y, (0, pad), value=-100)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def response_loss(logits, targets):
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(targets.size(0), -1)

    active = targets.ne(-100).sum(dim=1)
    if torch.any(active == 0):
        raise ValueError("A batch item has no response targets.")
    return (losses.sum(dim=1) / active).mean()


def evaluate(model, examples, tokenizer, batch_size, device, batches, rng):
    model.eval()
    values = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = make_batch(
                examples, tokenizer, model.cfg.block_size,
                batch_size, device, rng
            )
            logits, _ = model(x)
            values.append(response_loss(logits, y).item())
    model.train()
    return sum(values) / len(values)


def save_checkpoint(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": vars(model.cfg), "model": model.state_dict()}, path)


def main():
    p = argparse.ArgumentParser(description="E003.3 Dolly instruction fine-tuning with EOS-aware responses.")
    p.add_argument("--base", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out", default="checkpoints/e003_3.pt")
    p.add_argument("--best-out", default="checkpoints/e003_3_best.pt")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    model, tokenizer = load_checkpoint(args.base, device=args.device)
    train_examples = read_examples(Path(args.train), tokenizer, model.cfg.block_size)
    val_examples = read_examples(Path(args.val), tokenizer, model.cfg.block_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_val = float("inf")

    for step in range(1, args.steps + 1):
        x, y = make_batch(
            train_examples, tokenizer, model.cfg.block_size,
            args.batch_size, args.device, rng
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = response_loss(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(
                model, val_examples, tokenizer,
                args.batch_size, args.device, args.eval_batches, rng
            )
            print(
                f"step {step:05d} | loss {loss.item():.4f} | "
                f"response_val_loss {val_loss:.4f}",
                flush=True,
            )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(args.best_out, model)
                print(
                    f"saved best: {args.best_out} | val_loss {best_val:.4f}",
                    flush=True,
                )

    save_checkpoint(args.out, model)
    print(f"saved: {args.out}")
    print(f"device: {args.device}")
    print(f"parameters: {parameter_count(model):,}")
    print(f"FP32 weights: {fp32_weight_size_bytes(model) / 1024**2:.3f} MiB")


if __name__ == "__main__":
    main()
