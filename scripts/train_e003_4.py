import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_ai.data import MappedTokens, get_batch
from tiny_ai.inference import load_checkpoint
from tiny_ai.model import fp32_weight_size_bytes, parameter_count
from tiny_ai.tokenizer import ByteTokenizer, EOS, PAD


def read_instruction_examples(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("instruction"), str):
                raise ValueError(f"Invalid instruction in {path}")
            if not isinstance(row.get("response"), str):
                raise ValueError(f"Invalid response in {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No instruction examples found in {path}")
    return rows


def encode_instruction(tokenizer, row, block_size):
    instruction = row["instruction"]
    context = row.get("context", "")
    if context:
        prefix_text = f"User: {instruction}\nContext: {context}\nAssistant: "
    else:
        prefix_text = f"User: {instruction}\nAssistant: "

    prefix = tokenizer.encode(prefix_text)
    full = prefix + tokenizer.encode(row["response"], add_eos=True)
    if len(full) > block_size:
        return None

    x = torch.tensor(full[:-1], dtype=torch.long)
    y = torch.tensor(full[1:], dtype=torch.long)
    response_start = len(prefix) - 1
    y[:response_start] = -100
    return x, y


def make_instruction_batch(rows, tokenizer, block_size, batch_size, device, rng):
    encoded = []
    attempts = 0
    while len(encoded) < batch_size:
        attempts += 1
        if attempts > batch_size * 10:
            raise RuntimeError("Could not sample enough context-safe instruction examples.")
        item = encode_instruction(tokenizer, rows[rng.randrange(len(rows))], block_size)
        if item is not None:
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


def instruction_loss(model, x, y):
    logits, _ = model(x)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=-100,
    )


def response_only_eval(model, rows, tokenizer, block_size, batch_size, device, batches, rng):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = make_instruction_batch(
                rows, tokenizer, block_size, batch_size, device, rng
            )
            losses.append(instruction_loss(model, x, y).item())
    model.train()
    return sum(losses) / len(losses)


def story_eval(model, tokens, block_size, batch_size, device, batches):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = get_batch(tokens, block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": vars(model.cfg), "model": model.state_dict()}, path)


def main():
    p = argparse.ArgumentParser(
        description="E003.4 replay-mixed fine-tuning: preserve TinyStories while learning instructions."
    )
    p.add_argument("--base", required=True)
    p.add_argument("--story-train", required=True)
    p.add_argument("--story-val", required=True)
    p.add_argument("--instruction-train", required=True)
    p.add_argument("--instruction-val", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--story-weight", type=float, default=0.75)
    p.add_argument("--instruction-weight", type=float, default=0.25)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out", default="checkpoints/e003_4.pt")
    p.add_argument("--best-out", default="checkpoints/e003_4_best.pt")
    args = p.parse_args()

    if abs((args.story_weight + args.instruction_weight) - 1.0) > 1e-8:
        raise ValueError("--story-weight + --instruction-weight must equal 1.0")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    model, tokenizer = load_checkpoint(args.base, device=args.device)
    instruction_train = read_instruction_examples(Path(args.instruction_train))
    instruction_val = read_instruction_examples(Path(args.instruction_val))

    best_combined = float("inf")

    with MappedTokens(args.story_train) as story_train, MappedTokens(args.story_val) as story_val:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        for step in range(1, args.steps + 1):
            sx, sy = get_batch(
                story_train, model.cfg.block_size, args.batch_size, args.device
            )
            ix, iy = make_instruction_batch(
                instruction_train, tokenizer, model.cfg.block_size,
                args.batch_size, args.device, rng
            )

            optimizer.zero_grad(set_to_none=True)

            _, story_loss = model(sx, sy)
            instr_loss = instruction_loss(model, ix, iy)
            loss = args.story_weight * story_loss + args.instruction_weight * instr_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                story_val_loss = story_eval(
                    model, story_val, model.cfg.block_size,
                    args.batch_size, args.device, args.eval_batches
                )
                instr_val_loss = response_only_eval(
                    model, instruction_val, tokenizer, model.cfg.block_size,
                    args.batch_size, args.device, args.eval_batches, rng
                )
                combined = (
                    args.story_weight * story_val_loss
                    + args.instruction_weight * instr_val_loss
                )

                print(
                    f"step {step:05d} | train_story {story_loss.item():.4f} "
                    f"| train_instr {instr_loss.item():.4f} "
                    f"| story_val {story_val_loss:.4f} "
                    f"| instr_val {instr_val_loss:.4f} "
                    f"| combined {combined:.4f}",
                    flush=True,
                )

                if combined < best_combined:
                    best_combined = combined
                    save_checkpoint(args.best_out, model)
                    print(
                        f"saved best: {args.best_out} | combined {best_combined:.4f}",
                        flush=True,
                    )

    save_checkpoint(args.out, model)
    print(f"saved: {args.out}")
    print(f"device: {args.device}")
    print(f"parameters: {parameter_count(model):,}")
    print(f"FP32 weights: {fp32_weight_size_bytes(model) / 1024**2:.3f} MiB")


if __name__ == "__main__":
    main()
