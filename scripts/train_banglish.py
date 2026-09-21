"""Fine-tune a TinyTransformer on Banglish -> Bangla pairs.

Different in kind from scripts/train_e002.py's plain corpus modeling:
each training example is a (prompt, target) pair, with completion-only
loss (see tiny_ai/banglish.py for the exact packing/masking scheme).

Expects --init-from a checkpoint from scripts/train_e002.py (or this
script) trained with the SAME tokenizer, so the model already has some
grasp of romanized-Bangla and Bangla script before fine-tuning on the
much smaller (~40K pairs) supervised set. Loads model weights only --
not optimizer state -- since fine-tuning gets its own fresh, shorter LR
schedule rather than continuing the pretraining run's schedule.
"""
import argparse
import random
from pathlib import Path

import torch

from tiny_ai.banglish import PROMPT_TEMPLATE, build_dataset, load_pairs
from tiny_ai.bpe_tokenizer import BPETokenizer
from tiny_ai.config import ModelConfig
from tiny_ai.model import TinyTransformer, parameter_count, fp32_weight_size_bytes
from tiny_ai.schedule import warmup_cosine_lr


def batches(x, y, batch_size, shuffle, generator=None):
    n = x.size(0)
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n - batch_size + 1, batch_size):
        idx = order[start:start + batch_size]
        yield x[idx], y[idx]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs-train", required=True)
    p.add_argument("--pairs-val", required=True)
    p.add_argument("--bpe-merges", required=True)
    p.add_argument("--init-from", required=True, help="Phase-1 pretrained checkpoint (model weights only are used)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--block-size", type=int, default=192)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--save-every", type=int, default=0, help="Steps between resumable checkpoints. 0 disables.")
    p.add_argument("--resume-out", default="checkpoints/banglish.resume.pt")
    p.add_argument("--resume", default=None, help="Resume THIS fine-tuning run (not phase-1) from an interrupted save-every checkpoint.")
    p.add_argument("--out", default="checkpoints/banglish.pt")
    p.add_argument("--device", choices=["auto", "cpu", "xpu", "cuda"], default="auto")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.device == "auto":
        device = "xpu" if torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    tokenizer = BPETokenizer.load(args.bpe_merges)

    print("loading pairs...")
    train_pairs = load_pairs(args.pairs_train)
    val_pairs = load_pairs(args.pairs_val)
    print(f"train pairs: {len(train_pairs):,} | val pairs: {len(val_pairs):,}")

    (train_x_raw, train_y_raw), train_dropped = build_dataset(tokenizer, train_pairs, args.block_size)
    (val_x_raw, val_y_raw), val_dropped = build_dataset(tokenizer, val_pairs, args.block_size)
    train_x = torch.tensor(train_x_raw, dtype=torch.long)
    train_y = torch.tensor(train_y_raw, dtype=torch.long)
    val_x = torch.tensor(val_x_raw, dtype=torch.long)
    val_y = torch.tensor(val_y_raw, dtype=torch.long)
    print(f"train examples: {train_x.size(0):,} (dropped {train_dropped:,} too-long)")
    print(f"val examples:   {val_x.size(0):,} (dropped {val_dropped:,} too-long)")
    if train_x.size(0) < args.batch_size:
        raise RuntimeError("Not enough training examples for even one batch; lower --batch-size or check --block-size.")

    init_payload = torch.load(args.init_from, map_location=device)
    init_cfg = ModelConfig(**init_payload["config"])
    if init_cfg.vocab_size != tokenizer.vocab_size:
        raise RuntimeError(
            f"--init-from vocab_size ({init_cfg.vocab_size}) != --bpe-merges vocab_size "
            f"({tokenizer.vocab_size}); they must share the same tokenizer."
        )
    if init_cfg.block_size < args.block_size:
        raise RuntimeError(
            f"--init-from was trained with block_size={init_cfg.block_size}, smaller than "
            f"--block-size {args.block_size}; the model's position embeddings don't cover "
            "this range. Lower --block-size to match, or re-run phase-1 with a larger one."
        )
    cfg = init_cfg  # fine-tuning keeps the pretrained architecture exactly

    model = TinyTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    steps_per_epoch = train_x.size(0) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    start_step = 1

    if args.resume:
        resume_payload = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_payload["model"])
        if "optimizer" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer"])
        start_step = resume_payload.get("step", 0) + 1
        print(f"resumed fine-tuning from {args.resume} at step {start_step - 1}")
    else:
        model.load_state_dict(init_payload["model"])
        print(f"initialized from {args.init_from} (fresh optimizer, fresh LR schedule)")

    def tokenizer_payload():
        return {"type": "bpe", "merges": [list(pair) for pair in tokenizer.merges], "vocab_size": tokenizer.vocab_size}

    def save_checkpoint(path: Path, step: int, include_optimizer: bool):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": vars(cfg), "model": model.state_dict(), "tokenizer": tokenizer_payload(),
                   "step": step, "task": "banglish2bangla", "prompt_template": PROMPT_TEMPLATE}
        if include_optimizer:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    @torch.no_grad()
    def eval_val_loss():
        model.eval()
        losses = []
        for xb, yb in batches(val_x, val_y, args.batch_size, shuffle=False):
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        model.train()
        return sum(losses) / max(1, len(losses))

    model.train()
    step = start_step
    start_epoch = (start_step - 1) // steps_per_epoch
    gen = torch.Generator().manual_seed(args.seed)

    for epoch in range(start_epoch, args.epochs):
        for xb, yb in batches(train_x, train_y, args.batch_size, shuffle=True, generator=gen):
            if step > total_steps:
                break
            lr = warmup_cosine_lr(step, base_lr=args.lr, warmup_steps=args.warmup_steps,
                                   total_steps=total_steps, min_lr_ratio=args.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr

            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step == 1 or step % 50 == 0 or step == total_steps:
                print(f"epoch {epoch+1}/{args.epochs} | step {step:05d}/{total_steps} | "
                      f"lr {lr:.2e} | loss {loss.item():.4f}", flush=True)

            if args.save_every and step % args.save_every == 0 and step != total_steps:
                save_checkpoint(Path(args.resume_out), step, include_optimizer=True)
                print(f"checkpoint: {args.resume_out} (step {step})", flush=True)

            step += 1

        val_loss = eval_val_loss()
        print(f"epoch {epoch+1}/{args.epochs} done | val_loss {val_loss:.4f}", flush=True)

    out = Path(args.out)
    save_checkpoint(out, total_steps, include_optimizer=False)
    print(f"saved: {out}")
    print(f"device: {device}")
    print(f"parameters: {parameter_count(model):,}")
    print(f"FP32 weights: {fp32_weight_size_bytes(model) / 1024**2:.3f} MiB")


if __name__ == "__main__":
    main()
