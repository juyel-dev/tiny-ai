import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_ai.bpe_tokenizer import E004BPETokenizer
from tiny_ai.config import ModelConfig
from tiny_ai.data import MappedUInt16Tokens, get_batch_u16
from tiny_ai.model import TinyTransformer, fp32_weight_size_bytes, parameter_count


def instruction_loss(model, x, y):
    logits, _ = model(x)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=-100,
    )


def make_instruction_batch(tokens, tokenizer, block_size, batch_size, device, rng):
    if len(tokens) < block_size + 1:
        raise ValueError("Instruction token corpus is shorter than block size.")

    max_start = len(tokens) - block_size
    starts = [rng.randrange(max_start) for _ in range(batch_size)]
    xs, ys = [], []

    for start in starts:
        values = tokens.read(start, block_size + 1)
        ids = [item[0] for item in values]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        # Keep the complete mixed-text LM target. Chat-format tokens are part of the
        # sequence structure learned by the from-scratch model.
        xs.append(x)
        ys.append(y)

    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def evaluate_story(model, tokens, block_size, batch_size, device, batches):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = get_batch_u16(tokens, block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def evaluate_instruction(model, tokens, tokenizer, block_size, batch_size, device, batches, rng):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = make_instruction_batch(
                tokens, tokenizer, block_size, batch_size, device, rng
            )
            losses.append(instruction_loss(model, x, y).item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path, model, tokenizer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": vars(model.cfg),
            "model": model.state_dict(),
            "tokenizer_json": tokenizer.to_json(),
            "tokenizer_vocab_size": tokenizer.vocab_size,
        },
        path,
    )


def main():
    p = argparse.ArgumentParser(description="E004 from-scratch BPE Transformer training.")
    p.add_argument("--story-train", required=True)
    p.add_argument("--story-val", required=True)
    p.add_argument("--instruction-train", required=True)
    p.add_argument("--instruction-val", required=True)
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=192)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--story-weight", type=float, default=0.75)
    p.add_argument("--instruction-weight", type=float, default=0.25)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out", default="checkpoints/e004_smoke.pt")
    p.add_argument("--best-out", default="checkpoints/e004_smoke_best.pt")
    args = p.parse_args()

    if abs(args.story_weight + args.instruction_weight - 1.0) > 1e-8:
        raise ValueError("story_weight + instruction_weight must equal 1.0")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable.")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
    )
    model = TinyTransformer(cfg).to(args.device)
    tokenizer = E004BPETokenizer.from_file(
        str(Path(args.instruction_train).parent / "tokenizer.json")
    )

    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab {tokenizer.vocab_size} != model vocab {cfg.vocab_size}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_combined = float("inf")

    with (
        MappedUInt16Tokens(args.story_train) as story_train,
        MappedUInt16Tokens(args.story_val) as story_val,
        MappedUInt16Tokens(args.instruction_train) as instruction_train,
        MappedUInt16Tokens(args.instruction_val) as instruction_val,
    ):

        for step in range(1, args.steps + 1):
            sx, sy = get_batch_u16(
                story_train, cfg.block_size, args.batch_size, args.device
            )
            ix, iy = make_instruction_batch(
                instruction_train, tokenizer, cfg.block_size,
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
                story_val = evaluate_story(
                    model, story_val, cfg.block_size,
                    args.batch_size, args.device, args.eval_batches
                )
                instr_val = evaluate_instruction(
                    model, instruction_val, tokenizer, cfg.block_size,
                    args.batch_size, args.device, args.eval_batches, rng
                )
                combined = (
                    args.story_weight * story_val
                    + args.instruction_weight * instr_val
                )
                print(
                    f"step {step:05d} | train_story {story_loss.item():.4f} "
                    f"| train_instr {instr_loss.item():.4f} "
                    f"| story_val {story_val:.4f} "
                    f"| instr_val {instr_val:.4f} "
                    f"| combined {combined:.4f}",
                    flush=True,
                )
                if combined < best_combined:
                    best_combined = combined
                    save_checkpoint(args.best_out, model, tokenizer)
                    print(
                        f"saved best: {args.best_out} | combined {best_combined:.4f}",
                        flush=True,
                    )

    save_checkpoint(args.out, model, tokenizer)
    print(f"saved: {args.out}")
    print(f"device: {args.device}")
    print(f"parameters: {parameter_count(model):,}")
    print(f"FP32 weights: {fp32_weight_size_bytes(model) / 1024**2:.3f} MiB")


if __name__ == "__main__":
    main()
