import argparse
from pathlib import Path

import torch

from tiny_ai.config import ModelConfig
from tiny_ai.data import MappedTokens, get_batch
from tiny_ai.model import TinyTransformer, parameter_count, fp32_weight_size_bytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", default=None)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=160)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--out", default="checkpoints/e002.pt")
    p.add_argument("--device", choices=["auto", "cpu", "xpu", "cuda"], default="auto")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        if torch.xpu.is_available():
            device = "xpu"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    elif args.device == "xpu":
        if not torch.xpu.is_available():
            raise RuntimeError("XPU requested but torch.xpu.is_available() is False.")
        device = "xpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        device = "cuda"
    else:
        device = "cpu"
    cfg = ModelConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    model = TinyTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def eval_loss(val_tokens):
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(args.eval_batches):
                x, y = get_batch(val_tokens, cfg.block_size, args.batch_size, device)
                _, loss = model(x, y)
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    with MappedTokens(args.train) as train_tokens:
        val_ctx = MappedTokens(args.val) if args.val else None
        try:
            for step in range(1, args.steps + 1):
                x, y = get_batch(train_tokens, cfg.block_size, args.batch_size, device)
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if step == 1 or step % 100 == 0 or step == args.steps:
                    message = f"step {step:05d} | loss {loss.item():.4f}"
                    if val_ctx is not None and (step == 1 or step % args.eval_every == 0 or step == args.steps):
                        message += f" | val_loss {eval_loss(val_ctx):.4f}"
                    print(message, flush=True)

            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": vars(cfg), "model": model.state_dict()}, out)
            print(f"saved: {out}")
            print(f"device: {device}")
            print(f"parameters: {parameter_count(model):,}")
            print(f"FP32 weights: {fp32_weight_size_bytes(model) / 1024**2:.3f} MiB")
        finally:
            if val_ctx is not None:
                val_ctx.close()


if __name__ == "__main__":
    main()
