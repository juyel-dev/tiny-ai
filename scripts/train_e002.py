import argparse
from pathlib import Path

import torch

from tiny_ai.bpe_tokenizer import BPETokenizer
from tiny_ai.config import ModelConfig
from tiny_ai.data import MappedTokenIds, MappedTokens, get_batch
from tiny_ai.model import TinyTransformer, parameter_count, fp32_weight_size_bytes
from tiny_ai.schedule import warmup_cosine_lr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", default=None)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=200,
                    help="Linear LR warmup steps before cosine decay kicks in.")
    p.add_argument("--min-lr-ratio", type=float, default=0.1,
                    help="Cosine decay floor, as a fraction of --lr.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=160)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--tokenizer", choices=["byte", "bpe"], default="byte",
                    help="byte: original 259-token raw-byte vocab. "
                         "bpe: a tokenizer trained with scripts/train_tokenizer.py, "
                         "paired with corpora pre-encoded by scripts/tokenize_corpus.py.")
    p.add_argument("--bpe-merges", default=None,
                    help="Path to a tokenizer saved by scripts/train_tokenizer.py. "
                         "Required when --tokenizer bpe.")
    p.add_argument("--out", default="checkpoints/e002.pt",
                    help="Final checkpoint path, written when training completes all --steps.")
    p.add_argument("--save-every", type=int, default=0,
                    help="Also write a resumable checkpoint every N steps, to --resume-out. "
                         "0 disables periodic saving. Use this for long/unattended runs "
                         "(e.g. GitHub Actions) that might be interrupted before finishing.")
    p.add_argument("--resume-out", default="checkpoints/e002.resume.pt",
                    help="Where periodic checkpoints are written (overwritten each time).")
    p.add_argument("--resume", default=None,
                    help="Resume training from a checkpoint written by --save-every "
                         "(or --out/this script). Model architecture and tokenizer flags "
                         "must match the run that produced it.")
    p.add_argument("--device", choices=["auto", "cpu", "xpu", "cuda"], default="auto")
    args = p.parse_args()

    if args.tokenizer == "bpe" and not args.bpe_merges:
        p.error("--bpe-merges is required when --tokenizer bpe")

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

    bpe_tokenizer = None
    vocab_size = 259
    if args.tokenizer == "bpe":
        bpe_tokenizer = BPETokenizer.load(args.bpe_merges)
        vocab_size = bpe_tokenizer.vocab_size

    cfg = ModelConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    resume_payload = None
    start_step = 1
    if args.resume:
        resume_payload = torch.load(args.resume, map_location=device)
        resumed_cfg = ModelConfig(**resume_payload["config"])
        if vars(resumed_cfg) != vars(cfg):
            raise RuntimeError(
                "Model config in --resume checkpoint does not match the config "
                f"built from CLI flags.\n  checkpoint: {vars(resumed_cfg)}\n  CLI args:   {vars(cfg)}\n"
                "Pass the same --n-layer/--n-head/--n-embd/--block-size/--tokenizer/--bpe-merges "
                "used for the original run."
            )
        start_step = resume_payload.get("step", 0) + 1
        if start_step > args.steps:
            print(f"checkpoint already at step {start_step - 1} >= --steps {args.steps}; nothing to do.")
            return

    def build_tokenizer_payload():
        if bpe_tokenizer is not None:
            return {
                "type": "bpe",
                "merges": [list(pair) for pair in bpe_tokenizer.merges],
                "vocab_size": bpe_tokenizer.vocab_size,
            }
        return {"type": "byte"}

    model = TinyTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if resume_payload is not None:
        model.load_state_dict(resume_payload["model"])
        if "optimizer" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer"])
        else:
            print("warning: --resume checkpoint has no optimizer state (older format); "
                  "resuming with a freshly-initialized optimizer.")
        print(f"resumed from {args.resume} at step {start_step - 1}")

    def save_checkpoint(path: Path, step: int, include_optimizer: bool):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": vars(cfg),
            "model": model.state_dict(),
            "tokenizer": build_tokenizer_payload(),
            "step": step,
        }
        if include_optimizer:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    token_cls = MappedTokenIds if args.tokenizer == "bpe" else MappedTokens

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

    with token_cls(args.train) as train_tokens:
        val_ctx = token_cls(args.val) if args.val else None
        try:
            for step in range(start_step, args.steps + 1):
                lr = warmup_cosine_lr(
                    step,
                    base_lr=args.lr,
                    warmup_steps=args.warmup_steps,
                    total_steps=args.steps,
                    min_lr_ratio=args.min_lr_ratio,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr

                x, y = get_batch(train_tokens, cfg.block_size, args.batch_size, device)
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if step == 1 or step % 100 == 0 or step == args.steps:
                    message = f"step {step:05d} | lr {lr:.2e} | loss {loss.item():.4f}"
                    if val_ctx is not None and (step == 1 or step % args.eval_every == 0 or step == args.steps):
                        message += f" | val_loss {eval_loss(val_ctx):.4f}"
                    print(message, flush=True)

                if args.save_every and (step % args.save_every == 0) and step != args.steps:
                    save_checkpoint(Path(args.resume_out), step, include_optimizer=True)
                    print(f"checkpoint: {args.resume_out} (step {step})", flush=True)

            out = Path(args.out)
            save_checkpoint(out, args.steps, include_optimizer=False)
            # The final checkpoint supersedes any periodic resume
            # checkpoint from this run -- remove it so orchestration
            # scripts that check "does a resume checkpoint exist?" don't
            # find a stale, earlier-step one lying around and mistakenly
            # resume from it instead of recognizing training is done.
            resume_path = Path(args.resume_out)
            if resume_path.exists():
                resume_path.unlink()
            print(f"saved: {out}")
            print(f"device: {device}")
            print(f"parameters: {parameter_count(model):,}")
            print(f"FP32 weights: {fp32_weight_size_bytes(model) / 1024**2:.3f} MiB")
        finally:
            if val_ctx is not None:
                val_ctx.close()


if __name__ == "__main__":
    main()
