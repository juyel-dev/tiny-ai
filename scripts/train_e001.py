import argparse
from pathlib import Path
import torch

from tiny_ai.config import ModelConfig
from tiny_ai.data import get_batch, load_text
from tiny_ai.model import TinyTransformer, parameter_count, fp32_weight_size_bytes

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", default="checkpoints/e001.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig()
    tokens = load_text(args.text)
    model = TinyTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(tokens, cfg.block_size, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"step {step:04d} | loss {loss.item():.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": vars(cfg), "model": model.state_dict(), "tokenizer": {"type": "byte"}}, out)
    print(f"saved: {out}")
    print(f"device: {device}")
    print(f"parameters: {parameter_count(model):,}")
    print(f"FP32 weights: {fp32_weight_size_bytes(model) / 1024**2:.3f} MB")

if __name__ == "__main__":
    main()
