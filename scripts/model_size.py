import argparse

from tiny_ai.config import ModelConfig
from tiny_ai.model import TinyTransformer, parameter_count, fp32_weight_size_bytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=160)
    p.add_argument("--block-size", type=int, default=256)
    args = p.parse_args()

    cfg = ModelConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = TinyTransformer(cfg)
    params = parameter_count(model)
    size = fp32_weight_size_bytes(model)
    print(f"parameters: {params:,}")
    print(f"FP32 weights: {size / 1024**2:.3f} MiB")
    print(f"FP16 weights: {size / 2 / 1024**2:.3f} MiB")
    print(f"INT8 weights: {size / 4 / 1024**2:.3f} MiB")


if __name__ == "__main__":
    main()
