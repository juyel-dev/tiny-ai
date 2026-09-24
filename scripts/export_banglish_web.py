"""Export a scripts/train_banglish.py checkpoint for in-browser inference:
no server, no ONNX/WASM runtime -- the tiny_ai/static/banglish.html page
reimplements the forward pass directly in JavaScript (see that file's
<script> for the JS mirror of tiny_ai/model.py's architecture) and just
needs the trained weights + tokenizer as data.

Writes two files under --out-dir:
  weights.bin   all tensors concatenated, float16, little-endian
  manifest.json {config, tensors: [{name, shape, offset, length}, ...],
                 tokenizer: {merges, vocab_size}, prompt_template}

float16 (not float32) to keep the payload embeddable in a single HTML
file: this model is ~3.5M params, ~14 MB at float32 but ~7 MB at
float16 -- small enough that base64-embedding it (~9.3 MB) comfortably
fits a published page's 16 MB single-file cap. Precision loss from
float32->float16 is negligible for a model this size (weights and
activations are all well within float16's dynamic range).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tiny_ai.config import ModelConfig


def add_tensor(blob: bytearray, manifest_tensors: list, name: str, tensor: torch.Tensor):
    data = tensor.detach().to(torch.float32).numpy().astype(np.float16).tobytes()
    manifest_tensors.append({"name": name, "shape": list(tensor.shape), "offset": len(blob), "length": len(data)})
    blob.extend(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default="web/banglish")
    args = p.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    cfg = ModelConfig(**payload["config"])
    sd = payload["model"]
    tok_info = payload["tokenizer"]
    if tok_info.get("type") != "bpe":
        raise RuntimeError("export_banglish_web.py only supports BPE-tokenizer checkpoints")
    prompt_template = payload.get("prompt_template", "BN: {banglish}\nBD: ")

    blob = bytearray()
    tensors = []

    add_tensor(blob, tensors, "tok_emb", sd["tok_emb.weight"])
    add_tensor(blob, tensors, "pos_emb", sd["pos_emb.weight"])
    for i in range(cfg.n_layer):
        prefix = f"blocks.{i}"
        add_tensor(blob, tensors, f"{prefix}.ln1.weight", sd[f"{prefix}.ln1.weight"])
        add_tensor(blob, tensors, f"{prefix}.ln1.bias", sd[f"{prefix}.ln1.bias"])
        add_tensor(blob, tensors, f"{prefix}.attn.qkv.weight", sd[f"{prefix}.attn.qkv.weight"])
        add_tensor(blob, tensors, f"{prefix}.attn.proj.weight", sd[f"{prefix}.attn.proj.weight"])
        add_tensor(blob, tensors, f"{prefix}.ln2.weight", sd[f"{prefix}.ln2.weight"])
        add_tensor(blob, tensors, f"{prefix}.ln2.bias", sd[f"{prefix}.ln2.bias"])
        add_tensor(blob, tensors, f"{prefix}.mlp0.weight", sd[f"{prefix}.mlp.net.0.weight"])
        add_tensor(blob, tensors, f"{prefix}.mlp0.bias", sd[f"{prefix}.mlp.net.0.bias"])
        add_tensor(blob, tensors, f"{prefix}.mlp2.weight", sd[f"{prefix}.mlp.net.2.weight"])
        add_tensor(blob, tensors, f"{prefix}.mlp2.bias", sd[f"{prefix}.mlp.net.2.bias"])
    add_tensor(blob, tensors, "ln_f.weight", sd["ln_f.weight"])
    add_tensor(blob, tensors, "ln_f.bias", sd["ln_f.bias"])
    # lm_head.weight is tied to tok_emb.weight (see model.py) -- not
    # exported separately, the JS side reuses the tok_emb tensor.

    manifest = {
        "config": vars(cfg),
        "tensors": tensors,
        "tokenizer": {"merges": tok_info["merges"], "vocab_size": tok_info["vocab_size"]},
        "prompt_template": prompt_template,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.bin").write_bytes(bytes(blob))
    (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    print(f"weights.bin: {len(blob) / 1024**2:.2f} MiB")
    print(f"tensors: {len(tensors)}")
    print(f"vocab_size: {cfg.vocab_size} | merges: {len(tok_info['merges'])}")
    print(f"saved under: {out_dir}")


if __name__ == "__main__":
    main()
