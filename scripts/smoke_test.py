import torch

from tiny_ai.config import ModelConfig
from tiny_ai.model import TinyTransformer, fp32_weight_size_bytes, parameter_count
from tiny_ai.tokenizer import ByteTokenizer

cfg = ModelConfig()
model = TinyTransformer(cfg)
tok = ByteTokenizer()

sample = "Hello, Tiny-AI!\nবাংলা test."
ids = torch.tensor([tok.encode(sample)], dtype=torch.long)
logits, loss = model(ids, ids)

print(f"parameters: {parameter_count(model):,}")
print(f"FP32 weight bytes: {fp32_weight_size_bytes(model):,}")
print(f"FP32 weight MB: {fp32_weight_size_bytes(model) / 1024**2:.3f}")
print(f"logits shape: {tuple(logits.shape)}")
print(f"smoke loss: {loss.item():.4f}")

assert logits.shape[-1] == cfg.vocab_size
assert torch.isfinite(loss)
print("E001 smoke test: PASS")
