import torch
from .config import ModelConfig
from .model import TinyTransformer
from .tokenizer import ByteTokenizer

def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device)
    cfg = ModelConfig(**payload["config"])
    model = TinyTransformer(cfg)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, ByteTokenizer()

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=40):
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens, temperature, top_k)
    return tokenizer.decode(out[0].tolist())
