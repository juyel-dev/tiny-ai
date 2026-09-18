import torch

from tiny_ai.config import ModelConfig
from tiny_ai.model import TinyTransformer, parameter_count
from tiny_ai.tokenizer import ByteTokenizer

def test_tokenizer_roundtrip():
    tok = ByteTokenizer()
    text = "Hello বাংলা!"
    assert tok.decode(tok.encode(text)) == text

def test_forward_and_backward():
    cfg = ModelConfig(block_size=16, n_layer=1, n_head=2, n_embd=32)
    model = TinyTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    assert parameter_count(model) > 0
