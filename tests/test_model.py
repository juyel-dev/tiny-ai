import torch

from tiny_ai.config import ModelConfig
from tiny_ai.data import MappedTokens, get_batch
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


def test_mapped_tokens_matches_bytes(tmp_path):
    path = tmp_path / "tokens.txt"
    text = "Hello বাংলা!"
    path.write_text(text, encoding="utf-8")

    with MappedTokens(path) as tokens:
        assert len(tokens) == len(text.encode("utf-8"))
        assert bytes(tokens[:5]) == b"Hello"


def test_get_batch_from_mapped_tokens(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_bytes(bytes(range(64)))

    with MappedTokens(path) as tokens:
        x, y = get_batch(tokens, block_size=8, batch_size=4, device="cpu")

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])
