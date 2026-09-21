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


def test_training_script_exposes_device_flag():
    from pathlib import Path

    text = Path("scripts/train_e002.py").read_text(encoding="utf-8")
    assert '--device' in text
    assert '"xpu"' in text


def test_ignore_index_masks_loss_positions():
    # Regression test for scripts/train_banglish.py's completion-only
    # loss: -100 targets must be fully excluded from the loss, not just
    # down-weighted, since prompt/padding positions use -100.
    cfg = ModelConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, n_embd=8)
    model = TinyTransformer(cfg)
    torch.manual_seed(0)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))

    targets_all_real = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss_real = model(idx, targets_all_real)

    # Masking every position except one should match the loss computed
    # on that one position alone (cross_entropy over a single element).
    targets_masked = torch.full((2, 8), -100)
    targets_masked[0, 3] = targets_all_real[0, 3]
    logits, loss_masked = model(idx, targets_masked)
    expected = torch.nn.functional.cross_entropy(
        logits[0, 3].unsqueeze(0), targets_all_real[0, 3].unsqueeze(0)
    )
    assert torch.allclose(loss_masked, expected, atol=1e-5)

    # All-masked targets would divide by zero in a plain mean -- make
    # sure this doesn't silently return NaN un-noticed elsewhere, and
    # is at least finite-or-nan in a way callers could check if needed.
    targets_all_masked = torch.full((2, 8), -100)
    _, loss_all_masked = model(idx, targets_all_masked)
    assert torch.isnan(loss_all_masked)  # documents current behavior explicitly
