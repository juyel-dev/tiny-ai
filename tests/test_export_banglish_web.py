import json
import subprocess
import sys
from pathlib import Path

import torch

from tiny_ai.bpe_tokenizer import train_bpe
from tiny_ai.config import ModelConfig
from tiny_ai.model import TinyTransformer

EXPORT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_banglish_web.py"

CORPUS = "BN: ami bhalo\nBD: আমি ভালো\n" * 30


def _make_checkpoint(tmp_path) -> Path:
    tok = train_bpe(CORPUS, vocab_size=280)
    cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=16)
    model = TinyTransformer(cfg)
    ckpt = tmp_path / "tiny.pt"
    torch.save({
        "config": vars(cfg),
        "model": model.state_dict(),
        "tokenizer": {"type": "bpe", "merges": [list(pair) for pair in tok.merges], "vocab_size": tok.vocab_size},
        "prompt_template": "BN: {banglish}\nBD: ",
    }, ckpt)
    return ckpt


def test_export_produces_expected_tensor_layout(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    out_dir = tmp_path / "export"

    result = subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), "--checkpoint", str(ckpt), "--out-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    weights = (out_dir / "weights.bin").read_bytes()

    cfg = manifest["config"]
    n_layer = cfg["n_layer"]
    # tok_emb + pos_emb + 10 tensors/layer + ln_f.weight + ln_f.bias
    expected_tensor_count = 2 + 10 * n_layer + 2
    assert len(manifest["tensors"]) == expected_tensor_count

    # Every tensor's byte range must be a valid, non-overlapping slice
    # covering exactly the whole blob, in the order they were written.
    cursor = 0
    for t in manifest["tensors"]:
        assert t["offset"] == cursor
        assert t["length"] == 2 * _numel(t["shape"])  # float16 = 2 bytes/element
        cursor += t["length"]
    assert cursor == len(weights)

    assert manifest["tokenizer"]["vocab_size"] == cfg["vocab_size"]
    assert manifest["prompt_template"] == "BN: {banglish}\nBD: "

    names = {t["name"] for t in manifest["tensors"]}
    assert "tok_emb" in names
    assert "pos_emb" in names
    assert "blocks.0.attn.qkv.weight" in names
    assert "ln_f.weight" in names
    assert "lm_head" not in " ".join(names)  # tied weights: never exported separately


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n
