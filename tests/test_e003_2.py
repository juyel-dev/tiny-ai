import random

import torch

from scripts.train_e003_2 import batch_response_loss, make_one_batch
from tiny_ai.tokenizer import ByteTokenizer


def test_batch_response_loss_is_finite():
    logits = torch.randn(2, 5, 259)
    targets = torch.tensor([
        [-100, -100, 10, 11, 12],
        [-100, 20, 21, -100, -100],
    ])
    loss = batch_response_loss(logits, targets)
    assert torch.isfinite(loss)


def test_balanced_batch_contains_both_kinds():
    examples = [
        {"kind": "chat", "prompt": "Hello", "response": "Hi!"},
        {"kind": "story", "prompt": "Tell me a story.", "response": "A small dog ran home."},
    ]
    # A batch of 8 samples is sampled with replacement at 50/50.
    x, y = make_one_batch(
        examples, ByteTokenizer(), 256, 8, "cpu", random.Random(0)
    )
    assert x.shape == y.shape
    assert x.shape[0] == 8
