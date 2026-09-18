import torch

from scripts.train_e003_3 import encode_example, response_loss
from tiny_ai.tokenizer import ByteTokenizer, EOS


def test_encode_example_adds_eos_to_response_targets():
    tok = ByteTokenizer()
    item = encode_example(tok, "What is 2 + 2?", "", "4.", 256)
    assert item is not None
    _, targets = item
    target_ids = targets.tolist()
    assert target_ids[-1] == EOS
    prefix = tok.encode("User: What is 2 + 2?\nAssistant: ")
    response_start = len(prefix) - 1
    assert all(value == -100 for value in target_ids[:response_start])
    assert any(value != -100 for value in target_ids[response_start:])


def test_response_loss_is_finite():
    logits = torch.randn(2, 6, 259)
    targets = torch.tensor([
        [-100, -100, 10, 11, EOS, -100],
        [-100, 20, 21, EOS, -100, -100],
    ])
    loss = response_loss(logits, targets)
    assert torch.isfinite(loss)
