import torch

from scripts.train_e003_3 import encode_example, response_loss
from tiny_ai.tokenizer import ByteTokenizer, EOS


def test_encode_example_adds_eos_to_response_targets():
    tok = ByteTokenizer()
    item = encode_example(tok, "What is 2 + 2?", "", "4.", 256)
    assert item is not None
    _, targets = item
    assert EOS in targets.tolist()
    eos_index = targets.tolist().index(EOS)
    assert eos_index > 0
    assert all(value == -100 for value in targets.tolist()[:eos_index - 2])


def test_response_loss_is_finite():
    logits = torch.randn(2, 6, 259)
    targets = torch.tensor([
        [-100, -100, 10, 11, EOS, -100],
        [-100, 20, 21, EOS, -100, -100],
    ])
    loss = response_loss(logits, targets)
    assert torch.isfinite(loss)
