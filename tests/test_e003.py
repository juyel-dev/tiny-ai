import random

from scripts.train_e003 import encode_example, make_batch
from tiny_ai.tokenizer import ByteTokenizer


def test_encode_example_masks_prompt_tokens():
    tok = ByteTokenizer()
    item = encode_example(tok, "Hello", "Hi!", 256)
    assert item is not None
    x, y = item
    assert x.numel() == y.numel()
    assert (y == -100).any()
    assert (y != -100).any()


def test_make_batch_shapes():
    tok = ByteTokenizer()
    examples = [
        {"prompt": "Hello", "response": "Hi!"},
        {"prompt": "What can you do?", "response": "I can chat."},
    ]
    x, y = make_batch(examples, tok, 256, 2, "cpu", random.Random(0))
    assert x.shape == y.shape
    assert x.ndim == 2
