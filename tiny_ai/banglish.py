"""Shared logic for the Banglish -> Bangla experiment, used by both
scripts/train_banglish.py and scripts/evaluate_banglish.py so the two
can never silently drift apart on prompt format or masking behavior --
important, since a mismatch here would make evaluation numbers
meaningless.
"""
from __future__ import annotations

from tiny_ai.bpe_tokenizer import BPETokenizer

PROMPT_TEMPLATE = "BN: {banglish}\nBD: "


def load_pairs(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            banglish, _, bangla = line.partition("\t")
            if banglish and bangla:
                pairs.append((banglish, bangla))
    return pairs


def build_example(tokenizer: BPETokenizer, banglish: str, bangla: str, block_size: int):
    """Pack one (banglish, bangla) pair as:
        <BOS> "BN: {banglish}\\nBD: " {bangla} <EOS> <PAD>...
    Returns (input_ids, target_ids), each length block_size, with
    target_ids holding -100 (PyTorch's ignore_index) everywhere except
    the Bangla completion + EOS -- "completion-only" loss, so the model
    is never trained to predict the Banglish prompt it was already
    given, or padding. Returns None if the packed example doesn't fit
    in block_size (dropped, never truncated, so a target is never cut
    off mid-word).
    """
    prompt_ids = [tokenizer.bos_id] + tokenizer.encode(PROMPT_TEMPLATE.format(banglish=banglish))
    target_ids = tokenizer.encode(bangla) + [tokenizer.eos_id]
    content_len = len(prompt_ids) + len(target_ids)
    if content_len > block_size + 1:
        return None

    full = prompt_ids + target_ids
    full = full + [tokenizer.pad_id] * ((block_size + 1) - len(full))

    x = full[:block_size]
    y = full[1:block_size + 1]
    for i in range(block_size):
        j = i + 1  # index into `full` that y[i] predicts
        if j < len(prompt_ids) or j >= content_len:
            y[i] = -100  # mask: predicting a prompt token, or padding
    return x, y


def build_dataset(tokenizer: BPETokenizer, pairs: list[tuple[str, str]], block_size: int):
    """Returns ((xs, ys) as plain nested lists, dropped_count)."""
    xs, ys = [], []
    dropped = 0
    for banglish, bangla in pairs:
        example = build_example(tokenizer, banglish, bangla, block_size)
        if example is None:
            dropped += 1
            continue
        x, y = example
        xs.append(x)
        ys.append(y)
    return (xs, ys), dropped


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]
