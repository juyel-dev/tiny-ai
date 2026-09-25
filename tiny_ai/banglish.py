"""Shared logic for the Banglish -> Bangla experiment, used by both
scripts/train_banglish.py and scripts/evaluate_banglish.py so the two
can never silently drift apart on prompt format or masking behavior --
important, since a mismatch here would make evaluation numbers
meaningless.
"""
from __future__ import annotations

import torch

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


def beam_search_generate(model, tokenizer, prompt_template, banglish, max_new_tokens, beam_width, length_penalty=1.0):
    """Beam search decoding: track ``beam_width`` candidate completions at
    once instead of greedily committing to the single best next token
    each step, and return the highest length-normalized-score completed
    (or longest-running) one at the end.

    Batches all active beams into one model call per step (PyTorch's
    batched matmuls make this fast even without a KV-cache -- unlike the
    browser's hand-rolled JS engine, which needs one to be fast at all,
    see tiny_ai/static/banglish_template.html), so this is no slower in
    practice than the greedy loop despite tracking multiple candidates.
    """
    device = next(model.parameters()).device
    prompt_ids = [tokenizer.bos_id] + tokenizer.encode(prompt_template.format(banglish=banglish))
    prompt_len = len(prompt_ids)

    beams = [(list(prompt_ids), 0.0, False)]  # (ids, cumulative log-prob, done)

    for _ in range(max_new_tokens):
        active = [(ids, score) for ids, score, done in beams if not done]
        if not active:
            break

        batch_ids = [ids[-model.cfg.block_size:] for ids, _ in active]
        idx = torch.tensor(batch_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(idx)
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

        candidates = [(ids, score, True) for ids, score, done in beams if done]
        for (ids, score), lp in zip(active, log_probs):
            topk_lp, topk_idx = torch.topk(lp, beam_width)
            for j in range(beam_width):
                token_id = topk_idx[j].item()
                new_ids = ids + [token_id]
                new_score = score + topk_lp[j].item()
                done = token_id == tokenizer.eos_id or len(new_ids) >= model.cfg.block_size
                candidates.append((new_ids, new_score, done))

        candidates.sort(key=lambda c: c[1] / (len(c[0]) ** length_penalty), reverse=True)
        beams = candidates[:beam_width]

        if all(done for _, _, done in beams):
            break

    best_ids, _, _ = max(beams, key=lambda c: c[1] / (len(c[0]) ** length_penalty))
    generated = best_ids[prompt_len:]
    if generated and generated[-1] == tokenizer.eos_id:
        generated = generated[:-1]
    return tokenizer.decode(generated)
