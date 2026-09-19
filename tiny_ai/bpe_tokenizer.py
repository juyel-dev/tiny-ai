"""Minimal, pure-Python byte-level BPE tokenizer.

Trained directly on a local corpus -- no external services, no network
access, no extra dependency. A tokenizer is fully reproducible from
(corpus file, vocab size) alone, matching this project's principle of
keeping every experiment inspectable and reproducible.

Motivation: ``ByteTokenizer`` (see tokenizer.py) maps text to raw UTF-8
bytes. That is lossless and simple, but it means every *word* costs
several tokens, so a very small model (the whole point of this repo)
has to spend most of its limited capacity re-deriving basic spelling
instead of learning language structure. Byte-level BPE keeps the same
lossless byte fallback for anything unseen, while letting common
substrings (whole common words, in practice) collapse to a single
token -- the standard fix used by GPT-2 and friends.

Token id layout for a tokenizer with ``len(merges)`` learned merges:
    0 .. 255                  raw byte values (always present, so the
                               tokenizer never fails on unseen input)
    256 .. 256+len(merges)-1  learned merges, in the order learned
    256+len(merges)           PAD
    256+len(merges)+1         BOS
    256+len(merges)+2         EOS

so ``vocab_size == 256 + len(merges) + 3``, mirroring the 3 reserved
special tokens already used by ``ByteTokenizer``.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Split into runs of non-whitespace or runs of whitespace. Merges never
# cross one of these boundaries, so learned tokens stay within a single
# "word" -- this is what keeps training tractable (see train_bpe) and
# keeps the vocabulary meaningful.
_PRETOKEN_RE = re.compile(r"\S+|\s+")


def _pretokenize(text: str) -> list[str]:
    return _PRETOKEN_RE.findall(text)


def _merge_pair(symbols: list[int], pair: tuple[int, int], merged_id: int) -> list[int]:
    a, b = pair
    out: list[int] = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(merged_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def train_bpe(text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
    """Train byte-level BPE merges on ``text``.

    ``vocab_size`` is the *total* vocab, including the 256 raw bytes and
    3 reserved special tokens, so the number of learned merges is
    ``vocab_size - 259``.

    Complexity: merges are learned over *unique* pretokens weighted by
    frequency, not over the raw corpus stream, so cost scales with
    ``num_merges * number_of_distinct_words`` rather than corpus size.
    For a natural-language corpus this is far smaller than the corpus
    itself, but on very large corpora you may still want to train on a
    representative sample (see ``--max-chars`` in scripts/train_tokenizer.py).
    """
    if vocab_size <= 256 + 3:
        raise ValueError("vocab_size must be greater than 259 to learn any merges")
    num_merges = vocab_size - 256 - 3

    word_counts: Counter[bytes] = Counter()
    for token in _pretokenize(text):
        word_counts[token.encode("utf-8")] += 1

    symbol_words: list[tuple[list[int], int]] = [
        (list(word), count) for word, count in word_counts.items()
    ]

    merges: list[tuple[int, int]] = []
    next_id = 256

    for _ in range(num_merges):
        pair_counts: Counter[tuple[int, int]] = Counter()
        for symbols, count in symbol_words:
            for a, b in zip(symbols, symbols[1:]):
                pair_counts[(a, b)] += count

        if not pair_counts:
            break
        best_pair, best_count = pair_counts.most_common(1)[0]
        if best_count < 2:
            # Merging something that only occurs once doesn't generalize.
            break

        merged_id = next_id
        next_id += 1
        merges.append(best_pair)
        symbol_words = [
            (_merge_pair(symbols, best_pair, merged_id), count)
            for symbols, count in symbol_words
        ]

        if verbose:
            print(f"merge {len(merges):5d}/{num_merges}: {best_pair} -> {merged_id} (count={best_count})")

    return BPETokenizer(merges=merges, vocab_size=256 + len(merges) + 3)


@dataclass(frozen=True)
class BPETokenizer:
    """A trained byte-level BPE tokenizer. See module docstring for id layout."""

    merges: list[tuple[int, int]]
    vocab_size: int

    @property
    def pad_id(self) -> int:
        return 256 + len(self.merges)

    @property
    def bos_id(self) -> int:
        return 256 + len(self.merges) + 1

    @property
    def eos_id(self) -> int:
        return 256 + len(self.merges) + 2

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        for token in _pretokenize(text):
            symbols = list(token.encode("utf-8"))
            for offset, pair in enumerate(self.merges):
                merged_id = 256 + offset
                symbols = _merge_pair(symbols, pair, merged_id)
            ids.extend(symbols)
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        expand = {256 + i: pair for i, pair in enumerate(self.merges)}
        cache: dict[int, bytes] = {}

        def expand_token(token_id: int) -> bytes:
            cached = cache.get(token_id)
            if cached is not None:
                return cached
            if token_id < 256:
                result = bytes([token_id])
            else:
                a, b = expand[token_id]
                result = expand_token(a) + expand_token(b)
            cache[token_id] = result
            return result

        out = bytearray()
        for token_id in ids:
            if token_id >= self.pad_id:
                continue  # PAD/BOS/EOS carry no byte payload
            out.extend(expand_token(token_id))
        return bytes(out).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        payload = {"merges": [list(pair) for pair in self.merges], "vocab_size": self.vocab_size}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(pair) for pair in payload["merges"]]
        return cls(merges=merges, vocab_size=payload["vocab_size"])
