"""Encode a text corpus into a binary token-id file for BPE training.

Output format: little-endian uint16 token ids, readable directly by
tiny_ai.data.MappedTokenIds via mmap -- the BPE equivalent of the raw
.txt files MappedTokens reads for the byte tokenizer. See
scripts/train_tokenizer.py for the full workflow this fits into.
"""
import argparse
from pathlib import Path

from tiny_ai.bpe_tokenizer import BPETokenizer
from tiny_ai.data import write_token_ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", required=True, help="Path saved by scripts/train_tokenizer.py")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    text = Path(args.input).read_text(encoding="utf-8")
    ids = tokenizer.encode(text)

    write_token_ids(ids, args.output)
    print(f"tokens: {len(ids):,}")
    print(f"vocab size: {tokenizer.vocab_size:,}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
