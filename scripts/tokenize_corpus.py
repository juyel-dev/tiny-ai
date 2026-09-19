"""Encode a text corpus into a binary token-id file for BPE training.

Output format: little-endian uint16 token ids, readable directly by
tiny_ai.data.MappedTokenIds via mmap -- the BPE equivalent of the raw
.txt files MappedTokens reads for the byte tokenizer. See
scripts/train_tokenizer.py for the full workflow this fits into.

Streams the input line-by-line and flushes ids in batches rather than
loading the whole corpus (and the whole encoded id list) into memory at
once. This matters at real scale: TinyStories' training split alone is
~1.9 GB of text, and materializing that plus the encoded ids as Python
objects easily exceeds the RAM available on GitHub Actions' free CPU
runners (7 GB). Splitting on lines is safe here because
tiny_ai.bpe_tokenizer never merges across whitespace, and every line in
a corpus prepared by scripts/prepare_corpus.py ends in '\\n'.
"""
import argparse
from array import array
from pathlib import Path

from tiny_ai.bpe_tokenizer import BPETokenizer

FLUSH_EVERY_IDS = 1_000_000


def encode_stream(tokenizer: BPETokenizer, input_path: str, output_path: str) -> int:
    total_ids = 0
    buf = array("H")
    with open(input_path, "r", encoding="utf-8") as in_f, open(output_path, "wb") as out_f:
        for line in in_f:
            try:
                buf.extend(tokenizer.encode(line))
            except OverflowError as e:
                raise ValueError(
                    "Token id exceeds uint16 range; MappedTokenIds cannot store it."
                ) from e
            if len(buf) >= FLUSH_EVERY_IDS:
                buf.tofile(out_f)
                total_ids += len(buf)
                buf = array("H")
        if buf:
            buf.tofile(out_f)
            total_ids += len(buf)
    return total_ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", required=True, help="Path saved by scripts/train_tokenizer.py")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    total = encode_stream(tokenizer, args.input, args.output)

    print(f"tokens: {total:,}")
    print(f"vocab size: {tokenizer.vocab_size:,}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
