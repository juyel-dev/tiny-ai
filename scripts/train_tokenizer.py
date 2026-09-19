"""Train a byte-level BPE tokenizer on a local text corpus.

Runs fully offline: the tokenizer is trained directly on your training
text (tiny_ai/bpe_tokenizer.py), so it's exactly reproducible from
(corpus file, vocab size) alone -- no model hub, no network access.

Typical flow:
    python scripts/train_tokenizer.py \
        --corpus data/processed/e002/train.txt \
        --vocab-size 4096 \
        --out tokenizer/e002_bpe4096.json

    python scripts/tokenize_corpus.py \
        --tokenizer tokenizer/e002_bpe4096.json \
        --input data/processed/e002/train.txt \
        --output data/processed/e002/train.bpe.bin
    # repeat tokenize_corpus.py for val.txt

    python scripts/train_e002.py \
        --train data/processed/e002/train.bpe.bin \
        --val data/processed/e002/val.bpe.bin \
        --tokenizer bpe --bpe-merges tokenizer/e002_bpe4096.json
"""
import argparse
from pathlib import Path

from tiny_ai.bpe_tokenizer import train_bpe


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, help="Text file to train the tokenizer on (usually the training split).")
    p.add_argument("--vocab-size", type=int, default=4096,
                    help="Total vocab size, including the 256 raw bytes and 3 special tokens.")
    p.add_argument("--out", default="tokenizer/bpe.json")
    p.add_argument("--max-chars", type=int, default=None,
                    help="Optional cap on how many characters of the corpus to train on, "
                         "for speed AND memory on very large corpora (only that many "
                         "characters are read off disk, not the whole file). Training "
                         "cost scales with the number of distinct words, so this is "
                         "usually only needed for corpora with a huge, long-tailed "
                         "vocabulary -- but on a multi-GB corpus, skipping it also means "
                         "reading the entire file into memory at once.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    with open(args.corpus, "r", encoding="utf-8") as f:
        text = f.read(args.max_chars) if args.max_chars is not None else f.read()

    tokenizer = train_bpe(text, args.vocab_size, verbose=args.verbose)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out)

    print(f"trained merges: {len(tokenizer.merges):,}")
    print(f"vocab size: {tokenizer.vocab_size:,}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
