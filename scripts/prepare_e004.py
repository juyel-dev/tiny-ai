import argparse
import json
from pathlib import Path
from typing import Iterable

from tiny_ai.bpe_tokenizer import E004BPETokenizer, build_tokenizer


def iter_story_records(path: Path) -> Iterable[str]:
    parts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                parts.append(line)
            elif parts:
                record = "".join(parts).strip()
                if record:
                    yield record
                parts = []
        if parts:
            record = "".join(parts).strip()
            if record:
                yield record


def iter_dolly_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row.get("instruction"), str):
                    raise ValueError(f"Invalid instruction in {path}")
                if not isinstance(row.get("response"), str):
                    raise ValueError(f"Invalid response in {path}")
                yield row


def format_instruction(row: dict) -> str:
    instruction = row["instruction"]
    context = row.get("context", "")
    if context:
        return f"User: {instruction}\nContext: {context}\nAssistant: {row['response']}"
    return f"User: {instruction}\nAssistant: {row['response']}"


def tokenizer_iterator(story_path: Path, dolly_path: Path):
    for story in iter_story_records(story_path):
        yield story
    for row in iter_dolly_rows(dolly_path):
        yield format_instruction(row)


def encode_text(tokenizer, text: str) -> list[int]:
    return [tokenizer.bos_id] + tokenizer.tokenizer.encode(text, add_special_tokens=False).ids + [tokenizer.eos_id]


def write_u16(path: Path, sequences: Iterable[list[int]]):
    import struct

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as handle:
        for ids in sequences:
            if not ids:
                continue
            if max(ids) >= 65536:
                raise ValueError("Token ID exceeds uint16 storage range.")
            handle.write(struct.pack(f"<{len(ids)}H", *ids))
            count += len(ids)
    return count


def encode_story_file(tokenizer, source: Path, out: Path):
    return write_u16(
        out,
        (encode_text(tokenizer, story) for story in iter_story_records(source)),
    )


def encode_dolly_file(tokenizer, source: Path, out: Path):
    return write_u16(
        out,
        (encode_text(tokenizer, format_instruction(row)) for row in iter_dolly_rows(source)),
    )


def main():
    p = argparse.ArgumentParser(description="Train E004 BPE and create disk-backed uint16 corpora.")
    p.add_argument("--story-train", required=True)
    p.add_argument("--story-val", required=True)
    p.add_argument("--dolly-train", required=True)
    p.add_argument("--dolly-val", required=True)
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--out-dir", default="data/processed/e004")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_story_train = Path(args.story_train)
    raw_dolly_train = Path(args.dolly_train)

    tokenizer, trainer = build_tokenizer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    tokenizer.train_from_iterator(
        tokenizer_iterator(raw_story_train, raw_dolly_train),
        trainer=trainer,
    )

    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab != args.vocab_size:
        raise RuntimeError(
            f"Tokenizer produced vocab size {actual_vocab}, expected {args.vocab_size}."
        )

    tokenizer_path = out / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    tokenizer_id = E004BPETokenizer(tokenizer)

    counts = {
        "story_train_tokens": encode_story_file(
            tokenizer_id, Path(args.story_train), out / "story_train.u16"
        ),
        "story_val_tokens": encode_story_file(
            tokenizer_id, Path(args.story_val), out / "story_val.u16"
        ),
        "dolly_train_tokens": encode_dolly_file(
            tokenizer_id, Path(args.dolly_train), out / "dolly_train.u16"
        ),
        "dolly_val_tokens": encode_dolly_file(
            tokenizer_id, Path(args.dolly_val), out / "dolly_val.u16"
        ),
    }

    meta = {
        "vocab_size": actual_vocab,
        "special_tokens": {
            "bos": tokenizer_id.bos_id,
            "eos": tokenizer_id.eos_id,
            "pad": tokenizer_id.pad_id,
        },
        "counts": counts,
    }
    (out / "metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print(f"vocab size: {actual_vocab}")
    print(f"tokenizer: {tokenizer_path}")
    for key, value in counts.items():
        print(f"{key}: {value:,}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
