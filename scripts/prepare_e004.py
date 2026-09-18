import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from tiny_ai.bpe_tokenizer import E004BPETokenizer, build_tokenizer


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return CONTROL_RE.sub("", text)


def iter_story_records(path: Path):
    parts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                parts.append(line)
            elif parts:
                record = clean("".join(parts))
                if record:
                    yield record
                parts = []
        if parts:
            record = clean("".join(parts))
            if record:
                yield record


def iter_dolly_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            instruction = clean(row.get("instruction", ""))
            context = clean(row.get("context", ""))
            response = clean(row.get("response", ""))
            category = clean(row.get("category", "general_qa")) or "general_qa"
            if instruction and response:
                yield {
                    "instruction": instruction,
                    "context": context,
                    "response": response,
                    "category": category,
                }


def format_instruction(row: dict) -> str:
    if row["context"]:
        return (
            f"User: {row['instruction']}\n"
            f"Context: {row['context']}\n"
            f"Assistant: {row['response']}"
        )
    return f"User: {row['instruction']}\nAssistant: {row['response']}"


def tokenizer_iterator(story_path: Path, dolly_path: Path, max_story_examples: int):
    for index, story in enumerate(iter_story_records(story_path)):
        if index >= max_story_examples:
            break
        yield story
    for row in iter_dolly_rows(dolly_path):
        yield format_instruction(row)


def row_key(row: dict) -> str:
    return hashlib.sha256(
        "\0".join(
            [row["instruction"], row["context"], row["response"]]
        ).encode("utf-8")
    ).hexdigest()


def split_by_category(rows: list[dict], val_frac: float, seed: int):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)

    train, val = [], []
    for category, items in groups.items():
        rng.shuffle(items)
        val_count = max(1, round(len(items) * val_frac))
        val.extend(items[:val_count])
        train.extend(items[val_count:])
    return train, val


def round_robin_take(groups: list[dict], count: int):
    by_category = defaultdict(list)
    for row in groups:
        by_category[row["category"]].append(row)
    categories = sorted(by_category)
    selected = []
    index = 0
    while len(selected) < count and any(by_category.values()):
        category = categories[index % len(categories)]
        if by_category[category]:
            selected.append(by_category[category].pop())
        index += 1
    return selected


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_u16(path: Path, sequences):
    import struct

    path.parent.mkdir(parents=True, exist_ok=True)
    token_count = 0
    with path.open("wb") as handle:
        for ids in sequences:
            if ids and max(ids) >= 65536:
                raise ValueError("Token ID exceeds uint16 range.")
            if ids:
                handle.write(struct.pack(f"<{len(ids)}H", *ids))
                token_count += len(ids)
    return token_count


def encode_story_file(
    tokenizer: E004BPETokenizer, source: Path, out: Path, progress_every: int = 10_000
):
    token_count = 0
    story_count = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    import struct

    with out.open("wb") as handle:
        for story_count, story in enumerate(iter_story_records(source), start=1):
            ids = tokenizer.encode(story, add_bos=True, add_eos=True)
            if ids and max(ids) >= 65536:
                raise ValueError("Token ID exceeds uint16 range.")
            if ids:
                handle.write(struct.pack(f"<{len(ids)}H", *ids))
                token_count += len(ids)
            if story_count % progress_every == 0:
                print(
                    f"  {source.name}: {story_count:,} stories | "
                    f"{token_count:,} tokens written",
                    flush=True,
                )
    print(
        f"  {source.name}: DONE | {story_count:,} stories | "
        f"{token_count:,} tokens",
        flush=True,
    )
    return token_count


def encode_dolly_rows(
    tokenizer: E004BPETokenizer, rows, out: Path, progress_every: int = 200
):
    token_count = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    import struct

    with out.open("wb") as handle:
        for row_index, row in enumerate(rows, start=1):
            ids = tokenizer.encode(
                format_instruction(row),
                add_bos=True,
                add_eos=True,
            )
            if ids and max(ids) >= 65536:
                raise ValueError("Token ID exceeds uint16 range.")
            if ids:
                handle.write(struct.pack(f"<{len(ids)}H", *ids))
                token_count += len(ids)
            if row_index % progress_every == 0:
                print(
                    f"  {out.name}: {row_index:,} rows | "
                    f"{token_count:,} tokens written",
                    flush=True,
                )
    print(
        f"  {out.name}: DONE | {len(rows):,} rows | "
        f"{token_count:,} tokens",
        flush=True,
    )
    return token_count


def main():
    p = argparse.ArgumentParser(
        description="Train E004 BPE and build compact uint16 training corpora."
    )
    p.add_argument("--story-train", required=True)
    p.add_argument("--story-val", required=True)
    p.add_argument("--dolly-source", required=True)
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--tokenizer-story-examples", type=int, default=100000)
    p.add_argument("--max-instruction-bytes", type=int, default=300)
    p.add_argument("--max-context-bytes", type=int, default=250)
    p.add_argument("--max-response-bytes", type=int, default=450)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--max-dolly-train", type=int, default=1600)
    p.add_argument("--max-dolly-val", type=int, default=240)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default="data/processed/e004")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("E004 preparation started.", flush=True)
    print("Stage 1/5: training ByteLevel BPE tokenizer...", flush=True)

    tokenizer, trainer = build_tokenizer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    tokenizer.train_from_iterator(
        tokenizer_iterator(
            Path(args.story_train),
            Path(args.dolly_source),
            args.tokenizer_story_examples,
        ),
        trainer=trainer,
    )

    wrapper = E004BPETokenizer(tokenizer)
    if wrapper.vocab_size != args.vocab_size:
        raise RuntimeError(
            f"Tokenizer produced {wrapper.vocab_size} tokens; expected {args.vocab_size}."
        )

    tokenizer_path = out / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(
        f"Stage 1/5 DONE: vocab={wrapper.vocab_size}, saved={tokenizer_path}",
        flush=True,
    )

    print("Stage 2/5: filtering and splitting Dolly...", flush=True)
    eligible = []
    seen = set()
    source_count = 0
    for row in iter_dolly_rows(Path(args.dolly_source)):
        source_count += 1
        if (
            len(row["instruction"].encode("utf-8")) > args.max_instruction_bytes
            or len(row["context"].encode("utf-8")) > args.max_context_bytes
            or len(row["response"].encode("utf-8")) > args.max_response_bytes
        ):
            continue

        encoded = wrapper.encode(
            format_instruction(row),
            add_bos=True,
            add_eos=True,
        )
        if len(encoded) > args.block_size:
            continue

        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        eligible.append(row)

        if source_count % 2_000 == 0:
            print(
                f"  Dolly scanned: {source_count:,} | eligible: {len(eligible):,}",
                flush=True,
            )

    eligible_train, eligible_val = split_by_category(
        eligible, args.val_frac, args.seed
    )
    train_rows = round_robin_take(
        eligible_train, min(args.max_dolly_train, len(eligible_train))
    )
    val_rows = round_robin_take(
        eligible_val, min(args.max_dolly_val, len(eligible_val))
    )

    if not train_rows or not val_rows:
        raise RuntimeError("Dolly filtering produced an empty train or validation split.")

    write_jsonl(out / "dolly_train.jsonl", train_rows)
    write_jsonl(out / "dolly_val.jsonl", val_rows)
    print(
        f"Stage 2/5 DONE: source={source_count:,}, eligible={len(eligible):,}, "
        f"train={len(train_rows):,}, val={len(val_rows):,}",
        flush=True,
    )

    print("Stage 3/5: tokenizing full TinyStories train...", flush=True)
    story_train_tokens = encode_story_file(
        wrapper, Path(args.story_train), out / "story_train.u16"
    )

    print("Stage 4/5: tokenizing full TinyStories validation...", flush=True)
    story_val_tokens = encode_story_file(
        wrapper, Path(args.story_val), out / "story_val.u16"
    )

    print("Stage 5/5: tokenizing selected Dolly splits...", flush=True)
    dolly_train_tokens = encode_dolly_rows(
        wrapper, train_rows, out / "dolly_train.u16"
    )
    dolly_val_tokens = encode_dolly_rows(
        wrapper, val_rows, out / "dolly_val.u16"
    )

    counts = {
        "story_train_tokens": story_train_tokens,
        "story_val_tokens": story_val_tokens,
        "dolly_train_tokens": dolly_train_tokens,
        "dolly_val_tokens": dolly_val_tokens,
    }

    metadata = {
        "vocab_size": wrapper.vocab_size,
        "special_tokens": {
            "pad": wrapper.pad_id,
            "bos": wrapper.bos_id,
            "eos": wrapper.eos_id,
        },
        "tokenizer_story_examples": args.tokenizer_story_examples,
        "dolly_source_examples": source_count,
        "dolly_eligible_unique": len(eligible),
        "dolly_train_examples": len(train_rows),
        "dolly_val_examples": len(val_rows),
        "token_counts": counts,
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"vocab size: {wrapper.vocab_size}")
    print(f"pad id: {wrapper.pad_id}")
    print(f"bos id: {wrapper.bos_id}")
    print(f"eos id: {wrapper.eos_id}")
    print(f"dolly source examples: {source_count:,}")
    print(f"dolly eligible unique: {len(eligible):,}")
    print(f"dolly train examples: {len(train_rows):,}")
    print(f"dolly val examples: {len(val_rows):,}")
    for key, value in counts.items():
        print(f"{key}: {value:,}")
    print(f"output: {out}")
    print("E004 preparation COMPLETE.", flush=True)


if __name__ == "__main__":
    main()
