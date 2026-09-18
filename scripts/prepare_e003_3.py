import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from tiny_ai.tokenizer import ByteTokenizer

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = CONTROL_RE.sub("", text)
    return re.sub(r"[ \t]+", " ", text)


def record_key(instruction: str, context: str, response: str) -> str:
    raw = "\0".join((instruction, context, response)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def format_prompt(instruction: str, context: str) -> str:
    if context:
        return f"User: {instruction}\nContext: {context}\nAssistant: "
    return f"User: {instruction}\nAssistant: "


def fits(tokenizer, instruction: str, context: str, response: str, block_size: int,
         max_instruction_bytes: int, max_response_bytes: int) -> bool:
    if not instruction or not response:
        return False
    if len(instruction.encode("utf-8")) > max_instruction_bytes:
        return False
    if len(response.encode("utf-8")) > max_response_bytes:
        return False
    return len(
        tokenizer.encode(format_prompt(instruction, context) + response, add_eos=True)
    ) <= block_size


def load_eligible(path: Path, block_size: int, max_instruction_bytes: int,
                  max_context_bytes: int, max_response_bytes: int):
    tokenizer = ByteTokenizer()
    groups = defaultdict(list)
    seen = set()
    total = 0
    kept = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            instruction = clean(row.get("instruction", ""))
            context = clean(row.get("context", ""))
            response = clean(row.get("response", ""))
            category = clean(row.get("category", "general_qa")) or "general_qa"

            if len(context.encode("utf-8")) > max_context_bytes:
                continue
            if not fits(
                tokenizer, instruction, context, response, block_size,
                max_instruction_bytes, max_response_bytes
            ):
                continue

            key = record_key(instruction, context, response)
            if key in seen:
                continue
            seen.add(key)
            groups[category].append({
                "instruction": instruction,
                "context": context,
                "response": response,
                "category": category,
            })
            kept += 1

    return groups, total, kept


def balanced_take(groups, count, seed):
    rng = random.Random(seed)
    buckets = {category: list(rows) for category, rows in groups.items()}
    for rows in buckets.values():
        rng.shuffle(rows)

    selected = []
    categories = sorted(buckets)
    while len(selected) < count and any(buckets.values()):
        for category in categories:
            if buckets[category] and len(selected) < count:
                selected.append(buckets[category].pop())
    return selected


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description="Prepare short, context-safe Dolly instruction data.")
    p.add_argument("--source", required=True)
    p.add_argument("--max-train", type=int, default=1600)
    p.add_argument("--max-val", type=int, default=240)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--max-instruction-bytes", type=int, default=180)
    p.add_argument("--max-context-bytes", type=int, default=120)
    p.add_argument("--max-response-bytes", type=int, default=160)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default="data/processed/e003_3")
    args = p.parse_args()

    groups, total, kept = load_eligible(
        Path(args.source),
        args.block_size,
        args.max_instruction_bytes,
        args.max_context_bytes,
        args.max_response_bytes,
    )

    # Split each category before truncating, keeping category diversity in both sets.
    rng = random.Random(args.seed)
    train_pool = defaultdict(list)
    val_pool = defaultdict(list)
    for category, rows in groups.items():
        rng.shuffle(rows)
        cut = max(1, round(len(rows) * 0.15))
        val_pool[category] = rows[:cut]
        train_pool[category] = rows[cut:]

    train = balanced_take(train_pool, args.max_train, args.seed)
    val = balanced_take(val_pool, args.max_val, args.seed + 1)

    if not train or not val:
        raise ValueError("Not enough eligible Dolly examples after filtering.")

    out = Path(args.out_dir)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)

    categories = sorted({row["category"] for row in train + val})
    print(f"source examples: {total:,}")
    print(f"eligible unique examples: {kept:,}")
    print(f"train examples: {len(train):,}")
    print(f"val examples: {len(val):,}")
    print(f"categories: {', '.join(categories)}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
