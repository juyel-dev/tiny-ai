import argparse
import json
import random
from pathlib import Path

from tiny_ai.tokenizer import ByteTokenizer
from prepare_e003 import CHAT_EXAMPLES, reservoir_sample


def split_examples(examples, val_frac, seed):
    rng = random.Random(seed)
    items = list(examples)
    rng.shuffle(items)
    val_count = max(1, round(len(items) * val_frac))
    return items[val_count:], items[:val_count]


def write_jsonl(path: Path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description="Prepare balanced E003.2 chat/story fine-tuning data.")
    p.add_argument("--source", required=True)
    p.add_argument("--n-stories", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-chars", type=int, default=850)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default="data/processed/e003_2")
    args = p.parse_args()

    if not 0 < args.val_frac < 0.5:
        raise ValueError("--val-frac must be between 0 and 0.5")

    tokenizer = ByteTokenizer()
    stories, skipped = reservoir_sample(
        Path(args.source),
        args.n_stories,
        args.seed,
        args.max_chars,
        args.block_size,
    )

    story_examples = []
    for story in stories:
        prompt = random.Random(args.seed + len(story)).choice([
            "Tell me a story.",
            "Can you tell me a short story?",
            "Please tell me a story.",
            "I would like a story.",
            "Tell me a little story.",
        ])
        if len(tokenizer.encode(f"User: {prompt}\nAssistant: " + story)) <= args.block_size:
            story_examples.append({"kind": "story", "prompt": prompt, "response": story})

    chat_examples = [
        {"kind": "chat", "prompt": prompt, "response": response}
        for prompt, response in CHAT_EXAMPLES
    ]

    story_train, story_val = split_examples(story_examples, args.val_frac, args.seed)
    chat_train, chat_val = split_examples(chat_examples, args.val_frac, args.seed + 1)

    out = Path(args.out_dir)
    write_jsonl(out / "train.jsonl", story_train + chat_train)
    write_jsonl(out / "val.jsonl", story_val + chat_val)

    print(f"story train: {len(story_train):,}")
    print(f"story val: {len(story_val):,}")
    print(f"chat train: {len(chat_train):,}")
    print(f"chat val: {len(chat_val):,}")
    print(f"source stories skipped: {skipped:,}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
