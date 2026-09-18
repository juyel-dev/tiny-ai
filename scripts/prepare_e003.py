import argparse
import json
import random
from pathlib import Path

from tiny_ai.tokenizer import ByteTokenizer

CHAT_EXAMPLES = [
    ("Hello!", "Hello!"),
    ("Hi!", "Hi! How are you?"),
    ("Hey, how are you?", "I am doing well. How are you?"),
    ("Good morning.", "Good morning!"),
    ("Good night.", "Good night!"),
    ("How are you?", "I am doing well. Thanks for asking."),
    ("What can you do?", "I can chat with you and tell simple stories."),
    ("Can you help me?", "Yes. Tell me what you need help with."),
    ("Thank you.", "You are welcome!"),
    ("Thanks!", "You are welcome!"),
    ("Bye.", "Goodbye!"),
    ("See you later.", "See you later!"),
    ("Tell me something nice.", "You are doing great. Keep going!"),
    ("I am happy today.", "That is nice to hear!"),
    ("I am sad today.", "I am sorry you feel sad. I hope things get better."),
    ("I am tired.", "Take a little rest and then continue when you are ready."),
    ("What is your name?", "I am Tiny-AI."),
    ("Are you a person?", "No. I am a small language model."),
    ("Can we talk?", "Yes! What would you like to talk about?"),
    ("I need a story.", "Sure! Ask me to tell you a story."),
    ("Please tell me a short story.", "Once upon a time, a little child found a tiny bird and helped it get home."),
    ("Can you tell a story?", "Yes! Tell me what kind of story you would like."),
    ("Do you like stories?", "Yes. Stories are fun to read and tell."),
    ("What should I do next?", "Tell me what you are working on, and we can start there."),
    ("I made a mistake.", "That is okay. Mistakes help us learn."),
    ("I am bored.", "Let us make up a little story."),
    ("Can you repeat that?", "Sure. Please tell me which part you want me to repeat."),
    ("Okay.", "Great!"),
    ("Cool.", "Nice!"),
    ("Interesting.", "I am glad you think so!"),
    ("What is 2 + 2?", "4."),
    ("What is 5 + 3?", "8."),
    ("What is 10 - 4?", "6."),
    ("What is 3 + 3?", "6."),
]


def iter_records(path: Path):
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


def reservoir_sample(path: Path, count: int, seed: int, max_chars: int, block_size: int) -> tuple[list[str], int]:
    rng = random.Random(seed)
    tokenizer = ByteTokenizer()
    sample = []
    eligible = 0
    skipped = 0

    for record in iter_records(path):
        if len(record.encode("utf-8")) > max_chars:
            skipped += 1
            continue
        if len(tokenizer.encode("User: Tell me a story.\nAssistant: " + record)) > block_size:
            skipped += 1
            continue

        eligible += 1
        if len(sample) < count:
            sample.append(record)
            continue
        j = rng.randrange(eligible)
        if j < count:
            sample[j] = record

    return sample, skipped


def make_story_examples(stories, seed: int, block_size: int):
    rng = random.Random(seed)
    templates = [
        "Tell me a story.",
        "Can you tell me a short story?",
        "Please tell me a story.",
        "I would like a story.",
        "Tell me a little story.",
    ]
    tokenizer = ByteTokenizer()
    examples = []
    skipped = 0
    for story in stories:
        prompt = rng.choice(templates)
        encoded = tokenizer.encode(f"User: {prompt}\nAssistant: " + story)
        if len(encoded) > block_size:
            skipped += 1
            continue
        examples.append({"prompt": prompt, "response": story})
    return examples, skipped


def write_jsonl(path: Path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(
        description="Prepare the local E002 corpus for E003 chat/instruction fine-tuning."
    )
    p.add_argument("--source", required=True, help="E002 processed training text file.")
    p.add_argument("--n-stories", type=int, default=500)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--max-chars", type=int, default=850)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default="data/processed/e003")
    args = p.parse_args()

    if not 0 < args.val_frac < 0.5:
        raise ValueError("--val-frac must be between 0 and 0.5")

    stories, skipped = reservoir_sample(
        Path(args.source),
        args.n_stories,
        args.seed,
        args.max_chars,
        args.block_size,
    )
    if not stories:
        raise ValueError("No stories fit the configured E003 context limit.")

    story_examples, skipped = make_story_examples(stories, args.seed, args.block_size)
    examples = story_examples
    examples.extend({"prompt": prompt, "response": response} for prompt, response in CHAT_EXAMPLES)

    rng = random.Random(args.seed)
    rng.shuffle(examples)

    val_count = max(1, round(len(examples) * args.val_frac))
    val = examples[:val_count]
    train = examples[val_count:]

    out = Path(args.out_dir)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)

    print(f"story examples: {len(stories):,}")
    print(f"story examples sampled: {len(stories):,}")
    print(f"source stories skipped by filters: {skipped:,}")
    print(f"story examples kept: {len(story_examples):,}")
    print(f"chat examples: {len(CHAT_EXAMPLES):,}")
    print(f"train examples: {len(train):,}")
    print(f"val examples: {len(val):,}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
