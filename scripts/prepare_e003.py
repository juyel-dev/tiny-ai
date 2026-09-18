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

# Additional authored pairs make E003.1 less dependent on repeated greeting templates.
CHAT_EXAMPLES.extend([
    ("Hello there.", "Hello! Nice to meet you."),
    ("Hey there.", "Hey! How can I help?"),
    ("Hi there.", "Hi! What would you like to talk about?"),
    ("Good afternoon.", "Good afternoon!"),
    ("Good evening.", "Good evening!"),
    ("How is your day going?", "It is going well. How is your day going?"),
    ("What are you doing?", "I am here chatting with you."),
    ("What are you?", "I am Tiny-AI, a small language model."),
    ("What is your job?", "My job is to generate text and chat with you."),
    ("Who are you?", "I am Tiny-AI."),
    ("Are you human?", "No. I am a small language model."),
    ("Are you real?", "I am a computer program, not a person."),
    ("Can you chat with me?", "Yes!"),
    ("Can you answer a question?", "Yes. Ask me a question."),
    ("Can you tell me a joke?", "I can try to make up a little joke."),
    ("Can you tell me a poem?", "Sure. I can try to write a short poem."),
    ("Can you tell me a story about a dog?", "Sure! I can make up a short story about a dog."),
    ("Can you write something short?", "Yes. Tell me what you want written."),
    ("Can you explain this?", "Yes. Tell me what you want explained."),
    ("Can you help with homework?", "I can help you think through a homework question."),
    ("Please help me.", "Sure. What do you need help with?"),
    ("I need some help.", "Okay. Tell me what you need."),
    ("I have a question.", "Go ahead and ask it."),
    ("I have an idea.", "Great! Tell me your idea."),
    ("I want to learn.", "Great. What would you like to learn?"),
    ("Let's talk about dogs.", "Sure! Dogs are a fun topic."),
    ("Let's talk.", "Sure! What would you like to talk about?"),
    ("What should we talk about?", "We can talk about stories, ideas, or everyday things."),
    ("Tell me a fun fact.", "A day has 24 hours."),
    ("What comes after one?", "Two."),
    ("What comes before five?", "Four."),
    ("What number is between two and four?", "Three."),
    ("What is 1 + 1?", "2."),
    ("What is 1 + 2?", "3."),
    ("What is 2 + 3?", "5."),
    ("What is 4 + 4?", "8."),
    ("What is 5 + 5?", "10."),
    ("What is 6 + 2?", "8."),
    ("What is 7 + 2?", "9."),
    ("What is 8 + 1?", "9."),
    ("What is 9 + 1?", "10."),
    ("What is 10 + 5?", "15."),
    ("What is 10 - 1?", "9."),
    ("What is 10 - 2?", "8."),
    ("What is 10 - 3?", "7."),
    ("What is 10 - 5?", "5."),
    ("What is 8 - 3?", "5."),
    ("What is 7 - 4?", "3."),
    ("What is 6 - 1?", "5."),
    ("What is 5 - 2?", "3."),
    ("What is 4 - 1?", "3."),
    ("What is 12 - 2?", "10."),
    ("What is 2 times 3?", "6."),
    ("What is 3 times 3?", "9."),
    ("What is 4 times 2?", "8."),
    ("What is 2 times 5?", "10."),
    ("What color is the sky on a clear day?", "Blue."),
    ("How many days are in a week?", "Seven."),
    ("How many hours are in a day?", "24."),
    ("How many minutes are in an hour?", "60."),
    ("How many seconds are in a minute?", "60."),
    ("What do bees make?", "Honey."),
    ("What do cows give us?", "Milk."),
    ("What do fish live in?", "Water."),
    ("What do birds use to fly?", "Wings."),
    ("What do plants need to grow?", "Water, light, and nutrients."),
    ("What is frozen water called?", "Ice."),
    ("What is water made of?", "Water is H2O."),
    ("What planet do we live on?", "Earth."),
    ("What is the opposite of hot?", "Cold."),
    ("What is the opposite of big?", "Small."),
    ("What is the opposite of up?", "Down."),
    ("What is the opposite of day?", "Night."),
    ("What is the opposite of fast?", "Slow."),
    ("Say hello.", "Hello!"),
    ("Say goodbye.", "Goodbye!"),
    ("Answer with yes.", "Yes."),
    ("Answer with no.", "No."),
    ("Be brief.", "Okay."),
    ("Be friendly.", "Sure!"),
    ("Repeat: hello.", "Hello."),
    ("Repeat: thank you.", "Thank you."),
    ("What can I ask you?", "You can ask me simple questions or ask for a story."),
    ("Can you remember everything?", "No. I can make mistakes and I do not remember everything."),
    ("Do you make mistakes?", "Yes. I can make mistakes."),
    ("What if I do not understand?", "You can ask me to explain it another way."),
    ("Please keep it short.", "Okay. I will keep it short."),
    ("Please say that again.", "Sure. Tell me what you want repeated."),
    ("That sounds good.", "Great!"),
    ("That is funny.", "I am glad you enjoyed it!"),
    ("I understand.", "Great!"),
    ("I do not understand.", "That is okay. I can explain it more simply."),
    ("I am confused.", "No problem. We can go step by step."),
    ("I am learning.", "Keep going. Practice helps."),
    ("I finished.", "Nice work!"),
    ("I am ready.", "Great! Let us begin."),
    ("Let's begin.", "Okay. What should we start with?"),
    ("One more thing.", "Sure. What is it?"),
    ("Wait a moment.", "Okay."),
])


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

    stories, source_skipped = reservoir_sample(
        Path(args.source),
        args.n_stories,
        args.seed,
        args.max_chars,
        args.block_size,
    )
    if not stories:
        raise ValueError("No stories fit the configured E003 context limit.")

    story_examples, story_skipped = make_story_examples(stories, args.seed, args.block_size)
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

    print(f"story examples sampled: {len(stories):,}")
    print(f"source stories skipped by filters: {source_skipped:,}")
    print(f"sampled stories skipped by final template: {story_skipped:,}")
    print(f"chat examples: {len(CHAT_EXAMPLES):,}")
    print(f"train examples: {len(train):,}")
    print(f"val examples: {len(val):,}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
