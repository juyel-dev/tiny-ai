import argparse
import hashlib
import re
from pathlib import Path

WHITESPACE_RE = re.compile(r"[ \t\r]+")
BLANKLINES_RE = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_records(path: Path):
    parts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                parts.append(line)
            elif parts:
                record = normalize("".join(parts))
                if record:
                    yield record
                parts = []
        if parts:
            record = normalize("".join(parts))
            if record:
                yield record


def write_split(input_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    duplicates = 0
    examples = 0
    byte_tokens = 0

    with output_path.open("w", encoding="utf-8") as out:
        first = True
        for record in iter_records(input_path):
            digest = sha256_text(record)
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            if not first:
                out.write("\n\n")
            out.write(record)
            first = False
            examples += 1
            byte_tokens += len(record.encode("utf-8"))
        out.write("\n")

    return examples, byte_tokens, duplicates


def main():
    p = argparse.ArgumentParser(
        description="Normalize and deduplicate train/validation text while preserving their official split."
    )
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out-dir", default="data/processed/e002")
    args = p.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    out = Path(args.out_dir)

    train = write_split(train_path, out / "train.txt")
    val = write_split(val_path, out / "val.txt")

    print(f"train examples: {train[0]:,}")
    print(f"train byte tokens: {train[1]:,}")
    print(f"train duplicates removed: {train[2]:,}")
    print(f"val examples: {val[0]:,}")
    print(f"val byte tokens: {val[1]:,}")
    print(f"val duplicates removed: {val[2]:,}")


if __name__ == "__main__":
    main()
