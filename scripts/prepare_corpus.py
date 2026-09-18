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


def split_records(text: str):
    return [normalize(x) for x in re.split(r"\n\s*\n", text) if normalize(x)]


def write_records(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(records) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out-dir", default="data/processed/e002")
    p.add_argument("--validation-fraction", type=float, default=0.01)
    args = p.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5")

    records, seen, duplicates = [], set(), 0
    for raw_path in args.input:
        for text in split_records(Path(raw_path).read_text(encoding="utf-8")):
            digest = sha256_text(text)
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            records.append(text)

    records.sort(key=sha256_text)
    n_val = max(1, round(len(records) * args.validation_fraction))
    val, train = records[:n_val], records[n_val:]
    out = Path(args.out_dir)
    write_records(train, out / "train.txt")
    write_records(val, out / "val.txt")
    print(f"unique examples: {len(records):,}")
    print(f"train examples: {len(train):,}")
    print(f"val examples: {len(val):,}")
    print(f"train byte tokens: {sum(len(x.encode('utf-8')) for x in train):,}")
    print(f"val byte tokens: {sum(len(x.encode('utf-8')) for x in val):,}")
    print(f"duplicates removed: {duplicates:,}")


if __name__ == "__main__":
    main()
