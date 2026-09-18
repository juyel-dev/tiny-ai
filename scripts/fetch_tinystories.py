import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "roneneldan/TinyStories"
REVISION = "5485261731eaac25dd8e5ebbc3839d0a9870b185"
FILES = {
    "train": ("TinyStories-train.txt", "c5cf5e22ff13614e830a2e849d5dd2cd153d5bc024901afeade7e35379d8f7b52"),
    "valid": ("TinyStories-valid.txt", "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description="Fetch pinned TinyStories files and verify SHA-256.")
    p.add_argument("--split", choices=["train", "valid", "both"], default="valid")
    p.add_argument("--out-dir", default="data/raw/tinystories")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = ["train", "valid"] if args.split == "both" else [args.split]
    for split in splits:
        filename, expected = FILES[split]
        cached = Path(hf_hub_download(repo_id=REPO_ID, filename=filename, revision=REVISION, repo_type="dataset"))
        target = out / filename
        if cached.resolve() != target.resolve():
            target.write_bytes(cached.read_bytes())
        actual = sha256(target)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {filename}: expected {expected}, got {actual}")
        print(f"{split}: {target} | sha256={actual}")


if __name__ == "__main__":
    main()
