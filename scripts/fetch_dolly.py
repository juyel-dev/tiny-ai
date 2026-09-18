import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "databricks/databricks-dolly-15k"
REVISION = "9be7fbc9723eccdbdd4e9d0c3e6ae3519d06992a"
FILENAME = "databricks-dolly-15k.jsonl"
EXPECTED_SHA256 = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description="Download the pinned Dolly 15K source.")
    p.add_argument("--out", default="data/raw/dolly/databricks-dolly-15k.jsonl")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not args.force:
        actual = sha256_file(out)
        if actual == EXPECTED_SHA256:
            print(f"already present: {out}")
            print(f"sha256: {actual}")
            return
        raise RuntimeError(f"Existing file has unexpected SHA-256: {actual}. Use --force.")

    cached = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=FILENAME,
            revision=REVISION,
        )
    )
    shutil.copyfile(cached, out)

    actual = sha256_file(out)
    if actual != EXPECTED_SHA256:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual}")

    print(f"saved: {out}")
    print(f"revision: {REVISION}")
    print(f"sha256: {actual}")


if __name__ == "__main__":
    main()
