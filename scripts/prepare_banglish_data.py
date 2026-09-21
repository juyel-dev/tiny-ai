"""Fetch and prepare the BanglaTLit dataset (Fahim et al., EMNLP'24
Findings) for the Banglish -> Bangla transliteration experiment.

Source: https://huggingface.co/datasets/aplycaebous/BanglaTLit
~249,727 rows total; ~42,705 of them have a paired Bengali
back-transliteration (text_bengali), the rest are romanized-Bangla-only
(useful for language-model pretraining on the script even without a
label).

Writes, under --out-dir:
  pretrain_train.txt  one romanized-Bangla line per row (ALL rows, minus
                       the held-out slice below) -- for phase-1 plain
                       language-model pretraining, so the model sees
                       romanized Bangla broadly before it's asked to
                       transliterate it.
  pretrain_val.txt    held-out slice of the same, for tracking phase-1
                       pretrain loss (same hashing scheme as the pairs
                       split below, so it's deterministic and disjoint).
  pairs_train.tsv      banglish<TAB>bangla, one pair per line (paired rows only)
  pairs_val.tsv        same format, held out from training

Requires the `datasets` package (pip install datasets) and internet
access to huggingface.co -- intended to run inside GitHub Actions, not
in a network-restricted sandbox.
"""
import argparse
import hashlib
from pathlib import Path


def _clean(s: str) -> str:
    return " ".join(s.split())  # collapse all whitespace, strip


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="data/processed/banglish")
    p.add_argument("--val-fraction", type=float, default=0.03,
                    help="Fraction of paired rows held out for validation, "
                         "chosen deterministically by hashing the row id "
                         "(not randomly) so re-runs are reproducible.")
    args = p.parse_args()

    from datasets import load_dataset  # deferred: heavy import, network-dependent

    ds = load_dataset("aplycaebous/BanglaTLit", split="train")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_pretrain = 0
    n_pairs = 0
    n_train = 0
    n_val = 0

    with open(out_dir / "pretrain_train.txt", "w", encoding="utf-8") as pretrain_train_f, \
         open(out_dir / "pretrain_val.txt", "w", encoding="utf-8") as pretrain_val_f, \
         open(out_dir / "pairs_train.tsv", "w", encoding="utf-8") as train_f, \
         open(out_dir / "pairs_val.tsv", "w", encoding="utf-8") as val_f:
        for row in ds:
            banglish = row.get("text_transliterated")
            bangla = row.get("text_bengali")
            row_id = row.get("id", str(n_pretrain))
            bucket = int(hashlib.sha256(str(row_id).encode()).hexdigest(), 16) % 100

            if banglish:
                banglish = _clean(banglish)
                if banglish:
                    pretrain_target = pretrain_val_f if bucket < args.val_fraction * 100 else pretrain_train_f
                    pretrain_target.write(banglish + "\n")
                    n_pretrain += 1

            if banglish and bangla:
                bangla = _clean(bangla)
                if not bangla:
                    continue
                # No tab/newline should survive _clean, but guard the TSV
                # format explicitly rather than trust that.
                if "\t" in banglish or "\t" in bangla:
                    continue
                n_pairs += 1
                target_f = val_f if bucket < args.val_fraction * 100 else train_f
                target_f.write(f"{banglish}\t{bangla}\n")
                if target_f is val_f:
                    n_val += 1
                else:
                    n_train += 1

    print(f"pretrain lines: {n_pretrain:,}")
    print(f"  pretrain_train / pretrain_val split by the same {args.val_fraction:.0%} hash bucketing as pairs")
    print(f"paired rows: {n_pairs:,}")
    print(f"  train: {n_train:,}")
    print(f"  val:   {n_val:,}")
    print(f"saved under: {out_dir}")


if __name__ == "__main__":
    main()
