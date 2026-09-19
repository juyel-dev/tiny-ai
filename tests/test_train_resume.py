import subprocess
import sys
from pathlib import Path

import torch

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_e002.py"

COMMON_ARGS = [
    "--n-layer", "1", "--n-head", "2", "--n-embd", "16", "--block-size", "8",
    "--batch-size", "4", "--warmup-steps", "2", "--eval-batches", "1",
]


def _make_corpus(path: Path, n_tokens: int) -> None:
    # --tokenizer byte (the default, used throughout this test) reads raw
    # single-byte-per-token files via MappedTokens.
    ids = bytes((i * 37 + 5) % 256 for i in range(n_tokens))
    path.write_bytes(ids)


def _run(args, cwd):
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), *args],
        cwd=cwd, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_resume_continues_training_past_an_interrupted_run(tmp_path):
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _make_corpus(train_path, 500)
    _make_corpus(val_path, 200)

    resume_ckpt = tmp_path / "resume.pt"
    leg1_out = tmp_path / "leg1_final.pt"   # throwaway; leg 1 is the "interrupted" run
    leg2_out = tmp_path / "leg2_final.pt"   # the checkpoint we actually care about

    # Leg 1 stands in for a run that gets interrupted (e.g. a GH Actions
    # timeout) before finishing: it completes 3 steps of *some* target and
    # periodically saves a resumable checkpoint along the way.
    _run([
        "--train", str(train_path), "--val", str(val_path),
        "--steps", "3", "--save-every", "1",
        "--resume-out", str(resume_ckpt), "--out", str(leg1_out),
        "--device", "cpu", *COMMON_ARGS,
    ], cwd=tmp_path)

    assert resume_ckpt.exists()
    checkpoint = torch.load(resume_ckpt, map_location="cpu")
    assert checkpoint["step"] == 2  # save_every skips the final step of that leg
    assert "optimizer" in checkpoint

    # Leg 2: resume from the periodic checkpoint and train on to a larger
    # total step count, as if a fresh CI job picked up where leg 1 left off.
    result = _run([
        "--train", str(train_path), "--val", str(val_path),
        "--steps", "6", "--resume", str(resume_ckpt),
        "--out", str(leg2_out),
        "--device", "cpu", *COMMON_ARGS,
    ], cwd=tmp_path)

    assert "resumed from" in result.stdout
    assert "step 2" in result.stdout.split("resumed from", 1)[1].splitlines()[0]
    assert leg2_out.exists()
    final = torch.load(leg2_out, map_location="cpu")
    assert final["step"] == 6
    assert "optimizer" not in final  # final checkpoints stay lean


def test_resume_with_nothing_left_to_do_is_a_noop(tmp_path):
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _make_corpus(train_path, 500)
    _make_corpus(val_path, 200)
    resume_ckpt = tmp_path / "resume.pt"
    out = tmp_path / "final.pt"

    _run([
        "--train", str(train_path), "--val", str(val_path),
        "--steps", "2", "--save-every", "1", "--resume-out", str(resume_ckpt),
        "--out", str(out),
        "--device", "cpu", *COMMON_ARGS,
    ], cwd=tmp_path)

    result = _run([
        "--train", str(train_path), "--val", str(val_path),
        "--steps", "1", "--resume", str(resume_ckpt),
        "--device", "cpu", *COMMON_ARGS,
    ], cwd=tmp_path)
    assert "nothing to do" in result.stdout


def test_resume_rejects_mismatched_architecture(tmp_path):
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _make_corpus(train_path, 500)
    _make_corpus(val_path, 200)
    resume_ckpt = tmp_path / "resume.pt"

    _run([
        "--train", str(train_path), "--val", str(val_path),
        "--steps", "2", "--save-every", "1", "--resume-out", str(resume_ckpt),
        "--device", "cpu", *COMMON_ARGS,
    ], cwd=tmp_path)
    assert resume_ckpt.exists()

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT),
         "--train", str(train_path), "--val", str(val_path),
         "--steps", "4", "--resume", str(resume_ckpt),
         "--device", "cpu",
         "--n-layer", "2", "--n-head", "2", "--n-embd", "16", "--block-size", "8"],  # n-layer differs
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "does not match" in (result.stdout + result.stderr)
