import subprocess
import sys
import time
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
    _make_corpus(train_path, 2000)
    _make_corpus(val_path, 500)

    resume_ckpt = tmp_path / "resume.pt"
    never_reached = tmp_path / "should_not_exist.pt"
    leg2_out = tmp_path / "leg2_final.pt"

    # Simulate a real interruption (e.g. a GH Actions timeout): start a
    # long-target training run, wait for its first periodic checkpoint to
    # land on disk, then kill the process -- an actual mid-training cut,
    # not a run that simply reached a small --steps target on its own.
    proc = subprocess.Popen(
        [sys.executable, str(TRAIN_SCRIPT),
         "--train", str(train_path), "--val", str(val_path),
         "--steps", "100000", "--save-every", "20",
         "--resume-out", str(resume_ckpt), "--out", str(never_reached),
         "--device", "cpu", *COMMON_ARGS],
        cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        for _ in range(200):  # poll up to ~20s for the file to first appear
            if resume_ckpt.exists():
                break
            time.sleep(0.1)
        else:
            raise AssertionError("resume checkpoint was never written within the timeout")

        # torch.save() isn't atomic -- a file that merely exists() may
        # still be mid-write. Only kill once its size has stopped
        # changing across a few checks, so we don't truncate it.
        last_size = -1
        stable_checks = 0
        for _ in range(100):  # up to ~10s
            size = resume_ckpt.stat().st_size
            if size == last_size and size > 0:
                stable_checks += 1
                if stable_checks >= 3:
                    break
            else:
                stable_checks = 0
            last_size = size
            time.sleep(0.1)
        else:
            raise AssertionError("resume checkpoint never stabilized (still being written)")
    finally:
        proc.kill()
        proc.wait()

    assert resume_ckpt.exists()
    assert not never_reached.exists()  # genuinely never reached its own target
    checkpoint = torch.load(resume_ckpt, map_location="cpu")
    interrupted_step = checkpoint["step"]
    assert interrupted_step < 100000
    assert "optimizer" in checkpoint

    # Resume and finish training for real, to a small, quickly-reachable target.
    result = _run([
        "--train", str(train_path), "--val", str(val_path),
        "--steps", str(interrupted_step + 3), "--resume", str(resume_ckpt),
        "--out", str(leg2_out),
        "--device", "cpu", *COMMON_ARGS,
    ], cwd=tmp_path)

    assert "resumed from" in result.stdout
    assert leg2_out.exists()
    final = torch.load(leg2_out, map_location="cpu")
    assert final["step"] == interrupted_step + 3
    assert "optimizer" not in final  # final checkpoints stay lean


def test_resume_with_nothing_left_to_do_is_a_noop(tmp_path):
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _make_corpus(train_path, 500)
    _make_corpus(val_path, 200)
    out = tmp_path / "final.pt"

    _run([
        "--train", str(train_path), "--val", str(val_path),
        "--steps", "2", "--out", str(out),
        "--device", "cpu", *COMMON_ARGS,
    ], cwd=tmp_path)
    assert out.exists()

    # A completed run's final checkpoint is itself a valid --resume source
    # (it carries "step" and "config" too), so resuming past-target from
    # it should be recognized as already-done rather than erroring.
    result = _run([
        "--train", str(train_path), "--val", str(val_path),
        "--steps", "1", "--resume", str(out),
        "--device", "cpu", *COMMON_ARGS,
    ], cwd=tmp_path)
    assert "nothing to do" in result.stdout


def test_resume_rejects_mismatched_architecture(tmp_path):
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _make_corpus(train_path, 500)
    _make_corpus(val_path, 200)
    out = tmp_path / "final.pt"

    _run([
        "--train", str(train_path), "--val", str(val_path),
        "--steps", "2", "--out", str(out),
        "--device", "cpu", *COMMON_ARGS,
    ], cwd=tmp_path)
    assert out.exists()

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT),
         "--train", str(train_path), "--val", str(val_path),
         "--steps", "4", "--resume", str(out),
         "--device", "cpu",
         "--n-layer", "2", "--n-head", "2", "--n-embd", "16", "--block-size", "8"],  # n-layer differs
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "does not match" in (result.stdout + result.stderr)


def test_final_checkpoint_removes_stale_resume_checkpoint(tmp_path):
    # Regression test: a completed run's periodic --resume-out checkpoint
    # (from a step before the final one) must not linger after the final
    # checkpoint is written -- otherwise a downstream script that checks
    # "does a resume checkpoint exist?" to decide whether to resume would
    # find the stale one and needlessly redo already-finished training.
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _make_corpus(train_path, 500)
    _make_corpus(val_path, 200)
    resume_ckpt = tmp_path / "resume.pt"
    out = tmp_path / "final.pt"

    _run([
        "--train", str(train_path), "--val", str(val_path),
        "--steps", "4", "--save-every", "2",
        "--resume-out", str(resume_ckpt), "--out", str(out),
        "--device", "cpu", *COMMON_ARGS,
    ], cwd=tmp_path)

    assert out.exists()
    assert not resume_ckpt.exists()  # superseded, must be cleaned up
