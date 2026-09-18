from pathlib import Path

from scripts.prepare_corpus import normalize, sha256_text, split_records


def test_normalize_is_deterministic():
    assert normalize("  hello\r\n\r\n world  ") == "hello\n\n world"


def test_hash_changes_with_content():
    assert sha256_text("a") != sha256_text("b")


def test_split_records():
    assert split_records("a\n\nb\n\n a ") == ["a", "b", "a"]


def test_manifest_sample_exists():
    assert Path("data/manifests/tinystories.yaml").exists()
