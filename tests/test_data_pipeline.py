from pathlib import Path

from scripts.prepare_corpus import iter_records, normalize, sha256_text


def test_normalize_is_deterministic():
    assert normalize("  hello\r\n\r\n world  ") == "hello\n\n world"


def test_hash_changes_with_content():
    assert sha256_text("a") != sha256_text("b")


def test_iter_records_preserves_paragraphs(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("a\n\nb\n\n a \n", encoding="utf-8")
    assert list(iter_records(source)) == ["a", "b", "a"]


def test_manifest_sample_exists():
    assert Path("data/manifests/tinystories.yaml").exists()
