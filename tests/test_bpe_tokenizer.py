from tiny_ai.bpe_tokenizer import BPETokenizer, train_bpe

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox runs. the dog sleeps. "
    "once upon a time there was a small dog and a small fox. "
) * 5


def test_train_bpe_learns_requested_number_of_merges():
    tok = train_bpe(CORPUS, vocab_size=300)
    assert len(tok.merges) == 300 - 256 - 3
    assert tok.vocab_size == 300


def test_train_bpe_rejects_too_small_vocab():
    try:
        train_bpe(CORPUS, vocab_size=259)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for vocab_size <= 259")


def test_bpe_roundtrip_ascii():
    tok = train_bpe(CORPUS, vocab_size=300)
    text = "the quick brown fox"
    assert tok.decode(tok.encode(text)) == text


def test_bpe_roundtrip_unicode():
    tok = train_bpe(CORPUS, vocab_size=300)
    text = "বাংলা fox test!"
    assert tok.decode(tok.encode(text)) == text


def test_bpe_roundtrip_unseen_text():
    # Bytes not covered by any learned merge must still fall back losslessly.
    tok = train_bpe(CORPUS, vocab_size=280)
    text = "zzzzz qwerty 12345 @#$%"
    assert tok.decode(tok.encode(text)) == text


def test_bpe_save_and_load(tmp_path):
    tok = train_bpe(CORPUS, vocab_size=280)
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)

    text = "the lazy dog"
    assert loaded.merges == tok.merges
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(tok.encode(text)) == text


def test_bpe_produces_fewer_tokens_than_raw_bytes():
    tok = train_bpe(CORPUS, vocab_size=300)
    text = "the quick brown fox jumps over the lazy dog"
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_bpe_special_tokens_are_last_three_ids():
    tok = train_bpe(CORPUS, vocab_size=300)
    assert tok.pad_id == tok.vocab_size - 3
    assert tok.bos_id == tok.vocab_size - 2
    assert tok.eos_id == tok.vocab_size - 1


def test_bpe_add_bos_eos():
    tok = train_bpe(CORPUS, vocab_size=280)
    ids = tok.encode("dog", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
