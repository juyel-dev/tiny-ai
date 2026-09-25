from tiny_ai.banglish import PROMPT_TEMPLATE, build_dataset, build_example, levenshtein, load_pairs
from tiny_ai.bpe_tokenizer import train_bpe

CORPUS = (
    "BN: ami tomake bhalobashi\nBD: আমি তোমাকে ভালোবাসি\n"
    "BN: tumi kemon acho\nBD: তুমি কেমন আছো\n"
) * 20


def _tok():
    return train_bpe(CORPUS, vocab_size=280)


def test_load_pairs_parses_tsv(tmp_path):
    path = tmp_path / "pairs.tsv"
    path.write_text("ami bhalo\tআমি ভালো\nkhub bhalo\tখুব ভালো\n", encoding="utf-8")
    pairs = load_pairs(str(path))
    assert pairs == [("ami bhalo", "আমি ভালো"), ("khub bhalo", "খুব ভালো")]


def test_load_pairs_skips_blank_and_malformed_lines(tmp_path):
    path = tmp_path / "pairs.tsv"
    path.write_text("ok\tঠিক\n\nno_tab_here\n", encoding="utf-8")
    pairs = load_pairs(str(path))
    assert pairs == [("ok", "ঠিক")]


def test_build_example_masks_prompt_and_padding():
    tok = _tok()
    banglish, bangla = "ami tomake bhalobashi", "আমি তোমাকে ভালোবাসি"
    block_size = 64
    x, y = build_example(tok, banglish, bangla, block_size)

    assert len(x) == block_size
    assert len(y) == block_size

    prompt_ids = [tok.bos_id] + tok.encode(PROMPT_TEMPLATE.format(banglish=banglish))
    target_ids = tok.encode(bangla) + [tok.eos_id]
    content_len = len(prompt_ids) + len(target_ids)

    # Positions predicting a prompt token, or predicting padding, must be masked.
    for i in range(block_size):
        j = i + 1
        if j < len(prompt_ids) or j >= content_len:
            assert y[i] == -100, f"position {i} (predicts index {j}) should be masked"
        else:
            assert y[i] != -100, f"position {i} (predicts index {j}) should NOT be masked"

    # The unmasked target ids, in order, must reconstruct exactly the
    # Bangla completion + EOS -- not the prompt, not garbage.
    unmasked = [v for v in y if v != -100]
    assert unmasked == target_ids


def test_build_example_drops_too_long_rather_than_truncating():
    tok = _tok()
    long_bangla = "আমি তোমাকে ভালোবাসি " * 50  # far exceeds any reasonable block_size
    result = build_example(tok, "short banglish", long_bangla, block_size=16)
    assert result is None


def test_build_dataset_counts_dropped():
    tok = _tok()
    pairs = [
        ("ami tomake bhalobashi", "আমি তোমাকে ভালোবাসি"),
        ("x", "আমি তোমাকে ভালোবাসি " * 50),  # will be dropped at this block_size
    ]
    (xs, ys), dropped = build_dataset(tok, pairs, block_size=64)
    assert dropped == 1
    assert len(xs) == 1
    assert len(ys) == 1


def test_levenshtein_basic():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("kitten", "sitting") == 3


def test_levenshtein_bengali_text():
    a = "আমি তোমাকে ভালোবাসি"
    b = "আমি তোমাকে ভালোবাসি"
    assert levenshtein(a, b) == 0
    c = "আমি তোমাকে ভালোবাসি না"
    assert levenshtein(a, c) > 0


def test_beam_search_width_1_matches_greedy():
    # With beam_width=1 there's no alternative path to ever consider --
    # it must reduce to exactly the same deterministic argmax choices as
    # greedy decoding, token for token.
    import torch
    from tiny_ai.banglish import beam_search_generate
    from tiny_ai.config import ModelConfig
    from tiny_ai.model import TinyTransformer

    tok = _tok()
    cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=48, n_layer=2, n_head=2, n_embd=16)
    model = TinyTransformer(cfg)
    model.eval()

    banglish = "tumi kemon acho"

    beam_out = beam_search_generate(model, tok, PROMPT_TEMPLATE, banglish, max_new_tokens=20, beam_width=1)

    # Reproduce plain greedy decoding by hand for comparison.
    prompt_ids = [tok.bos_id] + tok.encode(PROMPT_TEMPLATE.format(banglish=banglish))
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        for _ in range(20):
            logits, _ = model(idx[:, -cfg.block_size:])
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, next_id), dim=1)
            if next_id.item() == tok.eos_id or idx.shape[1] >= cfg.block_size:
                break
    generated = idx[0, len(prompt_ids):].tolist()
    if generated and generated[-1] == tok.eos_id:
        generated = generated[:-1]
    greedy_out = tok.decode(generated)

    assert beam_out == greedy_out


def test_beam_search_wider_beam_runs_and_returns_string():
    import torch
    from tiny_ai.banglish import beam_search_generate
    from tiny_ai.config import ModelConfig
    from tiny_ai.model import TinyTransformer

    tok = _tok()
    cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=48, n_layer=2, n_head=2, n_embd=16)
    model = TinyTransformer(cfg)
    model.eval()

    out = beam_search_generate(model, tok, PROMPT_TEMPLATE, "ami tomake bhalobashi",
                                max_new_tokens=20, beam_width=4)
    assert isinstance(out, str)
