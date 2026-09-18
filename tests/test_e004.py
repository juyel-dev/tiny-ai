import struct

from tiny_ai.bpe_tokenizer import EOS_TOKEN, E004BPETokenizer, build_tokenizer
from tiny_ai.config import ModelConfig
from tiny_ai.data import MappedUInt16Tokens, get_batch_u16
from tiny_ai.model import TinyTransformer, fp32_weight_size_bytes, parameter_count


def make_tiny_bpe():
    tokenizer, trainer = build_tokenizer(vocab_size=300, min_frequency=1)
    tokenizer.train_from_iterator(
        ["hello world", "hello tiny model", "a small test"],
        trainer=trainer,
    )
    return E004BPETokenizer(tokenizer)


def test_bpe_round_trip_and_special_tokens():
    tokenizer = make_tiny_bpe()
    text = "Hello tiny model!"
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == text
    assert tokenizer.tokenizer.token_to_id(EOS_TOKEN) == tokenizer.eos_id


def test_e004_model_size():
    cfg = ModelConfig(vocab_size=2048, block_size=256, n_layer=4, n_head=6, n_embd=192)
    model = TinyTransformer(cfg)
    assert parameter_count(model) == 2_218_752
    assert fp32_weight_size_bytes(model) == 2_218_752 * 4


def test_u16_mapped_tokens(tmp_path):
    path = tmp_path / "tokens.u16"
    values = [0, 1, 2, 1000, 2047, 3, 4]
    path.write_bytes(struct.pack("<7H", *values))
    with MappedUInt16Tokens(path) as tokens:
        assert len(tokens) == len(values)
        assert [pair[0] for pair in tokens.read(0, 3)] == values[:3]
        x, y = get_batch_u16(tokens, block_size=4, batch_size=2, device="cpu")
        assert x.shape == (2, 4)
        assert y.shape == (2, 4)
