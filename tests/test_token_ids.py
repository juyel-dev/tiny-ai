import torch

from tiny_ai.data import MappedTokenIds, get_batch, write_token_ids


def test_write_and_read_token_ids_roundtrip(tmp_path):
    path = tmp_path / "tokens.bin"
    ids = [3, 40000, 259, 65535, 0, 1000]
    write_token_ids(ids, path)

    with MappedTokenIds(path) as tokens:
        assert len(tokens) == len(ids)
        assert [tokens[i] for i in range(len(ids))] == ids
        assert tokens[1:4] == ids[1:4]


def test_write_token_ids_rejects_out_of_range_id(tmp_path):
    path = tmp_path / "tokens.bin"
    try:
        write_token_ids([70000], path)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for id >= 65536")


def test_get_batch_from_mapped_token_ids(tmp_path):
    path = tmp_path / "tokens.bin"
    ids = list(range(0, 640, 10))  # values well above the byte range
    write_token_ids(ids, path)

    with MappedTokenIds(path) as tokens:
        x, y = get_batch(tokens, block_size=8, batch_size=4, device="cpu")

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert x.max().item() >= 256  # would be impossible with byte-range data
