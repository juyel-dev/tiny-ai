from __future__ import annotations

import mmap
from pathlib import Path

import torch


class MappedTokens:
    """Disk-backed byte-token sequence."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._size = self.path.stat().st_size
        if self._size == 0:
            self._file.close()
            raise ValueError(f"Token file is empty: {self.path}")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index):
        return self._map[index]

    def close(self) -> None:
        if getattr(self, "_map", None) is not None:
            self._map.close()
            self._map = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "MappedTokens":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def load_text(path: str) -> torch.Tensor:
    from .tokenizer import ByteTokenizer

    text = Path(path).read_text(encoding="utf-8")
    return torch.tensor(ByteTokenizer().encode(text), dtype=torch.long)


def get_batch(tokens, block_size: int, batch_size: int, device: str):
    if len(tokens) < block_size + 1:
        raise ValueError("Training text must contain at least block_size + 1 tokens.")

    max_start = len(tokens) - block_size
    starts = torch.randint(0, max_start, (batch_size,))

    xs = []
    ys = []
    for start in starts.tolist():
        start = int(start)
        chunk = bytes(tokens[start:start + block_size + 1])
        xs.append(torch.tensor(list(chunk[:-1]), dtype=torch.long))
        ys.append(torch.tensor(list(chunk[1:]), dtype=torch.long))

    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    return x, y
