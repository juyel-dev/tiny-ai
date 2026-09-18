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

    if isinstance(tokens, torch.Tensor):
        x = torch.stack([tokens[int(i):int(i) + block_size] for i in starts])
        y = torch.stack([tokens[int(i) + 1:int(i) + block_size + 1] for i in starts])
        return x.to(device), y.to(device)

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


class MappedUInt16Tokens:
    """Disk-backed unsigned-16-bit token ID sequence for vocabularies up to 65535."""

    def __init__(self, path: str | Path):
        import struct

        self.path = Path(path)
        self._file = self.path.open("rb")
        self._size_bytes = self.path.stat().st_size
        if self._size_bytes == 0 or self._size_bytes % 2:
            self._file.close()
            raise ValueError(f"Invalid uint16 token file: {self.path}")
        self._count = self._size_bytes // 2
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._struct = struct.Struct("<H")

    def __len__(self) -> int:
        return self._count

    def read(self, start: int, count: int) -> list[int]:
        if start < 0 or count < 0 or start + count > self._count:
            raise IndexError("Token slice is out of bounds.")
        offset = start * 2
        raw = self._map[offset:offset + count * 2]
        return list(__import__("struct").iter_unpack("<H", raw))

    def close(self) -> None:
        if getattr(self, "_map", None) is not None:
            self._map.close()
            self._map = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "MappedUInt16Tokens":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def get_batch_u16(tokens, block_size: int, batch_size: int, device: str):
    if len(tokens) < block_size + 1:
        raise ValueError("Training tokens must contain at least block_size + 1 tokens.")

    import torch

    max_start = len(tokens) - block_size
    starts = torch.randint(0, max_start, (batch_size,))
    xs, ys = [], []
    for start in starts.tolist():
        start = int(start)
        chunk = tokens.read(start, block_size + 1)
        values = [item[0] for item in chunk]
        xs.append(torch.tensor(values[:-1], dtype=torch.long))
        ys.append(torch.tensor(values[1:], dtype=torch.long))
    return torch.stack(xs).to(device), torch.stack(ys).to(device)
