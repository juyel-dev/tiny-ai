from __future__ import annotations

import array
import mmap
import struct
import sys
from pathlib import Path

import torch

# array.array('H', ...).tofile() writes in native byte order. Every
# platform this project targets (x86_64/ARM64 Linux, incl. GitHub Actions
# runners) is little-endian, matching the explicit "<H" reads in
# MappedTokenIds below -- fail loudly instead of silently writing a file
# MappedTokenIds would misread.
assert sys.byteorder == "little", (
    "write_token_ids assumes a little-endian platform to match MappedTokenIds' "
    "explicit little-endian reads; this platform is big-endian."
)


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


class MappedTokenIds:
    """Disk-backed token-id sequence for vocabularies larger than 256.

    ``MappedTokens`` stores one byte per token, which only works for the
    raw-byte tokenizer (vocab_size <= 256). A BPE tokenizer's vocabulary
    is larger than one byte can address, so token ids are stored instead
    as little-endian uint16 (2 bytes each, max vocab 65,536 -- far above
    anything this repo trains). Still mmap-backed, so training doesn't
    need to load the whole corpus into memory.
    """

    ITEM_SIZE = 2
    MAX_VOCAB_SIZE = 1 << 16

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("rb")
        size_bytes = self.path.stat().st_size
        if size_bytes == 0:
            self._file.close()
            raise ValueError(f"Token file is empty: {self.path}")
        if size_bytes % self.ITEM_SIZE != 0:
            self._file.close()
            raise ValueError(
                f"Token file size ({size_bytes} bytes) is not a multiple of "
                f"{self.ITEM_SIZE} (expected uint16 token ids): {self.path}"
            )
        self._size = size_bytes // self.ITEM_SIZE
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._size)
            raw = self._map[start * self.ITEM_SIZE:stop * self.ITEM_SIZE]
            values = list(struct.unpack(f"<{stop - start}H", raw))
            return values[::step] if step != 1 else values
        raw = self._map[index * self.ITEM_SIZE:(index + 1) * self.ITEM_SIZE]
        return struct.unpack("<H", raw)[0]

    def close(self) -> None:
        if getattr(self, "_map", None) is not None:
            self._map.close()
            self._map = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "MappedTokenIds":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def write_token_ids(ids, path: str | Path) -> None:
    """Write token ids to disk as little-endian uint16, for MappedTokenIds.

    Uses array.array rather than struct.pack(f"<{n}H", *ids) -- unpacking
    a huge list as function arguments is memory-hungry, and array.array
    stores ids in a compact C buffer instead of one Python int object
    each. Still materializes the whole list at once, so for a corpus too
    large to hold in memory, encode and write in chunks instead (see
    scripts/tokenize_corpus.py).
    """
    arr = array.array("H")
    try:
        arr.extend(ids)
    except OverflowError as e:
        raise ValueError("Token id exceeds uint16 range; MappedTokenIds cannot store it.") from e
    with Path(path).open("wb") as f:
        arr.tofile(f)


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
        chunk = tokens[start:start + block_size + 1]
        # MappedTokens (byte tokenizer) yields a bytes-like slice;
        # MappedTokenIds (BPE tokenizer) yields a list[int] already.
        chunk = list(chunk) if not isinstance(chunk, list) else chunk
        xs.append(torch.tensor(chunk[:-1], dtype=torch.long))
        ys.append(torch.tensor(chunk[1:], dtype=torch.long))

    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    return x, y
