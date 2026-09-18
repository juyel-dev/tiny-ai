from dataclasses import dataclass

PAD = 256
BOS = 257
EOS = 258

@dataclass(frozen=True)
class ByteTokenizer:
    """Lossless UTF-8 byte tokenizer with three reserved special tokens."""

    vocab_size: int = 259

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, BOS)
        if add_eos:
            ids.append(EOS)
        return ids

    def decode(self, ids: list[int]) -> str:
        data = bytes(i for i in ids if 0 <= i <= 255)
        return data.decode("utf-8", errors="replace")
