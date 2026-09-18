from __future__ import annotations

from dataclasses import dataclass

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


@dataclass
class E004BPETokenizer:
    tokenizer: Tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self.tokenizer.token_to_id(PAD_TOKEN)

    @property
    def bos_id(self) -> int:
        return self.tokenizer.token_to_id(BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self.tokenizer.token_to_id(EOS_TOKEN)

    @classmethod
    def from_file(cls, path: str) -> "E004BPETokenizer":
        return cls(Tokenizer.from_file(path))

    @classmethod
    def from_json(cls, data: str) -> "E004BPETokenizer":
        return cls(Tokenizer.from_str(data))

    def to_json(self) -> str:
        return self.tokenizer.to_str()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        special = {self.pad_id, self.bos_id, self.eos_id, self.tokenizer.token_to_id(UNK_TOKEN)}
        ids = [i for i in ids if i not in special]
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def build_tokenizer(vocab_size: int = 2048, min_frequency: int = 2) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN],
        initial_alphabet=ByteLevel.alphabet(),
    )
    return tokenizer, trainer
