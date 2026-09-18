from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 259
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
