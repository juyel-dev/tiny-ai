from pathlib import Path
import torch

def load_text(path: str) -> torch.Tensor:
    from .tokenizer import ByteTokenizer
    text = Path(path).read_text(encoding="utf-8")
    return torch.tensor(ByteTokenizer().encode(text), dtype=torch.long)

def get_batch(tokens, block_size, batch_size, device):
    if len(tokens) <= block_size + 1:
        raise ValueError("Training text must contain more than block_size + 1 tokens.")
    starts = torch.randint(0, len(tokens) - block_size - 1, (batch_size,))
    x = torch.stack([tokens[i:i + block_size] for i in starts])
    y = torch.stack([tokens[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
