import torch

from scripts.train_e003_4 import instruction_loss


def test_instruction_loss_ignores_prompt_and_padding():
    logits = torch.randn(2, 6, 259)
    targets = torch.tensor([
        [-100, -100, 10, 11, 258, -100],
        [-100, 20, 21, 22, 258, -100],
    ])
    loss = instruction_loss(
        type("M", (), {"__call__": lambda self, x: (logits, None)})(),
        torch.zeros((2, 6), dtype=torch.long),
        targets,
    )
    assert torch.isfinite(loss)
