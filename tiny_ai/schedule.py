"""Learning-rate schedules.

E002's original training loop used a constant learning rate for the
entire run. That's simple, but for a from-scratch model it usually
means: (a) early steps take gradient steps that are too large before
the model has any structure, and (b) the model never settles into a
low-loss basin at the end because the LR never decays. Linear warmup +
cosine decay is the standard fix and costs nothing to add.

Implemented as a pure function (no optimizer/scheduler object) so it's
trivial to unit test and to reason about independently of the training
loop.
"""
from __future__ import annotations

import math


def warmup_cosine_lr(
    step: int,
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Return the learning rate for a 1-indexed training ``step``.

    - Steps ``1..warmup_steps``: linear ramp from 0 to ``base_lr``.
    - Steps after warmup: cosine decay from ``base_lr`` down to
      ``min_lr_ratio * base_lr`` by ``total_steps``.
    - Steps at or beyond ``total_steps``: held at ``min_lr_ratio * base_lr``.
    """
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    if step >= total_steps or total_steps <= warmup_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)
