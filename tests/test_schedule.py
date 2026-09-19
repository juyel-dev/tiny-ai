from tiny_ai.schedule import warmup_cosine_lr


def test_warmup_ramps_linearly_to_base_lr():
    lrs = [warmup_cosine_lr(s, base_lr=1.0, warmup_steps=10, total_steps=100) for s in range(1, 11)]
    assert lrs == sorted(lrs)
    assert lrs[-1] == 1.0
    assert lrs[0] == 0.1


def test_decays_to_min_lr_ratio_by_total_steps():
    lr = warmup_cosine_lr(100, base_lr=2.0, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    assert abs(lr - 0.2) < 1e-9


def test_held_at_min_lr_ratio_past_total_steps():
    lr = warmup_cosine_lr(500, base_lr=2.0, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    assert abs(lr - 0.2) < 1e-9


def test_monotonic_decay_after_warmup():
    lrs = [warmup_cosine_lr(s, base_lr=1.0, warmup_steps=10, total_steps=100) for s in range(10, 101)]
    assert all(a >= b - 1e-12 for a, b in zip(lrs, lrs[1:]))


def test_zero_warmup_steps_starts_near_base_lr():
    # With no warmup, cosine decay begins immediately at step 1, so the
    # first step is very close to (not exactly) base_lr.
    lr = warmup_cosine_lr(1, base_lr=1.0, warmup_steps=0, total_steps=100)
    assert 0.99 < lr <= 1.0
