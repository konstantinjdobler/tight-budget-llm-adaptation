import math

import torch


def get_lr_with_cosine_schedule(it, learning_rate, warmup_period, lr_decay_period, min_lr):
    """
    Returns actual lr based on current iteration. Should be called every iteration.
    From lit-gpt: https://github.com/Lightning-AI/lit-gpt/blob/a21d46ae80f84c350ad871578d0348b470c83021/pretrain/redpajama.py#L301
    """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_period:
        return learning_rate * it / warmup_period
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_period:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_period) / (lr_decay_period - warmup_period)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


class LRScheduler:
    _step: int
    _lr: float

    def step(self, override_step=None) -> float:
        raise NotImplementedError

class ConstantScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, lr: float) -> None:
        self.optimizer = optimizer
        self.lr = lr
        self._step = 1
        self._lr = self.optimizer.param_groups[0]["lr"]
    
    def __repr__(self):
        return f"ConstantScheduler({self.lr=})"
    
    def step(self, override_step=None):
        self._update_lr(self.lr)
        self._step += 1
        self._lr = self.lr
        return self.lr
    
    def _update_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


class InfiniteLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        constant_lr: float,
        min_lr: float,
        warmup_steps: int,
        cooldown_steps: int,
        annealing_steps: int,
        max_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.constant_lr = constant_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.annealing_steps = annealing_steps
        self.max_steps = max_steps
        self.constant_steps = max_steps - warmup_steps - cooldown_steps - annealing_steps
        assert self.constant_steps >= 0
        assert self.max_steps == self.warmup_steps + self.cooldown_steps + self.annealing_steps + self.constant_steps
        self._step = 1
        self._lr = self.optimizer.param_groups[0]["lr"]

    def __repr__(self):
        return f"InfiniteLRScheduler({self.max_lr=}, {self.constant_lr=}, {self.min_lr=}, {self.warmup_steps=}, {self.cooldown_steps=}, {self.annealing_steps=}, {self.max_steps=})"

    def step(self, override_step=None):
        current_step = override_step if override_step is not None else self._step

        cooldown_period_end = self.warmup_steps + self.cooldown_steps
        cooldown_period_start = self.warmup_steps

        constant_period_end = cooldown_period_end + self.constant_steps

        # (1) linear warmup to max_lr over warmup_steps
        if current_step <= self.warmup_steps:
            new_lr = self.max_lr * current_step / self.warmup_steps

        # (2) cooldown to constant_lr over cooldown_steps w/ cosine decay
        elif current_step <= cooldown_period_end:
            decay_ratio = (current_step - cooldown_period_start) / (self.cooldown_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            new_lr = self.constant_lr + coeff * (self.max_lr - self.constant_lr)

        # (3) constant_lr for as long as wanted until max_steps - annealing_steps
        elif current_step <= cooldown_period_end + self.constant_steps:
            new_lr = self.constant_lr

        # (4) annealing to min_lr over annealing_steps
        elif current_step <= self.max_steps:
            decay_factor = self.min_lr / self.constant_lr
            decay_exponent = (current_step - constant_period_end) / self.annealing_steps
            new_lr = self.constant_lr * (decay_factor**decay_exponent)
        else:
            raise ValueError(
                f"Invalid step: {current_step} for given parameters {self.warmup_steps=}, {self.cooldown_steps=}, {self.annealing_steps=}, {self.constant_steps=}, {self.max_steps=}"
            )

        self._update_lr(new_lr)
        self._step += 1
        self._lr = new_lr
        return new_lr

    def _update_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


class CosineDecayScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, max_lr: float, min_lr: float, warmup_steps: int, decay_steps: int):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self._step = 1
        self._lr = self.optimizer.param_groups[0]["lr"]

    def __repr__(self):
        return f"CosineDecayScheduler({self.max_lr=}, {self.min_lr=}, {self.warmup_steps=}, {self.decay_steps=})"

    def step(self, override_step=None):
        current_step = override_step if override_step is not None else self._step

        # (1) linear warmup to max_lr over warmup_steps
        if current_step <= self.warmup_steps:
            new_lr = self.max_lr * current_step / self.warmup_steps

        # (2) cosine decay to min_lr over decay_steps
        elif current_step <= self.warmup_steps + self.decay_steps:
            decay_ratio = (current_step - self.warmup_steps) / self.decay_steps
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            new_lr = self.min_lr + coeff * (self.max_lr - self.min_lr)

        # (3) constant min_lr after decay_steps
        else:
            assert current_step > self.warmup_steps + self.decay_steps
            new_lr = self.min_lr

        self._update_lr(new_lr)
        self._step += 1
        self._lr = new_lr
        return new_lr

    def _update_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
