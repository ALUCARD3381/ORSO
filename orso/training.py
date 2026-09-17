"""Training utilities for ORSO Phase 5/6."""

from __future__ import annotations

import math

from orso_core import AdamW, cross_entropy


class CosineScheduler:
    def __init__(self, optimizer: AdamW, total_steps: int, warmup_steps: int = 0, min_lr: float = 0.0):
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if warmup_steps < 0 or warmup_steps > total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps]")
        if min_lr < 0:
            raise ValueError("min_lr must be >= 0")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.base_lr = optimizer.lr
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        if self.step_count <= self.warmup_steps and self.warmup_steps:
            scale = self.step_count / self.warmup_steps
            lr = self.base_lr * scale
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
        self.optimizer.lr = max(self.min_lr if self.min_lr > 0 else 1.0e-12, lr)
        return self.optimizer.lr

    def state_dict(self) -> dict[str, float | int]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
            "base_lr": self.base_lr,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.total_steps = int(state["total_steps"])
        self.warmup_steps = int(state["warmup_steps"])
        self.min_lr = float(state["min_lr"])
        self.base_lr = float(state["base_lr"])
        self.step_count = int(state["step_count"])


class Trainer:
    def __init__(self, model, optimizer: AdamW, scheduler: CosineScheduler | None = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.steps = 0

    def train_batch(self, inputs: list[list[int]], targets: list[list[int]]) -> dict[str, float]:
        self.optimizer.zero_grad()
        logits = self.model.forward(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        grad_norm = self.optimizer.step()
        self.steps += 1
        lr = self.optimizer.lr
        if self.scheduler is not None:
            lr = self.scheduler.step()
        return {"loss": loss.item(), "grad_norm": grad_norm, "lr": lr, "step": float(self.steps)}

    def state_dict(self) -> dict[str, int]:
        return {"steps": int(self.steps)}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.steps = int(state["steps"])
