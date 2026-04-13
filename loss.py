from __future__ import annotations

from torch import nn


def get_loss(mode: str, epsilon: float = 0.1) -> nn.Module:
    if mode == "baseline":
        return nn.CrossEntropyLoss()
    if mode == "ls":
        return nn.CrossEntropyLoss(label_smoothing=epsilon)
    raise ValueError(f"Unsupported mode: {mode}. Expected 'baseline' or 'ls'.")
