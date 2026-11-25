"""Loss functions and class-imbalance helpers."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_counts(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.bincount(labels, minlength=num_classes).astype(np.float64)


def class_weights(labels: np.ndarray, num_classes: int, power: float = 0.5) -> torch.Tensor:
    """Inverse-frequency weights ** power, normalised so the average per-sample weight is 1.

    power=0 -> no weighting, 0.5 -> sqrt inverse frequency, 1 -> fully balanced.
    """
    counts = np.maximum(class_counts(labels, num_classes), 1)
    w = (counts.sum() / (num_classes * counts)) ** power
    w = w * counts.sum() / (w * counts).sum()
    return torch.tensor(w, dtype=torch.float32)


def sample_weights(labels: np.ndarray, num_classes: int, power: float) -> np.ndarray:
    """Per-sample weights for WeightedRandomSampler (power 0.5 = sqrt, 1 = balanced)."""
    counts = np.maximum(class_counts(labels, num_classes), 1)
    return (1.0 / counts ** power)[labels]


class SoftFocalLoss(nn.Module):
    """Class-weighted focal loss that also accepts soft targets (MixUp/CutMix).

    gamma=0 reduces to (weighted, label-smoothed) cross-entropy.
    """

    def __init__(self, num_classes: int, gamma: float = 0.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight if weight is not None else torch.ones(num_classes))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 1:
            target = F.one_hot(target, self.num_classes).float()
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        logp = F.log_softmax(logits.float(), dim=-1)
        loss = -target * logp
        if self.gamma > 0:
            loss = loss * (1 - logp.exp()).pow(self.gamma)
        loss = loss * self.weight
        return loss.sum(-1).mean()
