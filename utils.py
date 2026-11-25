"""Metrics, seeding and post-hoc decision-bias tuning."""
from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from dataset import CLASSES, NUM_CLASSES


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    per_class = f1_score(labels, preds, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    return {
        "acc": float((labels == preds).mean()),
        "macro_f1": float(per_class.mean()),
        **{f"f1_{c}": float(f) for c, f in zip(CLASSES, per_class)},
    }


def format_confusion(labels: np.ndarray, preds: np.ndarray) -> str:
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    w = max(len(c) for c in CLASSES) + 2
    lines = ["true \\ pred".ljust(w) + "".join(c.rjust(w) for c in CLASSES)]
    for c, row in zip(CLASSES, cm):
        lines.append(c.ljust(w) + "".join(str(v).rjust(w) for v in row))
    return "\n".join(lines)


def tune_class_bias(probs: np.ndarray, labels: np.ndarray, grid=None, rounds: int = 3):
    """Coordinate ascent on per-class additive log-prob offsets to maximise macro-F1.

    Class 0 is the reference (offset 0). Beware: with few COVID-19 samples this can
    overfit the validation set -- prefer tuning on pooled out-of-fold predictions.
    """
    if grid is None:
        grid = np.round(np.linspace(-2, 2, 41), 2)
    logp = np.log(np.clip(probs, 1e-8, 1.0))
    bias = np.zeros(probs.shape[1])

    def score(b):
        return f1_score(labels, (logp + b).argmax(1), average="macro",
                        labels=list(range(NUM_CLASSES)), zero_division=0)

    base = best = score(bias)
    for _ in range(rounds):
        improved = False
        for c in range(1, probs.shape[1]):
            for g in grid:
                trial = bias.copy()
                trial[c] = g
                s = score(trial)
                if s > best + 1e-9:
                    best, bias, improved = s, trial, True
        if not improved:
            break
    return bias, base, best
