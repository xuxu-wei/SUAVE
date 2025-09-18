"""Evaluation helpers for SUAVE."""

from __future__ import annotations

from typing import Dict

import numpy as np


def evaluate_classification(
    probabilities: np.ndarray, targets: np.ndarray
) -> Dict[str, float]:
    """Return dummy metrics for the minimal implementation."""

    if probabilities.shape[0] != len(targets):
        raise ValueError("probabilities and targets must share the first dimension")
    return {"accuracy": float(np.mean(targets == targets))}
