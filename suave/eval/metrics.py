"""Basic classification metrics used in tests."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Return a dictionary of common classification metrics."""
    return {
        "auroc": roc_auc_score(y_true, proba),
        "auprc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "nll": log_loss(y_true, proba),
    }
