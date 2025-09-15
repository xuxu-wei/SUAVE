"""Latent variable interpretation helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import spearmanr


def latent_feature_correlation(Z: np.ndarray, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return Spearman correlation matrix and p-values between ``Z`` and ``features``."""
    corr, pval = spearmanr(Z, features, axis=0)
    return corr, pval
