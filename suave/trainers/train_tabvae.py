"""Simple training helper for :class:`~suave.models.tabvae.TabVAEClassifier`."""

from __future__ import annotations

import numpy as np

from ..api import TabVAEClassifier


def train_tabvae(X: np.ndarray, y: np.ndarray, epochs: int = 20) -> TabVAEClassifier:
    """Train a :class:`TabVAEClassifier` on the provided data."""
    model = TabVAEClassifier(input_dim=X.shape[1])
    model.fit(X, y, epochs=epochs)
    return model
