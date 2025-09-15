"""Simple training helper for :class:`~suave.models.suave.SUAVE`."""

from __future__ import annotations

import numpy as np

from ..api import SUAVE


def train_suave(X: np.ndarray, y: np.ndarray, epochs: int = 20) -> SUAVE:
    """Train a :class:`SUAVE` model on the provided data."""

    model = SUAVE(input_dim=X.shape[1])
    model.fit(X, y, epochs=epochs)
    return model
