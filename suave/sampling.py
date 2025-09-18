"""Sampling utilities for SUAVE."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample(
    n_samples: int,
    n_features: int,
    conditional: bool = False,
    y: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return a dataframe of zeros as a placeholder sample."""

    data = np.zeros((n_samples, n_features))
    columns = [f"feature_{idx}" for idx in range(n_features)]
    samples = pd.DataFrame(data, columns=columns)
    if conditional and y is not None:
        samples["condition"] = y
    return samples
