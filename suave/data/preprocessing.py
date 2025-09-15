"""Minimal preprocessing utilities for tests."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class TabularPreprocessor:
    """A light-weight wrapper around :class:`StandardScaler`."""

    scaler: StandardScaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "TabularPreprocessor":
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
