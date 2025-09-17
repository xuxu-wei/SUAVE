"""Calibration utilities for SUAVE."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TemperatureScaler:
    """Simple temperature scaling stub that rescales logits."""

    temperature: float = 1.0

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        return logits / self.temperature
