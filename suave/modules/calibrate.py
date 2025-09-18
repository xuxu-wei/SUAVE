"""Calibration utilities for SUAVE."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class TemperatureScaler:
    """Temperature scaling module optimised via gradient descent."""

    temperature: float = 1.0

    def fit(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        *,
        max_iter: int = 100,
        learning_rate: float = 0.01,
    ) -> "TemperatureScaler":
        """Optimise the temperature on held-out logits.

        Parameters
        ----------
        logits:
            Array with shape ``(n_samples, n_classes)`` containing the model
            logits on calibration data.
        targets:
            Integer encoded labels aligned with ``logits``.
        max_iter:
            Number of optimisation steps used by the Adam optimiser.
        learning_rate:
            Learning rate controlling the optimiser step size.
        """

        logits = np.asarray(logits, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.int64)
        if logits.ndim != 2:
            raise ValueError("logits must be a 2D array")
        if logits.shape[0] != targets.shape[0]:
            raise ValueError("logits and targets must have matching rows")
        if targets.size == 0:
            raise ValueError("calibration targets must be non-empty")

        logits_tensor = torch.from_numpy(logits)
        targets_tensor = torch.from_numpy(targets)

        log_temperature = torch.tensor(
            math.log(max(self.temperature, 1e-3)),
            dtype=logits_tensor.dtype,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([log_temperature], lr=learning_rate)
        for _ in range(max_iter):
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature)
            scaled_logits = logits_tensor / temperature
            loss = F.cross_entropy(scaled_logits, targets_tensor)
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimizer.step()

        temperature = torch.exp(log_temperature.detach()).clamp(min=1e-3, max=1e3)
        self.temperature = float(temperature.item())
        return self

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        return logits / self.temperature
