"""Calibration utilities including temperature scaling and ECE."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """A simple temperature scaling model for calibration."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> "TemperatureScaler":
        """Fit temperature using negative log-likelihood on validation data."""
        self.train()
        labels = labels.to(logits.device)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure() -> torch.Tensor:  # type: ignore[override]
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def predict_proba(self, logits: torch.Tensor) -> np.ndarray:
        """Return calibrated probabilities."""
        with torch.no_grad():
            scaled = self.forward(logits)
            return torch.softmax(scaled, dim=-1).cpu().numpy()


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute the Expected Calibration Error (ECE).

    Parameters
    ----------
    probs:
        Predicted probabilities for the positive class.
    labels:
        Ground truth binary labels.
    n_bins:
        Number of equally spaced bins.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(probs, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = inds == i
        if not np.any(mask):
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += np.abs(acc - conf) * (mask.sum() / len(probs))
    return float(ece)


def calibration_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return data necessary for plotting a calibration curve.

    Parameters
    ----------
    probs:
        Predicted probabilities for the positive class.
    labels:
        Ground truth binary labels.
    n_bins:
        Number of equal-width bins over ``[0, 1]``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Bin centers, empirical accuracy per bin and mean predicted probability
        per bin.  Bins with no samples receive ``nan`` values.
    """

    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(probs, bins) - 1
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    acc = np.full(n_bins, np.nan)
    conf = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = inds == i
        if np.any(mask):
            acc[i] = labels[mask].mean()
            conf[i] = probs[mask].mean()
    return bin_centers, acc, conf
