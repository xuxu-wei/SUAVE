"""Loss utilities for TabVAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_nll(x_hat: torch.Tensor, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Return a mean squared error based reconstruction negative log-likelihood."""
    return F.mse_loss(x_hat, x, reduction=reduction)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence between ``N(mu, sigma)`` and ``N(0, I)``."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def kl_anneal_weight(step: int, start: float, end: float, total_steps: int) -> float:
    """Linearly anneal a weight from ``start`` to ``end`` over ``total_steps``."""
    if total_steps <= 0:
        return end
    step = min(step, total_steps)
    return start + (end - start) * (step / total_steps)
