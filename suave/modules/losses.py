"""Loss utilities for TabVAE models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from ..models.tabvae import AnnealSchedule

def gaussian_nll(
    mu: torch.Tensor, log_sigma: torch.Tensor, x: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Negative log-likelihood of a diagonal Gaussian.

    Parameters
    ----------
    mu, log_sigma:
        Predicted mean and log standard deviation.
    x:
        Target values.
    mask:
        Binary mask where ``1`` denotes observed entries.
    """

    var = torch.exp(log_sigma) ** 2
    log_prob = 0.5 * ((x - mu) ** 2 / var + 2 * log_sigma + math.log(2 * math.pi))
    nll = (log_prob * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return nll.mean()


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence between ``N(mu, sigma)`` and ``N(0, I)``."""

    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def linear_anneal(step: int, schedule: "AnnealSchedule") -> float:
    """Linearly anneal ``schedule.start`` to ``schedule.end`` over ``schedule.epochs``."""

    total = max(schedule.epochs, 1)
    step = min(step, total)
    return schedule.start + (schedule.end - schedule.start) * (step / total)

