"""Loss utilities for SUAVE models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from ..models.suave import AnnealSchedule


def _rbf_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    """Compute an RBF kernel matrix between ``x`` and ``y``.

    The implementation is intentionally simple because batch sizes in the
    accompanying tests are small (full-batch optimisation on toy datasets).
    """

    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    diff = x[:, None, :] - y[None, :, :]
    dist_sq = diff.pow(2).sum(dim=-1)
    return torch.exp(-dist_sq / (2 * bandwidth**2))


def maximum_mean_discrepancy(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    kernel: str = "rbf",
    bandwidth: float = 1.0,
) -> torch.Tensor:
    """Unbiased estimate of the Maximum Mean Discrepancy (MMD).

    The MMD penalty is used by InfoVAE variants to align the aggregated
    posterior ``q(z)`` with the prior ``p(z)``.  The function returns a scalar
    tensor representing ``MMD^2`` between the two sets of samples.
    """

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("MMD expects 2-D tensors")

    if min(x.size(0), y.size(0)) < 2:
        return torch.zeros(1, device=x.device, dtype=x.dtype)

    if kernel != "rbf":
        raise NotImplementedError(f"Unsupported kernel: {kernel}")

    k_xx = _rbf_kernel(x, x, bandwidth)
    k_yy = _rbf_kernel(y, y, bandwidth)
    k_xy = _rbf_kernel(x, y, bandwidth)

    n_x = x.size(0)
    n_y = y.size(0)

    sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n_x * (n_x - 1))
    sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (n_y * (n_y - 1))
    sum_xy = k_xy.mean()

    return sum_xx + sum_yy - 2 * sum_xy

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

