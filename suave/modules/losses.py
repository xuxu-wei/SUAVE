"""Loss helpers implementing HI-VAE style objectives."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor


def elbo(recon_terms: Iterable[Tensor], kl: Tensor) -> Tensor:
    """Aggregate per-column reconstruction terms and subtract ``kl``.

    Parameters
    ----------
    recon_terms:
        Iterable of tensors containing per-sample log-likelihoods for the
        observed entries.  They are summed element-wise before computing the
        final ELBO.
    kl:
        Tensor with the KL divergence per sample (already including the desired
        scaling factor, e.g. the warm-up ``beta``).
    """

    recon = torch.stack(tuple(recon_terms), dim=0).sum(dim=0)
    return recon - kl


def kl_warmup(step: int, total_steps: int, beta_target: float) -> float:
    """Linearly anneal the KL term from ``0`` to ``beta_target``."""

    if total_steps <= 0:
        return beta_target
    progress = min(max(step, 0), total_steps) / float(total_steps)
    return beta_target * progress


def kl_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL divergence between ``N(mu, exp(logvar))`` and ``N(0, 1)``."""

    var = torch.exp(logvar)
    return 0.5 * (mu.pow(2) + var - 1.0 - logvar).sum(dim=-1)
