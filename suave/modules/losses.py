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


def sum_reconstruction_terms(recon_terms: Iterable[Tensor]) -> Tensor:
    """Return the aggregated reconstruction log-likelihood per sample."""

    return torch.stack(tuple(recon_terms), dim=0).sum(dim=0)


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


def kl_categorical(logits_q: Tensor, logits_p: Tensor) -> Tensor:
    """Return KL divergence between categorical distributions given logits."""

    log_q = torch.log_softmax(logits_q, dim=-1)
    log_p = torch.log_softmax(logits_p, dim=-1)
    probs_q = torch.softmax(logits_q, dim=-1)
    return (probs_q * (log_q - log_p)).sum(dim=-1)


def kl_normal_mixture(
    mu_q: Tensor,
    logvar_q: Tensor,
    mu_p: Tensor,
    logvar_p: Tensor,
    posterior_probs: Tensor,
) -> Tensor:
    """KL for diagonal Gaussians conditioned on mixture assignments."""

    if mu_q.dim() != 3 or logvar_q.dim() != 3:
        raise ValueError(
            "Expected posterior parameters with shape (batch, components, latent)"
        )
    if posterior_probs.shape != mu_q.shape[:2]:
        raise ValueError("posterior_probs must match the first two dimensions of mu_q")

    mu_p = mu_p.to(mu_q.device).unsqueeze(0)
    logvar_p = logvar_p.to(logvar_q.device).unsqueeze(0)
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p).clamp_min(1e-6)
    diff = mu_q - mu_p
    component_kl = 0.5 * (
        logvar_p - logvar_q + (var_q + diff.pow(2)) / var_p - 1.0
    ).sum(dim=-1)
    return (posterior_probs * component_kl).sum(dim=-1)
