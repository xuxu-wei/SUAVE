"""Distribution helpers shared by the SUAVE decoder components.

The generative module expresses reconstruction terms for every column type in
terms of log-likelihoods under the corresponding distribution.  This module
contains the PyTorch utilities used across the decoder heads.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Categorical, Normal, Poisson
from torch.nn import functional as F

EPS = 1e-6
_LOG_2PI = math.log(2 * math.pi)


def _apply_mask(value: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Apply ``mask`` to ``value`` if provided.

    Parameters
    ----------
    value:
        Tensor containing per-sample values such as negative log-likelihoods.
        The masking operation is applied to the last dimension.
    mask:
        Optional tensor of the same shape (or broadcastable to) ``value`` that
        marks valid observations with ``1`` and missing entries with ``0``.
    """

    if mask is None:
        return value
    if mask.shape != value.shape:
        mask = mask.expand_as(value)
    return value * mask


def nll_gaussian(x: Tensor, mu: Tensor, var: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Return the negative log-likelihood of ``x`` under a diagonal Gaussian.

    All tensors are expected to share the same trailing dimension.  The result
    is a one-dimensional tensor with the batch-wise NLL values after masking
    missing observations.
    """

    var = torch.clamp(var, min=EPS)
    log_var = torch.log(var)
    nll = 0.5 * (_LOG_2PI + log_var + (x - mu) ** 2 / var)
    nll = _apply_mask(nll, mask)
    return nll.sum(dim=-1)


def nll_categorical(x_onehot: Tensor, logits: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Return the categorical negative log-likelihood for one-hot targets."""

    log_probs = F.log_softmax(logits, dim=-1)
    nll = -(x_onehot * log_probs).sum(dim=-1)
    if mask is not None:
        nll = nll * mask.squeeze(-1)
    return nll


def sample_gaussian(mu: Tensor, var: Tensor) -> Tensor:
    """Sample from a diagonal Gaussian distribution with parameters ``mu``/``var``."""

    std = torch.sqrt(torch.clamp(var, min=EPS))
    eps = torch.randn_like(std)
    return mu + eps * std


def sample_categorical(logits: Tensor) -> Tensor:
    """Sample one-hot vectors from a categorical distribution defined by ``logits``."""

    distribution = Categorical(logits=logits)
    indices = distribution.sample()
    return F.one_hot(indices, num_classes=logits.size(-1)).float()


def make_normal(mu: Tensor, var: Tensor) -> Normal:
    """Helper constructing :class:`Normal` objects from mean and variance."""

    std = torch.sqrt(torch.clamp(var, min=EPS))
    return Normal(mu, std)


def nll_lognormal(
    log_x: Tensor, mean_log: Tensor, var_log: Tensor, mask: Optional[Tensor]
) -> Tensor:
    """Return the negative log-likelihood of log-normal variables.

    Parameters
    ----------
    log_x:
        ``log(1 + x)`` representation of the observed data.
    mean_log, var_log:
        Parameters of the Gaussian distribution in log-space.
    mask:
        Observation mask with ones for observed entries and zeros for missing.
    """

    var_log = torch.clamp(var_log, min=EPS)
    log_var = torch.log(var_log)
    nll = 0.5 * (_LOG_2PI + log_var + (log_x - mean_log) ** 2 / var_log) + log_x
    nll = _apply_mask(nll, mask)
    return nll.sum(dim=-1)


def sample_lognormal(mean_log: Tensor, var_log: Tensor) -> Tensor:
    """Sample positive values from a log-normal distribution."""

    std = torch.sqrt(torch.clamp(var_log, min=EPS))
    eps = torch.randn_like(std)
    return torch.exp(mean_log + eps * std) - 1.0


def nll_poisson(x: Tensor, rate: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Return the negative log-likelihood of count data under a Poisson model."""

    rate = torch.clamp(rate, min=EPS)
    distribution = Poisson(rate)
    nll = -distribution.log_prob(x)
    nll = _apply_mask(nll, mask)
    return nll.sum(dim=-1)


def sample_poisson(rate: Tensor) -> Tensor:
    """Sample from a Poisson distribution with intensity ``rate``."""

    distribution = Poisson(rate)
    return distribution.sample()


def ordinal_probabilities(partition: Tensor, mean: Tensor) -> tuple[Tensor, Tensor]:
    """Return class probabilities and ordered thresholds for ordinal variables."""

    epsilon = EPS
    spacings = F.softplus(partition) + epsilon
    thresholds = torch.cumsum(spacings, dim=-1)
    mean_expanded = mean
    if mean_expanded.dim() < thresholds.dim():
        mean_expanded = mean_expanded.unsqueeze(-1)
    logits = thresholds - mean_expanded
    sigmoid = torch.sigmoid(logits)
    probs = torch.cat([sigmoid, torch.ones_like(sigmoid[..., :1])], dim=-1)
    probs = probs - torch.cat([torch.zeros_like(sigmoid[..., :1]), sigmoid], dim=-1)
    probs = torch.clamp(probs, min=EPS, max=1.0)
    return probs, thresholds


def nll_ordinal(probs: Tensor, targets: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Negative log-likelihood for ordinal observations."""

    log_probs = torch.log(torch.clamp(probs, min=EPS))
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll = -gathered
    if mask is not None:
        if mask.shape != nll.shape:
            mask = mask.expand_as(nll)
        nll = nll * mask
    return nll


def sample_ordinal(probs: Tensor) -> Tensor:
    """Sample thermometer-encoded ordinal observations from ``probs``."""

    distribution = Categorical(probs=probs)
    indices = distribution.sample()
    n_classes = probs.size(-1)
    arange = torch.arange(n_classes, device=probs.device)
    return (arange.unsqueeze(0) <= indices.unsqueeze(-1)).float()
