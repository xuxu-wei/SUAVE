"""Distribution helpers mirroring the TensorFlow HI-VAE implementation.

The original HI-VAE code expresses reconstruction terms for every column type
in terms of log-likelihoods under the corresponding distribution.  This module
contains the PyTorch equivalents that are shared by the decoder heads.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Categorical, Normal
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
    """Helper mirroring the TF utility for constructing :class:`Normal` objects."""

    std = torch.sqrt(torch.clamp(var, min=EPS))
    return Normal(mu, std)
