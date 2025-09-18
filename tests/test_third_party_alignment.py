"""Tests comparing SUAVE's HI-VAE heads against the published TensorFlow formulas."""

from __future__ import annotations

import math

import numpy as np
import torch

from suave.modules.decoder import CatHead, RealHead


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus matching TensorFlow's implementation."""

    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _gaussian_log_likelihood(
    x_raw: np.ndarray,
    mask: np.ndarray,
    mean_raw: np.ndarray,
    var_raw: np.ndarray,
    mean_scale: float,
    std_scale: float,
) -> np.ndarray:
    """Reproduce the TensorFlow HI-VAE Gaussian reconstruction formula."""

    eps = 1e-6
    var = _softplus(var_raw) + eps
    mean = mean_raw * std_scale + mean_scale
    var_scaled = var * (std_scale**2)
    log_det = np.log(2.0 * math.pi * var_scaled)
    squared = (x_raw - mean) ** 2 / np.clip(var_scaled, a_min=eps, a_max=None)
    log_px = -0.5 * (log_det + squared)
    log_px = log_px * mask
    return log_px.sum(axis=-1)


def _categorical_log_likelihood(
    one_hot: np.ndarray,
    logits: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Reproduce the TensorFlow HI-VAE categorical reconstruction formula."""

    max_logits = np.max(logits, axis=-1, keepdims=True)
    log_probs = (
        logits
        - max_logits
        - np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True))
    )
    log_px = (one_hot * log_probs).sum(axis=-1)
    if mask is not None:
        log_px = log_px * mask.squeeze(-1)
    return log_px


def test_real_head_matches_third_party_formulation() -> None:
    rng = np.random.default_rng(42)
    batch = 6

    mean_scale = float(rng.normal(loc=1.0, scale=0.2))
    std_scale = float(rng.uniform(0.5, 1.5))

    x_raw = rng.normal(loc=mean_scale, scale=std_scale, size=(batch, 1)).astype(
        np.float32
    )
    mask = (rng.random(size=(batch, 1)) > 0.2).astype(np.float32)
    mean_raw = rng.normal(size=(batch, 1)).astype(np.float32)
    var_raw = rng.normal(size=(batch, 1)).astype(np.float32)

    x_norm = (x_raw - mean_scale) / std_scale
    params = np.concatenate([mean_raw, var_raw], axis=-1)

    head = RealHead()
    output = head(
        torch.from_numpy(x_norm),
        torch.from_numpy(params),
        {"mean": mean_scale, "std": std_scale},
        torch.from_numpy(mask),
    )

    expected = _gaussian_log_likelihood(
        x_raw, mask, mean_raw, var_raw, mean_scale, std_scale
    )
    torch.testing.assert_close(
        output["log_px"], torch.from_numpy(expected), atol=1e-5, rtol=1e-4
    )

    missing_mask = 1.0 - mask
    expected_missing = _gaussian_log_likelihood(
        x_raw, missing_mask, mean_raw, var_raw, mean_scale, std_scale
    )
    torch.testing.assert_close(
        output["log_px_missing"],
        torch.from_numpy(expected_missing),
        atol=1e-5,
        rtol=1e-4,
    )


def test_categorical_head_matches_third_party_formulation() -> None:
    rng = np.random.default_rng(7)
    batch = 5
    n_classes = 4

    logits = rng.normal(size=(batch, n_classes)).astype(np.float32)
    indices = rng.integers(low=0, high=n_classes, size=batch)
    one_hot = np.eye(n_classes, dtype=np.float32)[indices]
    mask = (rng.random(size=(batch, 1)) > 0.3).astype(np.float32)

    head = CatHead(n_classes=n_classes)
    output = head(
        torch.from_numpy(one_hot),
        torch.from_numpy(logits),
        None,
        torch.from_numpy(mask),
    )

    expected = _categorical_log_likelihood(one_hot, logits, mask)
    torch.testing.assert_close(
        output["log_px"], torch.from_numpy(expected), atol=1e-5, rtol=1e-4
    )

    expected_missing = _categorical_log_likelihood(one_hot, logits, 1.0 - mask)
    torch.testing.assert_close(
        output["log_px_missing"],
        torch.from_numpy(expected_missing),
        atol=1e-5,
        rtol=1e-4,
    )
