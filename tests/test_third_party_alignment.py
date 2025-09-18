"""Tests comparing SUAVE's HI-VAE heads against the published TensorFlow formulas."""

from __future__ import annotations

import math

import numpy as np
import torch

from suave.modules.decoder import CatHead, CountHead, OrdinalHead, PosHead, RealHead


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
    return log_px.sum(axis=-1).astype(np.float32)


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


def _log_normal_log_likelihood(
    x_raw: np.ndarray,
    mask: np.ndarray,
    mean_raw: np.ndarray,
    var_raw: np.ndarray,
    mean_log: float,
    std_log: float,
) -> np.ndarray:
    """Reproduce the TensorFlow HI-VAE log-normal reconstruction formula."""

    eps = 1e-6
    var = _softplus(var_raw) + eps
    mean = mean_raw * std_log + mean_log
    var_scaled = var * (std_log**2)
    log_x = np.log1p(x_raw)
    log_det = np.log(2.0 * math.pi * var_scaled)
    squared = (log_x - mean) ** 2 / np.clip(var_scaled, a_min=eps, a_max=None)
    log_px = -0.5 * (log_det + squared) - log_x
    log_px = log_px * mask
    return log_px.sum(axis=-1).astype(np.float32)


def _poisson_log_likelihood(
    x_log: np.ndarray,
    mask: np.ndarray,
    rate_raw: np.ndarray,
    offset: float,
) -> np.ndarray:
    """Reproduce the TensorFlow HI-VAE Poisson reconstruction formula."""

    eps = 1e-6
    rate = _softplus(rate_raw) + eps
    counts = np.exp(x_log) - offset
    counts = np.clip(counts, a_min=0.0, a_max=None)
    log_rate = np.log(np.clip(rate, a_min=eps, a_max=None))
    log_factorial = np.vectorize(math.lgamma)(counts + 1.0)
    log_px = counts * log_rate - rate - log_factorial
    log_px = log_px * mask
    return log_px.sum(axis=-1).astype(np.float32)


def _ordinal_probabilities(
    partition: np.ndarray, mean_param: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recreate the HI-VAE ordinal probability construction."""

    eps = 1e-6
    spacings = _softplus(partition) + eps
    thresholds = np.cumsum(spacings, axis=-1)
    mean_expanded = mean_param
    if mean_expanded.ndim < thresholds.ndim:
        mean_expanded = np.expand_dims(mean_expanded, axis=-1)
    logits = thresholds - mean_expanded
    sigmoid = 1.0 / (1.0 + np.exp(-logits))
    probs = np.concatenate(
        [sigmoid, np.ones_like(sigmoid[..., :1])], axis=-1
    ) - np.concatenate([
        np.zeros_like(sigmoid[..., :1]),
        sigmoid,
    ], axis=-1)
    probs = np.clip(probs, a_min=eps, a_max=1.0)
    return probs, thresholds


def _ordinal_log_likelihood(
    thermometer: np.ndarray,
    mask: np.ndarray,
    partition: np.ndarray,
    mean_param: np.ndarray,
) -> np.ndarray:
    """Reproduce the TensorFlow HI-VAE ordinal reconstruction formula."""

    probs, _ = _ordinal_probabilities(partition, mean_param)
    levels = np.round(thermometer.sum(axis=-1)).astype(int)
    targets = np.clip(levels - 1, a_min=0, a_max=probs.shape[-1] - 1)
    log_probs = np.log(probs)
    gathered = np.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    mask_vector = mask.squeeze(-1)
    gathered = gathered * mask_vector
    return gathered.astype(np.float32)


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


def test_positive_head_matches_third_party_formulation() -> None:
    rng = np.random.default_rng(21)
    batch = 5

    mean_log = float(rng.normal(loc=0.2, scale=0.5))
    std_log = float(rng.uniform(0.5, 1.5))

    x_raw = rng.uniform(low=0.0, high=5.0, size=(batch, 1)).astype(np.float32)
    mask = (rng.random(size=(batch, 1)) > 0.25).astype(np.float32)
    mean_raw = rng.normal(size=(batch, 1)).astype(np.float32)
    var_raw = rng.normal(size=(batch, 1)).astype(np.float32)

    x_norm = ((np.log1p(x_raw) - mean_log) / std_log).astype(np.float32)
    params = np.concatenate([mean_raw, var_raw], axis=-1)

    head = PosHead()
    output = head(
        torch.from_numpy(x_norm),
        torch.from_numpy(params),
        {"mean_log": mean_log, "std_log": std_log},
        torch.from_numpy(mask),
    )

    expected = _log_normal_log_likelihood(
        x_raw, mask, mean_raw, var_raw, mean_log, std_log
    )
    torch.testing.assert_close(
        output["log_px"], torch.from_numpy(expected), atol=1e-5, rtol=1e-4
    )

    missing_mask = 1.0 - mask
    expected_missing = _log_normal_log_likelihood(
        x_raw, missing_mask, mean_raw, var_raw, mean_log, std_log
    )
    torch.testing.assert_close(
        output["log_px_missing"],
        torch.from_numpy(expected_missing),
        atol=1e-5,
        rtol=1e-4,
    )


def test_count_head_matches_third_party_formulation() -> None:
    rng = np.random.default_rng(314)
    batch = 6

    counts = rng.integers(low=0, high=5, size=(batch, 1)).astype(np.float32)
    offset = 1.0 if (counts <= 0).any() else 0.0
    x_log = np.log(counts + offset).astype(np.float32)
    mask = (rng.random(size=(batch, 1)) > 0.35).astype(np.float32)
    rate_raw = rng.normal(size=(batch, 1)).astype(np.float32)

    head = CountHead()
    output = head(
        torch.from_numpy(x_log),
        torch.from_numpy(rate_raw),
        {"offset": offset},
        torch.from_numpy(mask),
    )

    expected = _poisson_log_likelihood(x_log, mask, rate_raw, offset)
    torch.testing.assert_close(
        output["log_px"], torch.from_numpy(expected), atol=1e-5, rtol=1e-4
    )

    missing_mask = 1.0 - mask
    expected_missing = _poisson_log_likelihood(
        x_log, missing_mask, rate_raw, offset
    )
    torch.testing.assert_close(
        output["log_px_missing"],
        torch.from_numpy(expected_missing),
        atol=1e-5,
        rtol=1e-4,
    )


def _make_thermometer(indices: np.ndarray, n_classes: int) -> np.ndarray:
    thermometer = np.zeros((indices.size, n_classes), dtype=np.float32)
    for row, idx in enumerate(indices):
        thermometer[row, : idx + 1] = 1.0
    return thermometer


def test_ordinal_head_matches_third_party_formulation() -> None:
    rng = np.random.default_rng(11)
    batch = 4
    n_classes = 4

    indices = rng.integers(low=0, high=n_classes, size=batch)
    thermometer = _make_thermometer(indices, n_classes)
    mask = (rng.random(size=(batch, 1)) > 0.4).astype(np.float32)

    partition = rng.normal(size=(batch, n_classes - 1)).astype(np.float32)
    mean_param = rng.normal(size=(batch, 1)).astype(np.float32)
    params = np.concatenate([partition, mean_param], axis=-1)

    head = OrdinalHead(n_classes=n_classes)
    output = head(
        torch.from_numpy(thermometer),
        torch.from_numpy(params),
        {},
        torch.from_numpy(mask),
    )

    expected = _ordinal_log_likelihood(
        thermometer,
        mask,
        partition,
        mean_param,
    )
    torch.testing.assert_close(
        output["log_px"], torch.from_numpy(expected), atol=1e-5, rtol=1e-4
    )

    missing_mask = 1.0 - mask
    expected_missing = _ordinal_log_likelihood(
        thermometer,
        missing_mask,
        partition,
        mean_param,
    )
    torch.testing.assert_close(
        output["log_px_missing"],
        torch.from_numpy(expected_missing),
        atol=1e-5,
        rtol=1e-4,
    )
