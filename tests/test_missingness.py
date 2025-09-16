from __future__ import annotations

import numpy as np

from tests.utils.missingness import inject_block_missing, inject_mcar


def test_inject_mcar_reproducible() -> None:
    X = np.ones((1000, 4))
    result = inject_mcar(X, cols=[0, 1], p=0.2, seed=42)
    assert result.data.shape == X.shape
    assert result.mask.shape == X.shape
    assert np.all(~result.mask[:, 2:])
    rate_col0 = result.mask[:, 0].mean()
    rate_col1 = result.mask[:, 1].mean()
    assert 0.15 < rate_col0 < 0.25
    assert 0.15 < rate_col1 < 0.25
    repeat = inject_mcar(X, cols=[0, 1], p=0.2, seed=42)
    assert np.array_equal(result.mask, repeat.mask)
    assert np.array_equal(np.isnan(result.data), np.isnan(repeat.data))


def test_inject_block_missing_pattern() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 6))
    result, heavy_cols = inject_block_missing(
        X,
        cols=[2, 3, 4, 5],
        col_frac=0.25,
        p=0.5,
        seed=7,
    )
    assert result.data.shape == X.shape
    assert result.mask[:, :2].sum() == 0
    heavy_rates = result.mask[:, list(heavy_cols)].mean(axis=0) if heavy_cols else np.array([])
    if heavy_rates.size:
        assert np.all(heavy_rates >= 0.6)
    medium_cols = [idx for idx in [2, 3, 4, 5] if idx not in heavy_cols]
    if medium_cols:
        medium_rates = result.mask[:, medium_cols].mean(axis=0)
        assert np.all(medium_rates >= 0.1)
        assert np.all(medium_rates <= 0.55)
    repeat, repeat_heavy = inject_block_missing(
        X,
        cols=[2, 3, 4, 5],
        col_frac=0.25,
        p=0.5,
        seed=7,
    )
    assert np.array_equal(result.mask, repeat.mask)
    assert heavy_cols == repeat_heavy
