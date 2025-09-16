"""Tests for the missingness injection utilities."""

from __future__ import annotations

import numpy as np

from tests.utils.missingness import MissingnessSpec, apply_missingness, inject_block_missing, inject_mcar


def test_inject_mcar_respects_probability() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    X[:, 2] = np.nan  # existing missing values should be preserved
    X_missing, mask = inject_mcar(X, cols=[0, 1], p=0.25, seed=2024)
    assert X_missing.shape == X.shape
    assert mask.shape == X.shape
    # Existing NaNs remain untouched
    assert np.all(np.isnan(X_missing[:, 2]))
    col_means = mask.mean(axis=0)
    assert abs(col_means[0] - 0.25) < 0.03
    assert abs(col_means[1] - 0.25) < 0.03
    # Unspecified columns are unaffected beyond existing NaNs
    assert col_means[3] == 0.0


def test_inject_block_missing_creates_heavy_columns() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1500, 6))
    derived_cols = [2, 3, 4, 5]
    X_missing, mask = inject_block_missing(X, derived_cols, col_frac=0.25, p=0.8, seed=99)
    assert X_missing.shape == X.shape
    rates = mask.mean(axis=0)
    heavy = [c for c in derived_cols if rates[c] > 0.6]
    assert heavy, "At least one column should exhibit heavy missingness"
    medium = [c for c in derived_cols if 0.25 < rates[c] < 0.6]
    assert medium, "Some columns should remain in the medium missingness range"
    assert np.all(rates[:2] == 0), "Raw columns should be untouched"


def test_apply_missingness_composes_strategies() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 10))
    raw_cols = list(range(6))
    derived_cols = list(range(6, 10))
    spec = MissingnessSpec(
        name="heavy",
        raw_mcar=0.2,
        derived_mcar=0.5,
        block_frac_range=(0.1, 0.3),
        block_strength=0.85,
    )
    X_missing, mask = apply_missingness(X, raw_cols, derived_cols, spec, seed=123)
    assert X_missing.shape == X.shape
    assert mask.shape == X.shape
    raw_rate = mask[:, raw_cols].mean()
    derived_rate = mask[:, derived_cols].mean()
    assert 0.15 < raw_rate < 0.25
    assert derived_rate > 0.45
    # Check that some derived columns approach the block strength
    per_col = mask[:, derived_cols].mean(axis=0)
    assert np.any(per_col > 0.75)
