"""Utilities for injecting controlled missingness patterns in synthetic data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MissingnessResult:
    """Container describing the outcome of a missingness injection."""

    data: np.ndarray
    mask: np.ndarray
    columns: Tuple[int, ...]

    def combined_mask(self, other: "MissingnessResult" | None = None) -> np.ndarray:
        """Return the missingness mask, optionally merged with ``other``.

        Parameters
        ----------
        other:
            Optional missingness result to be combined via logical OR with the
            current mask.
        """

        if other is None:
            return self.mask.copy()
        return np.logical_or(self.mask, other.mask)


def _normalise_columns(cols: Sequence[int] | np.ndarray, n_features: int) -> np.ndarray:
    if len(cols) == 0:
        return np.empty(0, dtype=int)
    arr = np.asarray(cols, dtype=int)
    if (arr < 0).any() or (arr >= n_features).any():
        raise ValueError("Column indices out of bounds")
    return arr


def inject_mcar(
    X: np.ndarray,
    cols: Sequence[int] | np.ndarray,
    p: float,
    *,
    seed: int | None = None,
) -> MissingnessResult:
    """Inject Missing Completely At Random (MCAR) values into ``X``.

    Parameters
    ----------
    X:
        Input matrix that will be copied before injecting missingness.
    cols:
        Indices of columns that should receive MCAR missingness.
    p:
        Probability of masking an entry in the specified columns.
    seed:
        Optional seed to make the randomness reproducible.

    Returns
    -------
    MissingnessResult
        Object containing the mutated data array, the boolean mask and the
        affected columns.
    """

    if not 0.0 <= p <= 1.0:
        raise ValueError("Missingness probability must be in [0, 1]")
    rng = np.random.default_rng(seed)
    cols_arr = _normalise_columns(cols, X.shape[1])
    mask = np.zeros_like(X, dtype=bool)
    if cols_arr.size == 0 or p == 0.0:
        return MissingnessResult(X.copy(), mask, tuple())
    col_mask = rng.random(size=(X.shape[0], cols_arr.size)) < p
    X_new = X.copy()
    X_new[:, cols_arr] = np.where(col_mask, np.nan, X_new[:, cols_arr])
    mask[:, cols_arr] = col_mask
    return MissingnessResult(X_new, mask, tuple(cols_arr.tolist()))


def inject_block_missing(
    X: np.ndarray,
    cols: Sequence[int] | np.ndarray,
    col_frac: float,
    p: float,
    *,
    seed: int | None = None,
    medium_range: Tuple[float, float] = (0.3, 0.5),
    heavy_range: Tuple[float, float] = (0.8, 0.95),
) -> Tuple[MissingnessResult, Tuple[int, ...]]:
    """Inject block-structured missingness on top of MCAR noise.

    Columns are split into two groups. A fraction ``col_frac`` is sampled to
    receive heavy missingness with rates drawn uniformly from ``heavy_range``.
    The remaining columns receive medium missingness with rates sampled from
    ``medium_range`` (capped by ``p``).

    Parameters
    ----------
    X:
        Input matrix that will be copied before injection.
    cols:
        Candidate columns among which heavy columns are sampled.
    col_frac:
        Fraction of columns that should become heavy-missing columns.
    p:
        Upper bound for the medium missingness rate.
    seed:
        Optional random seed for reproducibility.
    medium_range:
        Inclusive range for the medium missingness rate. The upper bound will
        be clipped to ``p``.
    heavy_range:
        Inclusive range for the heavy missingness rate applied to the sampled
        heavy columns.

    Returns
    -------
    Tuple[MissingnessResult, Tuple[int, ...]]
        Missingness result and the indices of the heavy-missing columns.
    """

    if not 0.0 <= col_frac <= 1.0:
        raise ValueError("Column fraction must be between 0 and 1")
    if not 0.0 <= p <= 1.0:
        raise ValueError("Missingness probability must be in [0, 1]")
    rng = np.random.default_rng(seed)
    cols_arr = _normalise_columns(cols, X.shape[1])
    X_new = X.copy()
    mask = np.zeros_like(X_new, dtype=bool)
    if cols_arr.size == 0:
        return MissingnessResult(X_new, mask, tuple()), tuple()

    n_heavy = int(np.round(cols_arr.size * col_frac))
    if n_heavy == 0 and cols_arr.size > 0 and col_frac > 0.0:
        n_heavy = 1
    heavy_cols = (
        tuple(sorted(rng.choice(cols_arr, size=n_heavy, replace=False).tolist()))
        if n_heavy > 0
        else tuple()
    )
    heavy_set = set(heavy_cols)
    medium_low, medium_high = medium_range
    medium_high = min(medium_high, p)
    heavy_low, heavy_high = heavy_range
    for col in cols_arr:
        if col in heavy_set:
            rate = float(rng.uniform(heavy_low, heavy_high))
        else:
            rate = float(rng.uniform(medium_low, medium_high))
        if rate <= 0.0:
            continue
        col_mask = rng.random(X_new.shape[0]) < rate
        mask[col_mask, col] = True
        X_new[col_mask, col] = np.nan
    return MissingnessResult(X_new, mask, tuple(cols_arr.tolist())), heavy_cols
