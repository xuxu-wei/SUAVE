"""Utilities for injecting controlled missingness patterns into arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

__all__ = [
    "MissingnessSpec",
    "inject_mcar",
    "inject_block_missing",
    "apply_missingness",
    "LITE_MISSING",
    "HEAVY_MISSING",
    "MISSING_VARIANTS",
]


@dataclass(frozen=True)
class MissingnessSpec:
    """Description of a missingness configuration.

    Parameters
    ----------
    name:
        Identifier for the configuration, e.g. ``"lite"`` or ``"heavy"``.
    raw_mcar:
        Probability of independently masking entries in the raw feature columns.
    derived_mcar:
        Probability of independently masking entries in the derived feature
        columns.
    block_frac_range:
        Range controlling what fraction of derived columns receive an additional
        heavy block-missing mask. ``None`` disables block missingness.
    block_strength:
        Baseline probability used for the heavy block-missing columns. The
        actual missing rate is sampled uniformly from ``[max(block_strength,
        0.6), 0.95]`` to avoid degenerate cases.
    seed_offset:
        Optional offset applied when sampling random seeds for the different
        masking steps.  This helps decouple deterministic callers.
    """

    name: str
    raw_mcar: float = 0.0
    derived_mcar: float = 0.0
    block_frac_range: Tuple[float, float] | None = None
    block_strength: float = 0.8
    seed_offset: int = 0


def _resolve_rng(seed: int | None = None) -> np.random.Generator:
    """Create a :class:`~numpy.random.Generator` from ``seed``."""

    if isinstance(seed, np.random.Generator):  # pragma: no cover - defensive
        return seed
    return np.random.default_rng(seed)


def _prepare_array(X: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Input array must be two-dimensional")
    return arr.copy()


def inject_mcar(
    X: np.ndarray | Sequence[Sequence[float]],
    cols: Iterable[int],
    p: float,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject MCAR missingness into the provided columns.

    The function returns a copy of ``X`` with ``np.nan`` values inserted and the
    boolean mask describing the final missingness pattern (including any
    missing values already present in ``X``).
    """

    if not 0.0 <= p <= 1.0:
        raise ValueError("Missing probability must lie in [0, 1]")
    array = _prepare_array(X)
    existing_mask = np.isnan(array)
    if p == 0.0:
        return array, existing_mask
    rng = _resolve_rng(seed)
    cols = list(cols)
    if not cols:
        return array, existing_mask
    mask = np.zeros_like(array, dtype=bool)
    mask[:, cols] = rng.random((array.shape[0], len(cols))) < p
    array[mask] = np.nan
    final_mask = existing_mask | mask
    return array, final_mask


def inject_block_missing(
    X: np.ndarray | Sequence[Sequence[float]],
    cols: Iterable[int],
    col_frac: float,
    p: float,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject block-missing patterns into ``cols``.

    ``col_frac`` denotes the fraction of columns subject to heavy missingness
    (selected without replacement).  The heavy columns are masked with
    probabilities drawn from ``[max(p, 0.6), 0.95]`` while the remaining columns
    receive milder missingness in the ``[0.3, 0.5]`` range.
    """

    if not 0.0 <= col_frac <= 1.0:
        raise ValueError("col_frac must lie in [0, 1]")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    array = _prepare_array(X)
    existing_mask = np.isnan(array)
    cols = list(cols)
    if not cols or col_frac == 0:
        return array, existing_mask
    rng = _resolve_rng(seed)
    n_cols = len(cols)
    n_heavy = max(1, int(np.ceil(col_frac * n_cols)))
    heavy_cols = set(rng.choice(cols, size=n_heavy, replace=False))
    mask = np.zeros_like(array, dtype=bool)
    for col in cols:
        if col in heavy_cols:
            rate = float(rng.uniform(max(p, 0.6), 0.95))
        else:
            rate = float(rng.uniform(0.3, 0.5))
        mask[:, col] = rng.random(array.shape[0]) < rate
    array[mask] = np.nan
    final_mask = existing_mask | mask
    return array, final_mask


def apply_missingness(
    X: np.ndarray,
    raw_cols: Sequence[int],
    derived_cols: Sequence[int],
    spec: MissingnessSpec,
    *,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a :class:`MissingnessSpec` to the dataset.

    The function sequentially applies MCAR masking to raw and derived columns
    and, if requested, augments the derived columns with block-missing patterns.
    The returned mask represents the final missingness configuration.
    """

    rng = _resolve_rng(None if seed is None else seed + spec.seed_offset)
    array = np.asarray(X, dtype=float)
    mask = np.isnan(array)
    result = array.copy()
    if spec.raw_mcar > 0 and raw_cols:
        result, mask = inject_mcar(
            result,
            raw_cols,
            spec.raw_mcar,
            seed=int(rng.integers(0, 1_000_000)),
        )
    if spec.derived_mcar > 0 and derived_cols:
        result, mask = inject_mcar(
            result,
            derived_cols,
            spec.derived_mcar,
            seed=int(rng.integers(0, 1_000_000)),
        )
    if spec.block_frac_range is not None and derived_cols:
        low, high = spec.block_frac_range
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("block_frac_range must lie within [0, 1]")
        col_frac = float(rng.uniform(low, high))
        result, mask = inject_block_missing(
            result,
            derived_cols,
            col_frac=col_frac,
            p=spec.block_strength,
            seed=int(rng.integers(0, 1_000_000)),
        )
    return result, mask


LITE_MISSING = MissingnessSpec(name="lite", raw_mcar=0.2)
HEAVY_MISSING = MissingnessSpec(
    name="heavy",
    raw_mcar=0.2,
    derived_mcar=0.5,
    block_frac_range=(0.1, 0.3),
    block_strength=0.8,
)

MISSING_VARIANTS = {
    "lite": LITE_MISSING,
    "heavy": HEAVY_MISSING,
}
