"""Synthetic benchmark task definitions and missingness helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .missingness import inject_block_missing, inject_mcar

DEFAULT_SEED = 20201021
_GLOBAL_RNG = np.random.default_rng(DEFAULT_SEED)


def _get_rng(seed: int | None) -> np.random.Generator:
    return _GLOBAL_RNG if seed is None else np.random.default_rng(seed)


@dataclass
class TaskData:
    """Container holding synthetic benchmark data."""

    features: np.ndarray
    targets: np.ndarray
    task_classes: List[int]
    masks: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class HardTaskSpec:
    """Specification of a hard benchmark task."""

    name: str
    generator: Callable[[Optional[int]], TaskData]
    raw_cols: Tuple[int, ...]
    derived_cols: Tuple[int, ...]


def _insert_missing(
    X: np.ndarray,
    rng: np.random.Generator,
    *,
    mcar: float = 0.0,
    mnar_indices: Sequence[int] | None = None,
    y: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject MCAR and MNAR missingness and return the mask."""

    X_new = X.copy()
    mask = np.zeros_like(X_new, dtype=bool)
    if mcar > 0:
        mcar_mask = rng.random(X_new.shape) < mcar
        X_new[mcar_mask] = np.nan
        mask |= mcar_mask
    if mnar_indices is not None and y is not None:
        for idx in mnar_indices:
            indicator = (y == 1)
            col_mask = indicator & (rng.random(indicator.shape[0]) < 0.5)
            X_new[col_mask, idx] = np.nan
            mask[col_mask, idx] = True
    return X_new, mask


def _combine_masks(masks: Iterable[np.ndarray]) -> np.ndarray:
    masks = list(masks)
    if not masks:
        return np.zeros((0,), dtype=bool)
    combined = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        combined |= m
    return combined


def generate_simple(seed: int | None = None) -> TaskData:
    rng = _get_rng(seed)
    n = 200
    X = rng.normal(size=(n, 10))
    y1 = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + rng.normal(scale=0.1, size=n) > 0).astype(int)
    w = X[:, 3] + 0.5 * X[:, 4] - X[:, 5]
    bins = np.quantile(w, [0, 1 / 3, 2 / 3, 1])
    y2 = np.digitize(w, bins[1:-1])
    Y = np.column_stack([y1, y2])
    X_missing, mask = _insert_missing(X, rng, mcar=0.05)
    return TaskData(
        features=X_missing,
        targets=Y,
        task_classes=[2, 3],
        masks={"base_missing": mask},
        metadata={"name": "simple", "seed": seed},
    )


def generate_medium(seed: int | None = None) -> TaskData:
    rng = _get_rng(seed)
    n = 1000
    part1 = rng.normal(size=(n, 10))
    part2 = rng.uniform(-3, 3, size=(n, 5))
    part3 = rng.exponential(scale=1.0, size=(n, 5))
    X = np.hstack([part1, part2, part3])
    x1, x2, x3, x4, x5, x6, x7, x8 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7]
    derived = np.column_stack(
        [
            x1 * x2,
            np.sin(x3),
            x4 ** 2,
            (x5 > 0).astype(float),
            x6 + rng.normal(scale=0.1, size=n),
        ]
    )
    noise = rng.normal(size=(n, 5))
    X = np.hstack([X, derived, noise])
    y1_score = x1 * x2 + np.sin(x3) + np.log(np.abs(x4) + 1) + rng.normal(scale=0.1, size=n)
    thresh = np.quantile(y1_score, 0.7)
    y1 = (y1_score > thresh).astype(int)
    y2_score = x5 ** 2 - x6 * x7 + np.sin(x8)
    bins = np.quantile(y2_score, [0, 1 / 3, 2 / 3, 1])
    y2 = np.digitize(y2_score, bins[1:-1])
    Y = np.column_stack([y1, y2])
    X_missing, mask = _insert_missing(X, rng, mcar=0.1, mnar_indices=[1], y=y1)
    return TaskData(
        features=X_missing,
        targets=Y,
        task_classes=[2, 3],
        masks={"base_missing": mask},
        metadata={"name": "medium", "seed": seed},
    )


def generate_hard(seed: int | None = None) -> TaskData:
    rng = _get_rng(seed)
    n = 3000
    cont_norm = rng.normal(size=(n, 10))
    cont_log = rng.lognormal(mean=0.0, sigma=1.0, size=(n, 5)) * 10
    cont_uni = rng.uniform(-5, 5, size=(n, 5))
    cont_exp = rng.exponential(scale=1.0, size=(n, 5))
    cont = np.hstack([cont_norm, cont_log, cont_uni, cont_exp])
    pois = rng.poisson(lam=3, size=(n, 5))
    bino = rng.binomial(n=10, p=0.3, size=(n, 5))
    cat = rng.integers(0, 4, size=(n, 5))
    base = np.hstack([cont, pois, bino, cat])
    d1 = base[:, 0] * base[:, 1]
    d2 = np.sin(base[:, 2])
    d3 = np.log(np.abs(base[:, 3]) + 1)
    d4 = (base[:, 4] > base[:, 5]).astype(float)
    d5 = base[:, 6] ** 2
    derived = np.column_stack([d1, d2, d3, d4, d5])
    noise = rng.normal(size=(n, 5))
    X = np.hstack([base, derived, noise])
    s1 = base[:, 0] * base[:, 1] + np.sin(base[:, 2]) - np.log(np.abs(base[:, 3]) + 1) + base[:, 4] ** 2
    t1 = np.quantile(s1, 0.9)
    y1 = (s1 > t1).astype(int)
    s2 = base[:, 5] * base[:, 6] - np.sin(base[:, 7]) + np.log1p(base[:, 8] ** 2) - base[:, 9]
    bins = np.quantile(s2, [0, 0.25, 0.5, 0.75, 1])
    y2 = np.digitize(s2, bins[1:-1])
    y3 = ((cont_norm[:, 0] > 0) ^ (cont_uni[:, 0] > 0)).astype(int)
    Y = np.column_stack([y1, y2, y3])
    X_missing, mask = _insert_missing(X, rng, mcar=0.1, mnar_indices=[0, 1], y=y1)
    return TaskData(
        features=X_missing,
        targets=Y,
        task_classes=[2, 4, 2],
        masks={"base_missing": mask},
        metadata={"name": "hard", "seed": seed},
    )


HARD_TASK_CONFIGS: Dict[str, HardTaskSpec] = {
    "hard": HardTaskSpec(
        name="hard",
        generator=generate_hard,
        raw_cols=tuple(range(40)),
        derived_cols=tuple(range(40, 45)),
    )
}


def get_hard_task_configs() -> Mapping[str, HardTaskSpec]:
    """Return available hard task configurations."""

    return HARD_TASK_CONFIGS


def create_missing_variant(
    spec: HardTaskSpec,
    variant: str,
    *,
    seed: int | None = None,
    raw_missing_rate: float = 0.2,
    derived_missing_rate: float = 0.5,
    block_frac_range: Tuple[float, float] = (0.1, 0.3),
) -> TaskData:
    """Create a missingness-enhanced variant of a hard task."""

    if variant not in {"lite", "heavy"}:
        raise ValueError(f"Unsupported variant: {variant}")
    base = spec.generator(seed)
    rng = np.random.default_rng(DEFAULT_SEED if seed is None else seed + 1337)
    X_missing = base.features.copy()
    masks: Dict[str, np.ndarray] = dict(base.masks)
    metadata = dict(base.metadata)
    metadata.update(
        {
            "base_task": spec.name,
            "variant": variant,
            "raw_missing_rate": raw_missing_rate,
            "derived_missing_rate": derived_missing_rate,
        }
    )
    raw_result = inject_mcar(X_missing, spec.raw_cols, raw_missing_rate, seed=int(rng.integers(0, 1_000_000)))
    X_missing = raw_result.data
    masks["raw_missing"] = raw_result.mask

    if variant == "heavy" and spec.derived_cols:
        derived_seed = int(rng.integers(0, 1_000_000))
        derived_result = inject_mcar(
            X_missing,
            spec.derived_cols,
            derived_missing_rate,
            seed=derived_seed,
        )
        X_missing = derived_result.data
        masks["derived_mcar"] = derived_result.mask
        low, high = block_frac_range
        if low < 0 or high > 1 or low > high:
            raise ValueError("Invalid block fraction range")
        col_frac = float(rng.uniform(low, high))
        medium_low = max(0.1, derived_missing_rate - 0.2)
        block_result, heavy_cols = inject_block_missing(
            X_missing,
            spec.derived_cols,
            col_frac=col_frac,
            p=derived_missing_rate,
            seed=int(rng.integers(0, 1_000_000)),
            medium_range=(medium_low, derived_missing_rate),
        )
        X_missing = block_result.data
        masks["derived_block"] = block_result.mask
        metadata["block_fraction"] = col_frac
        metadata["heavy_columns"] = list(heavy_cols)
    combined_mask = _combine_masks(masks.values())
    if combined_mask.size:
        masks["combined"] = combined_mask
    metadata["name"] = f"{spec.name}-{variant}"
    return TaskData(
        features=X_missing,
        targets=base.targets,
        task_classes=base.task_classes,
        masks=masks,
        metadata=metadata,
    )


def get_missing_task_generators(
    variants: Sequence[str] = ("lite", "heavy"),
) -> Dict[str, Callable[[Optional[int]], TaskData]]:
    """Return generators for all requested missingness variants."""

    generators: Dict[str, Callable[[Optional[int]], TaskData]] = {}
    for spec in HARD_TASK_CONFIGS.values():
        for variant in variants:
            name = f"{spec.name}-{variant}"
            generators[name] = lambda seed, spec=spec, variant=variant: create_missing_variant(
                spec,
                variant,
                seed=seed,
            )
    return generators
