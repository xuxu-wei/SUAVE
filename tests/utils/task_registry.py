"""Registry describing the synthetic hard benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

from tests.utils.missingness import MissingnessSpec, apply_missingness, inject_mcar

__all__ = [
    "TaskData",
    "TaskSpec",
    "HARD_TASK_SPECS",
    "TASK_REGISTRY",
    "get_hard_task_specs",
    "load_task",
    "make_missing_variant",
]

DEFAULT_SEED = 20201021


@dataclass
class TaskData:
    """Container holding the generated dataset and metadata."""

    features: np.ndarray
    targets: np.ndarray
    task_classes: List[int]
    raw_cols: List[int]
    derived_cols: List[int]
    base_missing_mask: np.ndarray
    seed: int


@dataclass(frozen=True)
class TaskSpec:
    """Description of a benchmark task."""

    name: str
    loader: Callable[[np.random.Generator, int], TaskData]
    tags: Sequence[str] = ()
    description: str = ""

    def load(self, seed: int | None = None) -> TaskData:
        actual_seed = DEFAULT_SEED if seed is None else seed
        rng = np.random.default_rng(actual_seed)
        return self.loader(rng, actual_seed)


def _hard_common_features(
    rng: np.random.Generator, n: int = 3000
) -> tuple[np.ndarray, Dict[str, np.ndarray], List[int], List[int]]:
    cont_norm = rng.normal(size=(n, 10))
    cont_log = rng.lognormal(mean=0.0, sigma=1.0, size=(n, 5)) * 10
    cont_uni = rng.uniform(-5, 5, size=(n, 5))
    cont_exp = rng.exponential(scale=1.0, size=(n, 5))
    cont = np.hstack([cont_norm, cont_log, cont_uni, cont_exp])
    pois = rng.poisson(lam=3, size=(n, 5)).astype(float)
    bino = rng.binomial(n=10, p=0.3, size=(n, 5)).astype(float)
    cat = rng.integers(0, 4, size=(n, 5)).astype(float)
    base = np.hstack([cont, pois, bino, cat])
    raw_cols = list(range(base.shape[1]))

    derived_main = np.column_stack(
        [
            base[:, 0] * base[:, 1],
            np.sin(base[:, 10]),
            np.log1p(np.abs(base[:, 15])),
            (base[:, 30] > 3).astype(float),
            (base[:, 35] == 0).astype(float),
        ]
    )
    derived_aux = np.column_stack(
        [
            base[:, 2] * base[:, 3] - base[:, 4],
            np.tanh(base[:, 11]),
            base[:, 18] * (base[:, 19] > 0).astype(float),
            (base[:, 26] > 2).astype(float) + (base[:, 27] > 2).astype(float),
            np.cos(base[:, 38]) + np.sin(base[:, 39]),
        ]
    )
    noise = rng.normal(scale=1.0, size=(n, 5))
    features = np.hstack([base, derived_main, derived_aux, noise])
    derived_cols = list(range(base.shape[1], features.shape[1]))

    components: Dict[str, np.ndarray] = {
        "base": base,
        "cont_norm": cont_norm,
        "cont_log": cont_log,
        "cont_uni": cont_uni,
        "cont_exp": cont_exp,
        "pois": pois,
        "bino": bino,
        "cat": cat,
        "derived_main": derived_main,
        "derived_aux": derived_aux,
        "noise": noise,
    }
    return features, components, raw_cols, derived_cols


def _apply_initial_missing(
    X: np.ndarray,
    rng: np.random.Generator,
    *,
    mcar: float = 0.0,
    mnar_indices: Iterable[int] | None = None,
    y_reference: np.ndarray | None = None,
    mnar_prob: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.asarray(X, dtype=float).copy()
    mask = np.isnan(result)
    if mcar > 0:
        result, mask = inject_mcar(
            result,
            cols=range(result.shape[1]),
            p=mcar,
            seed=int(rng.integers(0, 1_000_000)),
        )
    if mnar_indices and y_reference is not None:
        ref = np.asarray(y_reference)
        if ref.ndim > 1:
            ref = ref.ravel()
        for idx in mnar_indices:
            indicator = (ref > 0) & (rng.random(ref.shape[0]) < mnar_prob)
            result[indicator, idx] = np.nan
        mask = np.isnan(result)
    return result, mask


def _load_hard_core(rng: np.random.Generator, seed: int) -> TaskData:
    X, comps, raw_cols, derived_cols = _hard_common_features(rng)
    base = comps["base"]
    cont_norm = comps["cont_norm"]
    cont_uni = comps["cont_uni"]

    score1 = base[:, 0] * base[:, 1] + np.sin(base[:, 10]) - np.log(np.abs(base[:, 15]) + 1)
    score1 += base[:, 30]
    y1 = (score1 > np.quantile(score1, 0.9)).astype(int)

    score2 = base[:, 5] * base[:, 6] - np.sin(base[:, 7]) + np.log1p(base[:, 8] ** 2) - base[:, 25]
    bins2 = np.quantile(score2, [0, 0.25, 0.5, 0.75, 1])
    y2 = np.digitize(score2, bins2[1:-1])

    y3 = ((cont_norm[:, 0] > 0) ^ (cont_uni[:, 0] > 0)).astype(int)

    targets = np.column_stack([y1, y2, y3])
    task_classes = [2, 4, 2]
    features, mask = _apply_initial_missing(
        X,
        rng,
        mcar=0.1,
        mnar_indices=[0, 1],
        y_reference=y1,
        mnar_prob=0.5,
    )
    return TaskData(features, targets, task_classes, raw_cols, derived_cols, mask, seed)


def _load_hard_heterogeneous(rng: np.random.Generator, seed: int) -> TaskData:
    X, comps, raw_cols, derived_cols = _hard_common_features(rng)
    derived_main = comps["derived_main"]
    derived_aux = comps["derived_aux"]
    noise = comps["noise"]
    pois = comps["pois"]
    bino = comps["bino"]
    cat = comps["cat"]
    cont_log = comps["cont_log"]
    cont_exp = comps["cont_exp"]
    n = X.shape[0]

    score1 = (
        0.7 * derived_main[:, 0]
        - 0.4 * derived_main[:, 1]
        + 0.3 * derived_aux[:, 0]
        + 0.2 * noise[:, 0]
        + rng.normal(scale=0.2, size=n)
    )
    y1 = (score1 > np.quantile(score1, 0.8)).astype(int)

    score2 = (
        pois[:, 0]
        + 0.5 * bino[:, 1]
        - 0.8 * cat[:, 2]
        + np.sin(cont_log[:, 0] / 3.0)
        + 0.3 * derived_aux[:, 1]
    )
    bins2 = np.quantile(score2, [0, 0.2, 0.5, 0.8, 1])
    y2 = np.digitize(score2, bins2[1:-1])

    score3 = (
        cont_exp[:, 0]
        - cont_exp[:, 1]
        + derived_aux[:, 2]
        - derived_main[:, 2]
        + rng.normal(scale=0.3, size=n)
    )
    y3 = (score3 > np.median(score3)).astype(int)

    targets = np.column_stack([y1, y2, y3])
    task_classes = [2, 4, 2]
    features, mask = _apply_initial_missing(
        X,
        rng,
        mcar=0.08,
        mnar_indices=[2],
        y_reference=y1,
        mnar_prob=0.4,
    )
    return TaskData(features, targets, task_classes, raw_cols, derived_cols, mask, seed)


def _load_hard_progression(rng: np.random.Generator, seed: int) -> TaskData:
    X, comps, raw_cols, derived_cols = _hard_common_features(rng)
    cont_norm = comps["cont_norm"]
    cont_uni = comps["cont_uni"]
    derived_main = comps["derived_main"]
    derived_aux = comps["derived_aux"]
    noise = comps["noise"]
    cat = comps["cat"]
    bino = comps["bino"]
    pois = comps["pois"]
    n = X.shape[0]

    sum_norm = cont_norm[:, :4].sum(axis=1)
    sum_uni = cont_uni[:, :4].sum(axis=1)
    score1 = sum_norm - 0.5 * sum_uni + derived_aux[:, 3] + rng.normal(scale=0.3, size=n)
    y1 = (score1 > np.quantile(score1, 0.85)).astype(int)

    score2 = derived_main[:, 0] + derived_main[:, 3] + derived_aux[:, 4] + 0.1 * noise[:, 1]
    bins2 = np.quantile(score2, [0, 0.2, 0.4, 0.6, 0.8, 1])
    y2 = np.digitize(score2, bins2[1:-1])

    score3 = (
        (cat[:, 0] == 0).astype(float)
        + (cat[:, 1] == 1).astype(float)
        + (bino[:, 2] > 3).astype(float)
        + (pois[:, 3] > 2).astype(float)
    )
    y3 = (score3 >= 2).astype(int)

    targets = np.column_stack([y1, y2, y3])
    task_classes = [2, 5, 2]
    y2_high = (y2 >= 3).astype(int)
    features, mask = _apply_initial_missing(
        X,
        rng,
        mcar=0.12,
        mnar_indices=[3, 7],
        y_reference=y2_high,
        mnar_prob=0.45,
    )
    return TaskData(features, targets, task_classes, raw_cols, derived_cols, mask, seed)


HARD_TASK_SPECS: List[TaskSpec] = [
    TaskSpec(
        name="hard-core",
        loader=_load_hard_core,
        tags=("hard",),
        description="Baseline hard task with mixed feature types and MNAR missingness.",
    ),
    TaskSpec(
        name="hard-heterogeneous",
        loader=_load_hard_heterogeneous,
        tags=("hard",),
        description="Emphasises heterogeneity through discrete-heavy targets.",
    ),
    TaskSpec(
        name="hard-progression",
        loader=_load_hard_progression,
        tags=("hard",),
        description="Progression-style task with multi-class outcomes.",
    ),
]

TASK_REGISTRY: Dict[str, TaskSpec] = {spec.name: spec for spec in HARD_TASK_SPECS}


def get_hard_task_specs() -> List[TaskSpec]:
    """Return all task specifications tagged as "hard"."""

    return [spec for spec in HARD_TASK_SPECS if "hard" in spec.tags]


def load_task(name: str, seed: int | None = None) -> TaskData:
    """Load a task by name."""

    try:
        spec = TASK_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown task: {name}") from exc
    return spec.load(seed=seed)


def make_missing_variant(task: TaskData, spec: MissingnessSpec, *, seed_offset: int = 0) -> TaskData:
    """Create a new :class:`TaskData` instance with additional missingness."""

    features, mask = apply_missingness(
        task.features,
        task.raw_cols,
        task.derived_cols,
        spec,
        seed=task.seed + seed_offset,
    )
    return TaskData(
        features=features,
        targets=task.targets,
        task_classes=list(task.task_classes),
        raw_cols=list(task.raw_cols),
        derived_cols=list(task.derived_cols),
        base_missing_mask=mask,
        seed=task.seed + seed_offset,
    )
