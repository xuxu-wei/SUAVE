"""Smoke tests exercising the benchmark stack with missingness variants."""

from __future__ import annotations

import numpy as np

from tests.utils.benchmarking import prepare_data_split, run_suave_models
from tests.utils.missingness import MISSING_VARIANTS
from tests.utils.task_registry import TaskData, get_hard_task_specs, make_missing_variant

MAX_SMOKE_SAMPLES = 800


def _subset_task(task: TaskData, max_samples: int, seed: int) -> TaskData:
    if task.features.shape[0] <= max_samples:
        return task
    rng = np.random.default_rng(seed)
    idx = rng.choice(task.features.shape[0], max_samples, replace=False)
    return TaskData(
        features=task.features[idx],
        targets=task.targets[idx],
        task_classes=list(task.task_classes),
        raw_cols=list(task.raw_cols),
        derived_cols=list(task.derived_cols),
        base_missing_mask=task.base_missing_mask[idx],
        seed=task.seed,
    )


def test_missing_variants_suave_smoke() -> None:
    """Train SUAVE on lite/heavy missing variants to ensure wiring works."""

    hard_specs = get_hard_task_specs()
    assert hard_specs, "Expected at least one hard task specification"
    task_spec = hard_specs[0]
    base_seed = 123
    subset_seed = 321
    base_task = task_spec.load(seed=base_seed)
    base_task = _subset_task(base_task, MAX_SMOKE_SAMPLES, subset_seed)

    for offset, (variant_name, spec) in enumerate(MISSING_VARIANTS.items(), start=1):
        variant = make_missing_variant(base_task, spec, seed_offset=offset)
        X_train, X_test, y_train, y_test, split_seed = prepare_data_split(
            variant,
            seed=base_seed + offset,
            test_size=0.3,
            max_attempts=25,
        )
        results = run_suave_models(
            X_train,
            X_test,
            y_train,
            y_test,
            variant.task_classes,
            variants=("suave",),
            latent_dim=6,
            epochs=5,
            base_seed=variant.seed + split_seed,
        )
        suave_metrics = results["suave"]["metrics"]
        for task_name, task_metrics in suave_metrics.items():
            assert task_metrics, f"No metrics computed for {task_name} ({variant_name})"
            for metric_name, value in task_metrics.items():
                assert 0.0 <= value <= 1.0, f"Metric {metric_name} out of range for {variant_name}"
