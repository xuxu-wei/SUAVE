from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from tests.utils.benchmark_runner import BenchmarkOutcome, run_benchmark_for_task
from tests.utils.benchmark_tasks import (
    DEFAULT_SEED,
    generate_hard,
    get_missing_task_generators,
)

BASE_MODELS: List[str] = [
    "Linear",
    "SVM",
    "KNN",
    "RandomForest",
    "suave",
    "suave-single",
    "suave-impute",
]
MISSING_MODELS: List[str] = ["Linear", "suave", "suave-impute"]
BASELINE_DIR = Path("reports/baselines")


def _print_outcome(outcome: BenchmarkOutcome) -> None:
    if outcome.auc_table.empty:
        raise AssertionError(f"No results recorded for {outcome.name}")
    table = outcome.auc_table.pivot(index="model", columns="task", values="auc")
    print(f"\nBenchmark AUCs ({outcome.name}):\n", table.to_markdown())


def _print_reconstruction(outcomes: Iterable[BenchmarkOutcome]) -> None:
    rows = [
        {"dataset": outcome.name, "recon": outcome.reconstruction_loss}
        for outcome in outcomes
        if outcome.reconstruction_loss is not None
    ]
    if rows:
        recon_df = pd.DataFrame(rows)
        print("\nSUAVE Reconstruction Loss:\n", recon_df.to_markdown(index=False))


def _build_payload(outcomes: Iterable[BenchmarkOutcome]) -> dict:
    tasks = {}
    seeds = {}
    reconstruction = {}
    skipped = {}
    for outcome in outcomes:
        seeds[outcome.name] = outcome.split_random_state
        if outcome.reconstruction_loss is not None:
            reconstruction[outcome.name] = outcome.reconstruction_loss
        if outcome.suave_metrics:
            tasks[outcome.name] = outcome.suave_metrics
        if outcome.skipped_models:
            skipped[outcome.name] = outcome.skipped_models
    payload = {"tasks": tasks, "seeds": seeds}
    if reconstruction:
        payload["reconstruction"] = reconstruction
    if skipped:
        payload["skipped"] = skipped
    return payload


def _write_baseline(outcomes: Iterable[BenchmarkOutcome]) -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    payload = _build_payload(outcomes)
    cand_path = BASELINE_DIR / "candidate.json"
    cand_path.write_text(json.dumps(payload, indent=2))
    curr_path = BASELINE_DIR / "current.json"
    if not curr_path.exists():
        curr_path.write_text(cand_path.read_text())


def test_benchmarks() -> None:
    hard_outcome = run_benchmark_for_task(
        generate_hard(seed=DEFAULT_SEED),
        model_names=BASE_MODELS,
        autogluon_enabled=False,
        autogluon_time_limit=60,
        split_seed=DEFAULT_SEED,
        suave_epochs=20,
        suave_patience=5,
    )
    _print_outcome(hard_outcome)

    missing_outcomes: List[BenchmarkOutcome] = []
    for name, generator in get_missing_task_generators().items():
        outcome = run_benchmark_for_task(
            generator(seed=DEFAULT_SEED),
            model_names=MISSING_MODELS,
            autogluon_enabled=False,
            autogluon_time_limit=60,
            split_seed=DEFAULT_SEED,
            suave_epochs=15,
            suave_patience=5,
        )
        _print_outcome(outcome)
        missing_outcomes.append(outcome)

    outcomes = [hard_outcome, *missing_outcomes]
    _print_reconstruction(outcomes)
    _write_baseline(outcomes)


def test_hard_missing_smoke() -> None:
    generators = get_missing_task_generators()
    for name, generator in generators.items():
        outcome = run_benchmark_for_task(
            generator(seed=DEFAULT_SEED),
            model_names=["suave"],
            autogluon_enabled=False,
            autogluon_time_limit=30,
            split_seed=DEFAULT_SEED,
            suave_epochs=10,
            suave_patience=3,
        )
        assert not outcome.auc_table.empty, f"No results for {name}"
        assert "suave" in outcome.auc_table["model"].values
