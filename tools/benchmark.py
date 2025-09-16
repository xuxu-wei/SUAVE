"""Run the SUAVE benchmark suite across hard and hard-missing tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd

from tests.utils.benchmark_runner import BenchmarkOutcome, run_benchmark_for_task
from tests.utils.benchmark_tasks import (
    DEFAULT_SEED,
    get_hard_task_configs,
    get_missing_task_generators,
)

BENCHMARK_MODELS: List[str] = [
    "Linear",
    "SVM",
    "RandomForest",
    "KNN",
    "suave",
    "suave-single",
    "suave-impute",
    "autogluon",
]


def _ensure_autogluon() -> bool:
    try:  # pragma: no cover - exercised via CLI
        import autogluon.tabular  # noqa: F401

        return True
    except Exception:  # pragma: no cover - optional dependency
        print("AutoGluon not found. Attempting to install autogluon.tabular...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "autogluon.tabular"],
            check=False,
        )
        if result.returncode != 0:
            print("AutoGluon installation failed, continuing without it.")
            return False
        try:
            import autogluon.tabular  # noqa: F401

            print("AutoGluon installed successfully.")
            return True
        except Exception as err:  # pragma: no cover - optional dependency
            print(f"AutoGluon import still failing ({err!r}), skipping it.")
            return False


def _build_payload(outcomes: Iterable[BenchmarkOutcome]) -> Dict[str, object]:
    tasks: Dict[str, Dict[str, Dict[str, float]]] = {}
    seeds: Dict[str, int] = {}
    reconstruction: Dict[str, float] = {}
    skipped: Dict[str, Dict[str, str]] = {}
    for outcome in outcomes:
        seeds[outcome.name] = outcome.split_random_state
        if outcome.reconstruction_loss is not None:
            reconstruction[outcome.name] = outcome.reconstruction_loss
        if outcome.suave_metrics:
            tasks[outcome.name] = outcome.suave_metrics
        if outcome.skipped_models:
            skipped[outcome.name] = outcome.skipped_models
    payload: Dict[str, object] = {"tasks": tasks, "seeds": seeds}
    if reconstruction:
        payload["reconstruction"] = reconstruction
    if skipped:
        payload["skipped"] = skipped
    return payload


def _summarise(outcome: BenchmarkOutcome) -> None:
    if outcome.auc_table.empty:
        return
    table = outcome.auc_table.pivot(index="model", columns="task", values="auc")
    print(f"\nResults for {outcome.name}:\n{table.to_markdown()}")


def _collect_tasks() -> List[Tuple[str, Callable]]:
    tasks: List[Tuple[str, Callable]] = []
    for spec in get_hard_task_configs().values():
        tasks.append((spec.name, spec.generator))
    for name, generator in get_missing_task_generators().items():
        tasks.append((name, generator))
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SUAVE hard benchmark suite.")
    parser.add_argument("--suave-epochs", type=int, default=20)
    parser.add_argument("--suave-patience", type=int, default=5)
    parser.add_argument("--autogluon-time-limit", type=int, default=120)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baselines"))
    parser.add_argument("--skip-autogluon-install", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    autogluon_enabled = False
    if not args.skip_autogluon_install:
        autogluon_enabled = _ensure_autogluon()
    else:
        try:
            import autogluon.tabular  # noqa: F401

            autogluon_enabled = True
        except Exception:
            autogluon_enabled = False
    tasks = _collect_tasks()
    outcomes: List[BenchmarkOutcome] = []
    total = len(tasks)
    for idx, (name, generator) in enumerate(tasks, start=1):
        print(f"[{idx}/{total}] Running benchmark for {name}")
        data = generator(seed=args.split_seed)
        outcome = run_benchmark_for_task(
            data,
            model_names=BENCHMARK_MODELS,
            autogluon_enabled=autogluon_enabled,
            autogluon_time_limit=args.autogluon_time_limit,
            split_seed=args.split_seed,
            suave_epochs=args.suave_epochs,
            suave_patience=args.suave_patience,
        )
        outcomes.append(outcome)
        _summarise(outcome)
        if outcome.skipped_models:
            print(f"Skipped models for {name}: {outcome.skipped_models}")
    payload = _build_payload(outcomes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate.json"
    candidate_path.write_text(json.dumps(payload, indent=2))
    current_path = args.output_dir / "current.json"
    if not current_path.exists():
        current_path.write_text(candidate_path.read_text())
    print(f"\nBenchmark complete. Results saved to {candidate_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
