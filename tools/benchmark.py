"""Run the hard benchmark suite including missingness variants."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tests.utils.benchmarking import (
    compute_task_metrics,
    prepare_data_split,
    run_baseline_models,
    run_suave_models,
)
from tests.utils.missingness import MISSING_VARIANTS
from tests.utils.task_registry import TaskData, get_hard_task_specs, make_missing_variant

DEFAULT_CANDIDATE = Path("reports/baselines/candidate.json")
DEFAULT_CURRENT = Path("reports/baselines/current.json")
BASELINE_MODELS = ["linear", "SVM", "RandomForest", "KNN"]
SUAVE_VARIANTS = ("suave", "suave-impute", "suave-single")


@dataclass
class BenchmarkConfig:
    epochs: int
    latent_dim: int
    batch_size: Optional[int]
    max_train_samples: Optional[int]
    autogluon_time_limit: int
    output: Path
    current: Path


def _ensure_autogluon() -> Tuple[Optional[object], Optional[str]]:
    try:
        from autogluon.tabular import TabularPredictor  # type: ignore

        return TabularPredictor, None
    except Exception:  # pragma: no cover - import guard
        print("[benchmark] autogluon not found, attempting installation", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "autogluon.tabular"],
            check=False,
        )
        if result.returncode != 0:
            return None, "install_failed"
        try:
            from autogluon.tabular import TabularPredictor  # type: ignore

            return TabularPredictor, None
        except Exception as exc:  # pragma: no cover - defensive
            return None, f"import_failed:{exc}"  # pylint: disable=unsubscriptable-object


def _maybe_subsample(
    X: np.ndarray, Y: np.ndarray, max_samples: Optional[int], seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples is None or X.shape[0] <= max_samples:
        return X, Y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], max_samples, replace=False)
    return X[idx], Y[idx]


def _group_baseline_results(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    for row in rows:
        name = str(row["model"])
        task = str(row["task"])
        metrics = row["metrics"]
        entry = grouped.setdefault(name, {"metrics": {}})
        entry["metrics"][task] = metrics
    return grouped


def _evaluate_autogluon(
    predictor_cls: Optional[object],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_classes: Sequence[int],
    *,
    time_limit: int,
    seed: int,
) -> Dict[str, object]:
    if predictor_cls is None:
        return {"status": "skipped"}
    results: Dict[str, Dict[str, float]] = {}
    try:
        for idx, num_classes in enumerate(task_classes):
            label = "label"
            train_df = pd.DataFrame(X_train)
            train_df[label] = y_train[:, idx]
            test_df = pd.DataFrame(X_test)
            with tempfile.TemporaryDirectory() as tmpdir:
                predictor = predictor_cls(label=label, path=tmpdir, verbosity=0)
                predictor.fit(
                    train_df,
                    time_limit=time_limit,
                    presets="medium_quality_faster_train",
                )
                proba_df = predictor.predict_proba(test_df)
                proba = proba_df.to_numpy()
                pred = predictor.predict(test_df).to_numpy()
                metrics = compute_task_metrics(y_test[:, idx], proba, pred, num_classes)
                results[f"y{idx + 1}"] = metrics
        return {"status": "ok", "metrics": results, "time_limit": time_limit}
    except Exception as exc:  # pragma: no cover - defensive guard
        return {"status": "error", "error": str(exc)}


def _evaluate_task(
    task_name: str,
    task: TaskData,
    *,
    kind: str,
    config: BenchmarkConfig,
    predictor_cls: Optional[object],
    base_task: str | None = None,
    variant: str | None = None,
) -> Dict[str, object]:
    print(f"[benchmark] running {task_name} ({kind})", flush=True)
    X_train, X_test, y_train, y_test, split_seed = prepare_data_split(task, seed=task.seed)
    X_train_sub, y_train_sub = _maybe_subsample(
        X_train,
        y_train,
        config.max_train_samples,
        seed=task.seed + split_seed,
    )
    suave = run_suave_models(
        X_train_sub,
        X_test,
        y_train_sub,
        y_test,
        task.task_classes,
        variants=SUAVE_VARIANTS,
        latent_dim=config.latent_dim,
        epochs=config.epochs,
        batch_size=config.batch_size,
        base_seed=task.seed + split_seed,
    )
    baselines = _group_baseline_results(
        run_baseline_models(
            BASELINE_MODELS,
            X_train_sub,
            X_test,
            y_train_sub,
            y_test,
            task.task_classes,
            random_state=task.seed + split_seed,
        )
    )
    autogluon_entry = _evaluate_autogluon(
        predictor_cls,
        X_train_sub,
        X_test,
        y_train_sub,
        y_test,
        task.task_classes,
        time_limit=config.autogluon_time_limit,
        seed=task.seed + split_seed,
    )
    models: Dict[str, object] = {}
    models.update(suave)
    models.update(baselines)
    models["autogluon"] = autogluon_entry
    result: Dict[str, object] = {
        "kind": kind,
        "seed": int(task.seed),
        "split_random_state": int(split_seed),
        "train_samples": int(X_train_sub.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "models": models,
    }
    if base_task is not None:
        result["base_task"] = base_task
    if variant is not None:
        result["variant"] = variant
    return result


def run_benchmark(config: BenchmarkConfig) -> Dict[str, object]:
    predictor_cls, ag_status = _ensure_autogluon()
    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "latent_dim": config.latent_dim,
        "max_train_samples": config.max_train_samples,
        "autogluon_status": ag_status if ag_status else "available",
        "models": SUAVE_VARIANTS + tuple(BASELINE_MODELS) + ("autogluon",),
    }
    tasks_payload: Dict[str, object] = {}

    for spec in get_hard_task_specs():
        base_task = spec.load()
        base_name = spec.name
        tasks_payload[base_name] = _evaluate_task(
            base_name,
            base_task,
            kind="base",
            config=config,
            predictor_cls=predictor_cls,
        )
        for offset, (variant_name, missing_spec) in enumerate(MISSING_VARIANTS.items(), start=1):
            variant_task = make_missing_variant(base_task, missing_spec, seed_offset=offset)
            variant_key = f"{base_name}-missing-{variant_name}"
            tasks_payload[variant_key] = _evaluate_task(
                variant_key,
                variant_task,
                kind="missing",
                config=config,
                predictor_cls=predictor_cls,
                base_task=base_name,
                variant=variant_name,
            )

    return {"metadata": metadata, "tasks": tasks_payload}


def _update_current(candidate: Dict[str, object], current_path: Path) -> None:
    if not current_path.exists():
        current_path.parent.mkdir(parents=True, exist_ok=True)
        current_path.write_text(json.dumps(candidate, indent=2))
        return
    with current_path.open() as f:
        current = json.load(f)
    current_tasks = current.setdefault("tasks", {})
    for name, entry in candidate.get("tasks", {}).items():
        if entry.get("kind") == "missing":
            current_tasks[name] = entry
    current["metadata"] = candidate.get("metadata", {})
    with current_path.open("w") as f:
        json.dump(current, f, indent=2)


def parse_args(argv: Iterable[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Run SUAVE hard benchmark")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for SUAVE models")
    parser.add_argument("--latent-dim", type=int, default=8, help="Latent dimension for SUAVE models")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size for SUAVE models (use -1 for full batch)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on the number of training samples",
    )
    parser.add_argument(
        "--autogluon-time-limit",
        type=int,
        default=120,
        help="Per-task time limit (seconds) for AutoGluon",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CANDIDATE,
        help="Path to candidate.json output",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=DEFAULT_CURRENT,
        help="Path to current.json baseline",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return BenchmarkConfig(
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        batch_size=None if args.batch_size in (-1, 0) else args.batch_size,
        max_train_samples=args.max_train_samples,
        autogluon_time_limit=args.autogluon_time_limit,
        output=args.output,
        current=args.current,
    )


def main(argv: Iterable[str] | None = None) -> int:
    config = parse_args(argv)
    candidate = run_benchmark(config)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with config.output.open("w") as f:
        json.dump(candidate, f, indent=2)
    _update_current(candidate, config.current)
    print(f"[benchmark] wrote candidate results to {config.output}")
    print(f"[benchmark] updated current baseline at {config.current}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
