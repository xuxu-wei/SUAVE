"""Shared utilities for running benchmark tasks and aggregating results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from .benchmark_models import (
    CLASSICAL_MODEL_FACTORIES,
    MODEL_RANDOM_STATE,
    SVM_MAX_TRAIN_SAMPLES,
    build_suave_classifier,
    build_suave_single,
)
from .benchmark_tasks import TaskData

try:  # pragma: no cover - optional dependency
    from autogluon.tabular import TabularPredictor

    HAS_AUTOGLOON = True
except Exception:  # pragma: no cover
    HAS_AUTOGLOON = False


@dataclass
class BenchmarkOutcome:
    """Results for a single benchmark task."""

    name: str
    auc_table: pd.DataFrame
    suave_metrics: Dict[str, Dict[str, float]]
    reconstruction_loss: Optional[float]
    split_random_state: int
    skipped_models: Dict[str, str]
    task_classes: List[int]
    seed: Optional[int]


def split_task_data(
    data: TaskData,
    *,
    rng_seed: int = MODEL_RANDOM_STATE,
    test_size: float = 0.2,
    max_attempts: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Create a stratified train/test split ensuring all classes are present."""

    rng = np.random.default_rng(rng_seed)
    for _ in range(max_attempts):
        random_state = int(rng.integers(0, 1_000_000))
        X_train, X_test, y_train, y_test = train_test_split(
            data.features,
            data.targets,
            test_size=test_size,
            random_state=random_state,
        )
        if all(
            np.unique(y_train[:, idx]).size >= cls
            for idx, cls in enumerate(data.task_classes)
        ):
            return X_train, X_test, y_train, y_test, random_state
    raise RuntimeError("Failed to generate a split covering all classes")


def _ensure_two_column_proba(proba: np.ndarray) -> np.ndarray:
    if proba.ndim == 1:
        return np.column_stack([1 - proba, proba])
    if proba.shape[1] == 1:
        return np.column_stack([1 - proba[:, 0], proba[:, 0]])
    return proba


def _prepare_proba(proba: np.ndarray, num_classes: int) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if num_classes == 2:
        proba = _ensure_two_column_proba(proba)
    proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = proba.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).ravel()
    if zero_rows.any():
        proba[zero_rows, :] = 1.0 / num_classes
    non_zero = ~zero_rows
    if non_zero.any():
        proba[non_zero, :] = proba[non_zero, :] / row_sums[non_zero]
    return proba


def compute_auc_metric(y_true: np.ndarray, proba: np.ndarray, num_classes: int) -> float:
    if num_classes > 2:
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    proba = _ensure_two_column_proba(proba)
    return float(roc_auc_score(y_true, proba[:, 1]))


def compute_task_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if num_classes > 2:
        classes = np.arange(num_classes)
        y_true_bin = label_binarize(y_true, classes=classes)
        metrics["auroc_macro"] = float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        )
        metrics["auroc_micro"] = float(roc_auc_score(y_true_bin, proba, average="micro"))
        metrics["auprc_macro"] = float(average_precision_score(y_true_bin, proba, average="macro"))
        metrics["auprc_micro"] = float(average_precision_score(y_true_bin, proba, average="micro"))
        metrics["acc_top1"] = float(accuracy_score(y_true, pred))
        metrics["f1_macro"] = float(f1_score(y_true, pred, average="macro"))
    else:
        metrics["auroc_macro"] = float(roc_auc_score(y_true, proba[:, 1]))
        metrics["auroc_micro"] = metrics["auroc_macro"]
        metrics["auprc_macro"] = float(average_precision_score(y_true, proba[:, 1]))
        metrics["auprc_micro"] = metrics["auprc_macro"]
        metrics["acc_top1"] = float(accuracy_score(y_true, pred))
        metrics["f1_macro"] = float(f1_score(y_true, pred))
    return metrics


def _evaluate_classical_model(
    name: str,
    factory: Callable[[], object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_classes: Sequence[int],
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for idx, num_classes in enumerate(task_classes):
        model = factory()
        y_tr = y_train[:, idx]
        X_tr = X_train
        if name == "SVM" and len(X_tr) > SVM_MAX_TRAIN_SAMPLES:
            sample_idx = rng.choice(len(X_tr), SVM_MAX_TRAIN_SAMPLES, replace=False)
            X_tr = X_tr[sample_idx]
            y_tr = y_tr[sample_idx]
        model.fit(X_tr, y_tr)
        proba = _prepare_proba(model.predict_proba(X_test), num_classes)
        auc = compute_auc_metric(y_test[:, idx], proba, num_classes)
        rows.append({"model": name, "task": f"y{idx + 1}", "auc": auc})
    return rows


def _evaluate_suave(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_classes: Sequence[int],
    *,
    epochs: int,
    patience: int,
) -> tuple[List[Dict[str, float]], Dict[str, Dict[str, float]], Optional[float]]:
    model = build_suave_classifier(X_train.shape[1], list(task_classes))
    model.fit(X_train, y_train, epochs=epochs, patience=patience, verbose=False)
    probas = model.predict_proba(X_test)
    preds = model.predict(X_test)
    metrics: Dict[str, Dict[str, float]] = {}
    rows: List[Dict[str, float]] = []
    for idx, num_classes in enumerate(task_classes):
        y_true = y_test[:, idx]
        proba = _prepare_proba(probas[idx], num_classes)
        pred = np.asarray(preds[idx])
        auc = compute_auc_metric(y_true, proba, num_classes)
        task_metrics = compute_task_metrics(y_true, proba, pred, num_classes)
        task_metrics["auc"] = float(auc)
        metrics[f"y{idx + 1}"] = task_metrics
        rows.append({"model": "suave", "task": f"y{idx + 1}", "auc": float(auc)})
    recon_loss: Optional[float] = None
    if model.models:
        _, recon_loss_val, *_ = model.models[0].eval_loss(X_test, y_test[:, 0])
        recon_loss = float(recon_loss_val)
        if not np.isfinite(recon_loss):
            recon_loss = None
    return rows, metrics, recon_loss


def _evaluate_suave_impute(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_classes: Sequence[int],
    *,
    epochs: int,
    patience: int,
) -> List[Dict[str, float]]:
    imputer = IterativeImputer(random_state=MODEL_RANDOM_STATE)
    X_tr = imputer.fit_transform(X_train)
    X_te = imputer.transform(X_test)
    model = build_suave_classifier(X_tr.shape[1], list(task_classes))
    model.fit(X_tr, y_train, epochs=epochs, patience=patience, verbose=False)
    probas = model.predict_proba(X_te)
    rows = []
    for idx, num_classes in enumerate(task_classes):
        proba = _prepare_proba(probas[idx], num_classes)
        auc = compute_auc_metric(y_test[:, idx], proba, num_classes)
        rows.append({"model": "suave-impute", "task": f"y{idx + 1}", "auc": float(auc)})
    return rows


def _evaluate_suave_single(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_classes: Sequence[int],
    *,
    epochs: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for idx, num_classes in enumerate(task_classes):
        model = build_suave_single(X_train.shape[1], num_classes)
        model.fit(X_train, y_train[:, idx], epochs=epochs)
        proba = _prepare_proba(model.predict_proba(X_test), num_classes)
        auc = compute_auc_metric(y_test[:, idx], proba, num_classes)
        rows.append({"model": "suave-single", "task": f"y{idx + 1}", "auc": auc})
    return rows


def _evaluate_autogluon(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_classes: Sequence[int],
    *,
    time_limit: int,
) -> List[Dict[str, float]]:
    if not HAS_AUTOGLOON:
        raise RuntimeError("AutoGluon is not available")
    rows: List[Dict[str, float]] = []
    for idx, num_classes in enumerate(task_classes):
        train_df = pd.DataFrame(X_train)
        train_df["label"] = y_train[:, idx]
        test_df = pd.DataFrame(X_test)
        predictor = TabularPredictor(label="label", verbosity=0).fit(train_df, time_limit=time_limit)
        proba_df = predictor.predict_proba(test_df)
        proba = _prepare_proba(proba_df.to_numpy(), num_classes)
        auc = compute_auc_metric(y_test[:, idx], proba, num_classes)
        rows.append({"model": "autogluon", "task": f"y{idx + 1}", "auc": auc})
    return rows


def run_benchmark_for_task(
    data: TaskData,
    *,
    model_names: Sequence[str],
    autogluon_enabled: bool = False,
    autogluon_time_limit: int = 60,
    split_seed: int = MODEL_RANDOM_STATE,
    suave_epochs: int = 20,
    suave_patience: int = 5,
) -> BenchmarkOutcome:
    """Execute the benchmark suite for a single task definition."""

    X_train, X_test, y_train, y_test, split_random_state = split_task_data(
        data, rng_seed=split_seed
    )
    rows: List[Dict[str, float]] = []
    suave_metrics: Dict[str, Dict[str, float]] = {}
    recon_loss: Optional[float] = None
    skipped: Dict[str, str] = {}
    model_rng = np.random.default_rng(split_random_state + MODEL_RANDOM_STATE)

    for name in model_names:
        if name == "suave":
            suave_rows, suave_metrics, recon_loss = _evaluate_suave(
                X_train,
                y_train,
                X_test,
                y_test,
                data.task_classes,
                epochs=suave_epochs,
                patience=suave_patience,
            )
            rows.extend(suave_rows)
        elif name == "suave-impute":
            rows.extend(
                _evaluate_suave_impute(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    data.task_classes,
                    epochs=suave_epochs,
                    patience=suave_patience,
                )
            )
        elif name == "suave-single":
            rows.extend(
                _evaluate_suave_single(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    data.task_classes,
                    epochs=suave_epochs,
                )
            )
        elif name == "autogluon":
            if autogluon_enabled:
                try:
                    rows.extend(
                        _evaluate_autogluon(
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            data.task_classes,
                            time_limit=autogluon_time_limit,
                        )
                    )
                except Exception as err:  # pragma: no cover - optional dependency
                    skipped[name] = str(err)
            else:
                skipped[name] = "autogluon disabled"
        elif name in CLASSICAL_MODEL_FACTORIES:
            rows.extend(
                _evaluate_classical_model(
                    name,
                    CLASSICAL_MODEL_FACTORIES[name],
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    data.task_classes,
                    model_rng,
                )
            )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown model: {name}")

    auc_table = pd.DataFrame(rows)
    return BenchmarkOutcome(
        name=str(data.metadata.get("name", "task")),
        auc_table=auc_table,
        suave_metrics=suave_metrics,
        reconstruction_loss=recon_loss,
        split_random_state=split_random_state,
        skipped_models=skipped,
        task_classes=list(data.task_classes),
        seed=data.metadata.get("seed"),
    )
