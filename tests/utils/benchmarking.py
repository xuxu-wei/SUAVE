"""Shared benchmarking utilities for tests and tooling."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from tests.utils.models import (
    SingleTaskSuave,
    SuaveImputeWrapper,
    create_baseline_model,
    create_suave_classifier,
)
from tests.utils.task_registry import TaskData

__all__ = [
    "compute_task_metrics",
    "prepare_data_split",
    "run_baseline_models",
    "run_suave_models",
]


def compute_task_metrics(
    y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray, num_classes: int
) -> Dict[str, float]:
    """Compute standard classification metrics for a task."""

    if num_classes > 2:
        classes = np.arange(num_classes)
        y_true_bin = label_binarize(y_true, classes=classes)
        metrics = {
            "auroc_macro": float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
            ),
            "auroc_micro": float(roc_auc_score(y_true_bin, proba, average="micro")),
            "auprc_macro": float(average_precision_score(y_true_bin, proba, average="macro")),
            "auprc_micro": float(average_precision_score(y_true_bin, proba, average="micro")),
            "acc_top1": float(accuracy_score(y_true, pred)),
            "f1_macro": float(f1_score(y_true, pred, average="macro")),
        }
    else:
        metrics = {
            "auroc_macro": float(roc_auc_score(y_true, proba[:, 1])),
            "auroc_micro": float(roc_auc_score(y_true, proba[:, 1])),
            "auprc_macro": float(average_precision_score(y_true, proba[:, 1])),
            "auprc_micro": float(average_precision_score(y_true, proba[:, 1])),
            "acc_top1": float(accuracy_score(y_true, pred)),
            "f1_macro": float(f1_score(y_true, pred, average="macro")),
        }
    return metrics


def prepare_data_split(
    task: TaskData,
    *,
    test_size: float = 0.2,
    max_attempts: int = 50,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Prepare a train/test split ensuring all classes are represented."""

    rng = np.random.default_rng(seed)
    X, Y = task.features, task.targets
    for _ in range(max_attempts):
        random_state = int(rng.integers(0, 1_000_000))
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )
        if all(
            np.unique(y_train[:, idx]).size == classes
            for idx, classes in enumerate(task.task_classes)
        ):
            return X_train, X_test, y_train, y_test, random_state
    raise RuntimeError("Failed to produce a valid split after multiple attempts")


def run_baseline_models(
    model_names: Sequence[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_classes: Sequence[int],
    *,
    random_state: int,
) -> List[Dict[str, object]]:
    """Evaluate classical baselines with imputation pipelines."""

    rng = np.random.default_rng(random_state)
    results: List[Dict[str, object]] = []
    for task_idx, num_classes in enumerate(task_classes):
        ytr = y_train[:, task_idx]
        yte = y_test[:, task_idx]
        for name in model_names:
            model_key = name.lower()
            model_seed = random_state + 31 * (task_idx + 1) + len(results)
            estimator = create_baseline_model(model_key, random_state=model_seed)
            Xtr = X_train
            ysub = ytr
            if model_key == "svm" and X_train.shape[0] > 1200:
                idx = rng.choice(X_train.shape[0], 1200, replace=False)
                Xtr = X_train[idx]
                ysub = ytr[idx]
            estimator.fit(Xtr, ysub)
            proba = estimator.predict_proba(X_test)
            pred = estimator.predict(X_test)
            metrics = compute_task_metrics(yte, proba, pred, num_classes)
            results.append(
                {
                    "model": name,
                    "task": f"y{task_idx + 1}",
                    "metrics": metrics,
                }
            )
    return results


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_suave_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_classes: Sequence[int],
    *,
    variants: Iterable[str] = ("suave", "suave-impute", "suave-single"),
    latent_dim: int = 8,
    epochs: int = 20,
    batch_size: int | None = None,
    base_seed: int = 0,
) -> Dict[str, Dict[str, object]]:
    """Train and evaluate SUAVE-based models."""

    results: Dict[str, Dict[str, object]] = {}
    variant_set = set(variants)
    input_dim = X_train.shape[1]

    if "suave" in variant_set:
        _set_seed(base_seed)
        model = create_suave_classifier(input_dim, list(task_classes), latent_dim=latent_dim)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        probas = model.predict_proba(X_test)
        preds = model.predict(X_test)
        metrics = {
            f"y{idx + 1}": compute_task_metrics(y_test[:, idx], probas[idx], preds[idx], num_classes)
            for idx, num_classes in enumerate(task_classes)
        }
        score = {
            f"y{idx + 1}": float(val)
            for idx, val in enumerate(model.score(X_test, y_test))
        }
        results["suave"] = {"metrics": metrics, "score": score}

    if "suave-impute" in variant_set:
        _set_seed(base_seed + 1)
        wrapper = SuaveImputeWrapper(
            input_dim=input_dim,
            task_classes=list(task_classes),
            latent_dim=latent_dim,
            random_state=base_seed + 1,
        )
        wrapper.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        probas = wrapper.predict_proba(X_test)
        preds = wrapper.predict(X_test)
        metrics = {
            f"y{idx + 1}": compute_task_metrics(y_test[:, idx], probas[idx], preds[idx], num_classes)
            for idx, num_classes in enumerate(task_classes)
        }
        score = {
            f"y{idx + 1}": float(val)
            for idx, val in enumerate(wrapper.score(X_test, y_test))
        }
        results["suave-impute"] = {"metrics": metrics, "score": score}

    if "suave-single" in variant_set:
        single_metrics: Dict[str, Dict[str, float]] = {}
        for idx, num_classes in enumerate(task_classes):
            _set_seed(base_seed + 2 + idx)
            model = SingleTaskSuave(input_dim=input_dim, num_classes=num_classes, latent_dim=latent_dim)
            model.fit(X_train, y_train[:, idx], epochs=epochs, batch_size=batch_size)
            proba = model.predict_proba(X_test)
            pred = model.predict(X_test)
            single_metrics[f"y{idx + 1}"] = compute_task_metrics(
                y_test[:, idx], proba, pred, num_classes
            )
        results["suave-single"] = {"metrics": single_metrics}

    return results
