"""Model factories used across the benchmark utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from suave.api import SUAVE
from suave.sklearn import SuaveClassifier

__all__ = [
    "SingleTaskSuave",
    "SuaveImputeWrapper",
    "create_baseline_model",
    "create_suave_classifier",
]


@dataclass
class SingleTaskSuave:
    """Wrapper around :class:`suave.api.SUAVE` for single-task evaluation."""

    input_dim: int
    num_classes: int
    latent_dim: int = 8

    def __post_init__(self) -> None:
        self.model = SUAVE(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            num_classes=self.num_classes,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        *,
        batch_size: int | None = None,
    ) -> "SingleTaskSuave":
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        proba = self.predict_proba(X)
        if self.num_classes > 2:
            return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
        classes = np.unique(y)
        if proba.ndim == 1 or proba.shape[1] == 1:
            scores = proba.squeeze()
        else:
            pos_label = classes[-1]
            class_order = classes if classes.size == proba.shape[1] else np.arange(proba.shape[1])
            matches = np.flatnonzero(class_order == pos_label)
            pos_idx = int(matches[0]) if matches.size else proba.shape[1] - 1
            scores = proba[:, pos_idx]
        return float(roc_auc_score(y, scores))


class SuaveImputeWrapper:
    """Combine :class:`IterativeImputer` with :class:`SuaveClassifier`."""

    def __init__(
        self,
        input_dim: int,
        task_classes: List[int],
        *,
        latent_dim: int = 8,
        random_state: int | None = None,
    ) -> None:
        self.task_classes = task_classes
        self.imputer = IterativeImputer(random_state=random_state)
        self.model = SuaveClassifier(
            input_dim=input_dim,
            task_classes=task_classes,
            latent_dim=latent_dim,
        )

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 20,
        *,
        batch_size: int | None = None,
    ) -> "SuaveImputeWrapper":
        X_imp = self.imputer.fit_transform(X)
        self.model.fit(X_imp, Y, epochs=epochs, batch_size=batch_size)
        return self

    def predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        X_imp = self.imputer.transform(X)
        return self.model.predict_proba(X_imp)

    def predict(self, X: np.ndarray) -> List[np.ndarray]:
        X_imp = self.imputer.transform(X)
        return self.model.predict(X_imp)

    def score(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X_imp = self.imputer.transform(X)
        return self.model.score(X_imp, Y)


def create_baseline_model(name: str, random_state: int | None = None) -> Pipeline:
    """Create a scikit-learn baseline wrapped with :class:`IterativeImputer`."""

    key = name.lower()
    imputer = IterativeImputer(random_state=random_state)
    if key == "linear":
        estimator = LogisticRegression(max_iter=1000, random_state=random_state)
        steps = [
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    elif key == "svm":
        estimator = SVC(kernel="rbf", probability=True, random_state=random_state)
        steps = [
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    elif key == "knn":
        estimator = KNeighborsClassifier()
        steps = [
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    elif key == "randomforest":
        estimator = RandomForestClassifier(n_estimators=200, random_state=random_state)
        steps = [
            ("imputer", imputer),
            ("clf", estimator),
        ]
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unknown model: {name}")
    return Pipeline(steps)


def create_suave_classifier(
    input_dim: int,
    task_classes: List[int],
    *,
    latent_dim: int = 8,
) -> SuaveClassifier:
    """Instantiate :class:`SuaveClassifier` for the provided task."""

    return SuaveClassifier(input_dim=input_dim, task_classes=task_classes, latent_dim=latent_dim)
