"""Model factories used across benchmark suites."""

from __future__ import annotations

from typing import Callable, Dict

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from suave.api import SUAVE
from suave.sklearn import SuaveClassifier

MODEL_RANDOM_STATE = 20201021
SVM_MAX_TRAIN_SAMPLES = 1000


def _iterative_imputer() -> IterativeImputer:
    return IterativeImputer(random_state=MODEL_RANDOM_STATE)


def build_linear_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", _iterative_imputer()),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=MODEL_RANDOM_STATE),
            ),
        ]
    )


def build_svm_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", _iterative_imputer()),
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    probability=True,
                    max_iter=200,
                    random_state=MODEL_RANDOM_STATE,
                ),
            ),
        ]
    )


def build_knn_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", _iterative_imputer()),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ]
    )


def build_rf_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", _iterative_imputer()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=MODEL_RANDOM_STATE,
                ),
            ),
        ]
    )


CLASSICAL_MODEL_FACTORIES: Dict[str, Callable[[], Pipeline]] = {
    "Linear": build_linear_model,
    "SVM": build_svm_model,
    "KNN": build_knn_model,
    "RandomForest": build_rf_model,
}


def build_suave_classifier(input_dim: int, task_classes: list[int]) -> SuaveClassifier:
    return SuaveClassifier(input_dim=input_dim, task_classes=task_classes, latent_dim=8)


def build_suave_single(input_dim: int, num_classes: int) -> SUAVE:
    return SUAVE(input_dim=input_dim, latent_dim=8, num_classes=num_classes)
