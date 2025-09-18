from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suave import SUAVE
from suave.modules.calibrate import TemperatureScaler


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    normaliser = exp_logits.sum(axis=1, keepdims=True)
    normaliser[normaliser == 0.0] = 1.0
    return exp_logits / normaliser


def brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(targets)), targets] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probs: np.ndarray, targets: np.ndarray, *, n_bins: int = 10
) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        if end == 1.0:
            mask = (confidences >= start) & (confidences <= end)
        else:
            mask = (confidences >= start) & (confidences < end)
        if not np.any(mask):
            continue
        bin_confidence = float(confidences[mask].mean())
        bin_accuracy = float(accuracies[mask].mean())
        ece += abs(bin_confidence - bin_accuracy) * (mask.mean())
    return float(ece)


class DummySUAVE(SUAVE):
    def __init__(self) -> None:
        super().__init__()
        self._logit_lookup: dict[int, np.ndarray] = {}

    def set_logits(self, frame: pd.DataFrame, logits: np.ndarray) -> None:
        self._logit_lookup[id(frame)] = logits

    def _compute_logits(self, X: pd.DataFrame) -> np.ndarray:  # type: ignore[override]
        logits = self._logit_lookup.get(id(X))
        if logits is None:
            raise KeyError("Logits for the provided frame are not registered")
        self._cached_logits = logits
        return logits


@pytest.mark.parametrize("temperature", [2.5])
def test_temperature_scaling_improves_calibration(temperature: float) -> None:
    rng = np.random.default_rng(42)
    n_classes = 2

    def make_split(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        base_logits = rng.normal(size=(n_samples, n_classes)).astype(np.float32)
        base_logits[:, 1] += 0.6
        true_probs = softmax(base_logits)
        draws = [rng.choice(n_classes, p=prob) for prob in true_probs]
        logits = base_logits * temperature
        return logits.astype(np.float32), np.asarray(draws, dtype=np.int64)

    logits_cal, y_cal = make_split(2048)
    logits_test, y_test = make_split(4096)

    X_cal = pd.DataFrame({"split": np.zeros(len(y_cal), dtype=int)})
    X_test = pd.DataFrame({"split": np.ones(len(y_test), dtype=int)})

    model = DummySUAVE()
    model._is_fitted = True
    model._classes = np.array([0, 1])
    model._class_to_index = {0: 0, 1: 1}
    model._temperature_scaler = TemperatureScaler()
    model._temperature_scaler_state = model._temperature_scaler.state_dict()
    model._cached_logits = None
    model._cached_probabilities = None
    model.set_logits(X_cal, logits_cal)
    model.set_logits(X_test, logits_test)

    probs_before = model.predict_proba(X_test)
    brier_before = brier_score(probs_before, y_test)
    ece_before = expected_calibration_error(probs_before, y_test)

    model.calibrate(X_cal, y_cal)
    assert model._is_calibrated
    assert model._temperature_scaler_state is not None

    probs_after = model.predict_proba(X_test)
    brier_after = brier_score(probs_after, y_test)
    ece_after = expected_calibration_error(probs_after, y_test)

    assert brier_after < brier_before
    assert ece_after < ece_before
