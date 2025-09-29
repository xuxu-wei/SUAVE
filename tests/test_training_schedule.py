import numpy as np
import pandas as pd
import pytest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suave import SUAVE, Schema
from suave.modules.heads import ClassificationHead


def _toy_dataset() -> tuple[pd.DataFrame, pd.Series, Schema]:
    X = pd.DataFrame(
        {
            "age": [42.0, 55.0, 37.0, 60.0],
            "sofa": [3.0, 8.0, 4.0, 7.0],
            "gender": [0, 1, 0, 1],
        }
    )
    y = pd.Series([0, 1, 0, 1], name="outcome")
    schema = Schema(
        {
            "age": {"type": "real"},
            "sofa": {"type": "real"},
            "gender": {"type": "cat", "n_classes": 2},
        }
    )
    return X, y, schema


def test_training_schedule_runs_all_phases():
    X, y, schema = _toy_dataset()
    model = SUAVE(schema=schema, latent_dim=3, n_components=2, batch_size=2)

    model.fit(
        X,
        y,
        warmup_epochs=1,
        head_epochs=1,
        finetune_epochs=2,
        joint_decoder_lr_scale=0.5,
        early_stop_patience=0,
    )

    assert model.warmup_epochs == 1
    assert model.head_epochs == 1
    assert model.finetune_epochs == 2
    assert np.isclose(model.joint_decoder_lr_scale, 0.5)
    assert model.early_stop_patience == 0

    assert model._warmup_val_history, "Warm-up history should record validation stats"
    assert model.classification_loss_weight is not None
    assert model.classification_loss_weight > 0
    if model._joint_val_metrics is not None:
        assert "nll" in model._joint_val_metrics
        assert "joint_objective" in model._joint_val_metrics

    assert model._train_latent_mu is not None
    cached_rows = model._train_latent_mu.shape[0]
    assert cached_rows > 0
    assert model._train_latent_mu.shape[1] == model.latent_dim
    assert model._train_component_logits is not None
    assert model._train_component_logits.shape[0] == cached_rows
    assert model._train_component_probs is not None
    assert model._train_component_probs.shape[0] == cached_rows

    assert model._train_target_indices is not None
    assert len(model._train_target_indices) == cached_rows
    assert model._classifier is not None


def test_save_load_preserves_beta_and_gumbel_temperature(tmp_path):
    X, y, schema = _toy_dataset()
    model = SUAVE(
        schema=schema,
        latent_dim=3,
        n_components=2,
        batch_size=2,
        beta=2.75,
        gumbel_temperature=0.42,
    )

    model.fit(
        X,
        y,
        warmup_epochs=1,
        head_epochs=1,
        finetune_epochs=1,
        early_stop_patience=0,
    )

    path = tmp_path / "model.pt"
    model.save(path)

    restored = SUAVE.load(path)

    assert restored.beta == pytest.approx(model.beta)
    assert restored.gumbel_temperature == pytest.approx(model.gumbel_temperature)


def test_training_monitor_keeps_elbo_semantics(monkeypatch):
    X, y, schema = _toy_dataset()
    updates: list[tuple[int, dict[str, float | None], dict[str, float | None]]] = []

    class DummyMonitor:
        def __init__(self, behaviour: str) -> None:
            self.behaviour = behaviour

        def update(
            self,
            *,
            epoch: int,
            train_metrics: dict[str, float | None] | None = None,
            val_metrics: dict[str, float | None] | None = None,
            beta: float | None = None,
            classification_loss_weight: float | None = None,
        ) -> None:
            updates.append(
                (
                    int(epoch),
                    dict(train_metrics or {}),
                    dict(val_metrics or {}),
                )
            )

    monkeypatch.setattr("suave.plots.TrainingPlotMonitor", DummyMonitor)

    model = SUAVE(schema=schema, latent_dim=3, n_components=2, batch_size=2)
    model.fit(
        X,
        y,
        warmup_epochs=1,
        head_epochs=1,
        finetune_epochs=1,
        early_stop_patience=0,
        plot_monitor=True,
    )

    by_epoch = {epoch: (train, val) for epoch, train, val in updates}
    assert {0, 1, 2}.issubset(by_epoch.keys())

    warmup_train, warmup_val = by_epoch[0]
    assert warmup_train["total_loss"] is not None
    assert warmup_train.get("joint_objective") is None
    assert warmup_val.get("joint_objective") is None

    head_train, head_val = by_epoch[1]
    assert head_train.get("total_loss") is None
    assert head_train.get("joint_objective") is None
    assert head_train.get("classification_loss") is not None
    assert head_val.get("total_loss") is None
    assert head_val.get("joint_objective") is None

    joint_train, joint_val = by_epoch[2]
    assert joint_train.get("total_loss") is not None
    assert joint_train.get("joint_objective") is not None
    assert joint_train["joint_objective"] >= joint_train["total_loss"] - 1e-6
    assert np.isfinite(joint_val.get("joint_objective", np.nan))
    assert np.isfinite(joint_val.get("total_loss", np.nan))


def test_classification_weight_heuristic_clipping():
    assert SUAVE._derive_classification_loss_weight(10.0, 2.0) == pytest.approx(5.0)
    assert SUAVE._derive_classification_loss_weight(1.0, 1e-6) == pytest.approx(1000.0)
    assert SUAVE._derive_classification_loss_weight(1.0, 1e3) == pytest.approx(0.1)


def test_joint_classification_weight_warmup(monkeypatch):
    X, y, schema = _toy_dataset()
    recorded_weights: list[float | None] = []

    def tracking_scores(self, *args, classification_loss_weight=None, **kwargs):
        if classification_loss_weight is not None:
            recorded_weights.append(float(classification_loss_weight))
        return {
            "nll": 0.5,
            "classification_loss": 0.25,
            "joint_objective": float("nan"),
            "reconstruction": 0.0,
            "categorical_kl": 0.0,
            "gaussian_kl": 0.0,
            "brier": float("nan"),
            "ece": float("nan"),
            "auroc": float("nan"),
        }

    monkeypatch.setattr(SUAVE, "_compute_validation_scores", tracking_scores)

    model = SUAVE(
        schema=schema,
        latent_dim=3,
        n_components=2,
        batch_size=2,
        classification_loss_weight=2.0,
        class_weight_start_frac=0.25,
        class_weight_warmup_epochs=2,
        class_weight_schedule="linear",
    )

    model.fit(
        X,
        y,
        warmup_epochs=0,
        head_epochs=0,
        finetune_epochs=2,
        early_stop_patience=5,
    )

    expected = [0.5, 1.25, 2.0]
    assert len(recorded_weights) >= len(expected)
    assert recorded_weights[: len(expected)] == pytest.approx(expected)


def test_joint_finetune_patience_respects_warmup(monkeypatch):
    X, y, schema = _toy_dataset()
    recorded_weights: list[float | None] = []

    def tracking_scores(self, *args, classification_loss_weight=None, **kwargs):
        if classification_loss_weight is not None:
            recorded_weights.append(float(classification_loss_weight))
        return {
            "nll": 1.0,
            "classification_loss": 0.5,
            "joint_objective": 1.5,
            "reconstruction": 0.0,
            "categorical_kl": 0.0,
            "gaussian_kl": 0.0,
            "brier": float("nan"),
            "ece": float("nan"),
            "auroc": float("nan"),
        }

    monkeypatch.setattr(SUAVE, "_compute_validation_scores", tracking_scores)

    model = SUAVE(
        schema=schema,
        latent_dim=3,
        n_components=2,
        batch_size=2,
        classification_loss_weight=1.0,
        class_weight_warmup_epochs=2,
    )

    monkeypatch.setattr(
        model, "_is_better_metrics", lambda current, best: False, raising=False
    )

    model.fit(
        X,
        y,
        warmup_epochs=0,
        head_epochs=0,
        finetune_epochs=3,
        early_stop_patience=0,
    )

    assert len(recorded_weights) == 4
    assert recorded_weights[0] == pytest.approx(0.25)
    assert recorded_weights[1] < recorded_weights[2]
    assert recorded_weights[2] == pytest.approx(1.0)
    assert recorded_weights[3] == pytest.approx(1.0)


def test_head_phase_early_stop_restores_best(monkeypatch):
    X, y, schema = _toy_dataset()
    loss_sequence = [1.0, 0.9, 0.95, 0.8, 0.7]
    initial_length = len(loss_sequence)
    batches_processed = {"count": 0}
    captured: list[tuple[int, object]] = []
    restored: list[object] = []

    def fake_loss(self, logits, targets):
        batches_processed["count"] += 1
        value = loss_sequence.pop(0)
        return logits.sum() * 0 + logits.new_tensor(value)

    original_state_to_cpu = SUAVE._state_dict_to_cpu

    def tracking_state_to_cpu(module):
        state = original_state_to_cpu(module)
        if isinstance(module, ClassificationHead) and batches_processed["count"] > 0:
            epoch_index = batches_processed["count"] - 1
            captured.append((epoch_index, state))
        return state

    original_load = ClassificationHead.load_state_dict

    def tracking_load(self, state):
        restored.append(state)
        return original_load(self, state)

    monkeypatch.setattr(ClassificationHead, "loss", fake_loss)
    monkeypatch.setattr(SUAVE, "_state_dict_to_cpu", staticmethod(tracking_state_to_cpu))
    monkeypatch.setattr(ClassificationHead, "load_state_dict", tracking_load)

    model = SUAVE(
        schema=schema,
        latent_dim=3,
        n_components=2,
        batch_size=2,
        head_early_stop_patience=0,
    )
    model.fit(
        X,
        y,
        warmup_epochs=0,
        head_epochs=5,
        finetune_epochs=0,
        early_stop_patience=1,
    )

    # Expect early stopping before exhausting the queued loss values
    assert 0 < len(loss_sequence) < initial_length
    assert batches_processed["count"] < 5
    assert captured
    assert captured[0][0] == 0
    assert captured[1][0] == 1
    assert restored
    assert restored[-1] is captured[1][1]
    assert model.head_early_stop_patience == 0


def test_joint_finetune_early_stops_on_joint_objective(monkeypatch):
    X, y, schema = _toy_dataset()

    metrics_sequence = [
        {
            "nll": 1.0,
            "classification_loss": 0.5,
            "joint_objective": 1.5,
            "reconstruction": 0.0,
            "categorical_kl": 0.0,
            "gaussian_kl": 0.0,
            "brier": 0.2,
            "ece": 0.1,
            "auroc": 0.8,
        },
        {
            "nll": 0.9,
            "classification_loss": 0.8,
            "joint_objective": 1.7,
            "reconstruction": 0.0,
            "categorical_kl": 0.0,
            "gaussian_kl": 0.0,
            "brier": 0.25,
            "ece": 0.12,
            "auroc": 0.75,
        },
    ]

    def fake_compute_validation_scores(
        self,
        *args,
        classification_loss_weight: float | None = None,
        **kwargs,
    ) -> dict[str, float]:
        if classification_loss_weight is None:
            return {
                "nll": 2.0,
                "classification_loss": 0.4,
                "joint_objective": float("nan"),
                "reconstruction": 0.0,
                "categorical_kl": 0.0,
                "gaussian_kl": 0.0,
                "brier": float("nan"),
                "ece": float("nan"),
                "auroc": float("nan"),
            }
        if not metrics_sequence:
            return {}
        return metrics_sequence.pop(0).copy()

    captured_states: list[dict[str, object]] = []
    restored_states: list[dict[str, object]] = []

    original_capture = SUAVE._capture_model_state
    original_restore = SUAVE._restore_model_state

    def tracking_capture(self: SUAVE) -> dict[str, object]:
        state = original_capture(self)
        captured_states.append(state)
        return state

    def tracking_restore(self: SUAVE, state: dict[str, object], device) -> None:
        restored_states.append(state)
        original_restore(self, state, device)

    monkeypatch.setattr(
        SUAVE, "_compute_validation_scores", fake_compute_validation_scores
    )
    monkeypatch.setattr(SUAVE, "_capture_model_state", tracking_capture)
    monkeypatch.setattr(SUAVE, "_restore_model_state", tracking_restore)

    model = SUAVE(
        schema=schema,
        latent_dim=3,
        n_components=2,
        batch_size=2,
        classification_loss_weight=1.0,
    )
    model.fit(
        X,
        y,
        warmup_epochs=1,
        head_epochs=1,
        finetune_epochs=1,
        early_stop_patience=0,
    )

    assert not metrics_sequence
    assert captured_states
    assert restored_states
    assert restored_states[-1] == captured_states[0]
    assert model._joint_val_metrics is not None
    assert model._joint_val_metrics["joint_objective"] == pytest.approx(1.5)
    assert model.classification_loss_weight == pytest.approx(1.0)
