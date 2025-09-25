import numpy as np
import pandas as pd
import pytest

from suave import SUAVE, Schema


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
