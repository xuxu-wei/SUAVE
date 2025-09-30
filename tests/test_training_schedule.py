import numpy as np
from collections.abc import Mapping

import pandas as pd
import pytest
import torch

import suave.model as suave_model

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


def test_decoder_refinement_freezes_encoder_and_logs(monkeypatch):
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

    original_adam = suave_model.Adam
    recorded_groups: list[dict[str, list[list[int]] | list[float]]] = []
    tracking_flag = [False]

    def recording_adam(params, *args, **kwargs):
        optimizer = original_adam(params, *args, **kwargs)
        if tracking_flag[0]:
            recorded_groups.append(
                {
                    "params": [
                        [id(param) for param in group["params"]]
                        for group in optimizer.param_groups
                    ],
                    "lrs": [group["lr"] for group in optimizer.param_groups],
                }
            )
        return optimizer

    monkeypatch.setattr(suave_model, "Adam", recording_adam)

    original_refine = SUAVE._refine_decoder_after_joint
    changes: dict[str, object] = {}

    def tracking_refine(self: SUAVE, *args, **kwargs) -> tuple[int, dict[str, torch.Tensor] | None]:
        decoder_ids = {id(param) for param in self._decoder.parameters()}
        prior_params = self._prior_parameters_for_optimizer()
        prior_ids = {id(param) for param in prior_params}
        encoder_ids = {id(param) for param in self._encoder.parameters()}
        classifier_ids = (
            {id(param) for param in self._classifier.parameters()}
            if self._classifier is not None
            else set()
        )
        changes["decoder_ids"] = decoder_ids
        changes["prior_ids"] = prior_ids
        changes["encoder_ids"] = encoder_ids
        changes["classifier_ids"] = classifier_ids
        recorded_groups.clear()
        tracking_flag[0] = True
        try:
            epochs_completed, stats = original_refine(self, *args, **kwargs)
        finally:
            tracking_flag[0] = False
        changes["epochs_completed"] = epochs_completed
        changes["stats"] = stats
        changes["optimizer_group"] = recorded_groups[-1] if recorded_groups else None
        return epochs_completed, stats

    monkeypatch.setattr(SUAVE, "_refine_decoder_after_joint", tracking_refine)

    model = SUAVE(schema=schema, latent_dim=3, n_components=2, batch_size=2)

    model.fit(
        X,
        y,
        warmup_epochs=1,
        head_epochs=1,
        finetune_epochs=1,
        decoder_refine_epochs=1,
        early_stop_patience=1,
        plot_monitor=True,
    )

    assert changes.get("epochs_completed") == 1
    assert changes.get("stats") is None
    recorded = changes.get("optimizer_group")
    assert recorded is not None

    decoder_ids = changes["decoder_ids"]
    prior_ids = changes["prior_ids"]
    encoder_ids = changes["encoder_ids"]
    classifier_ids = changes["classifier_ids"]

    assert recorded["params"], "Optimizer should include at least one parameter group"
    for lr in recorded["lrs"]:
        assert lr == pytest.approx(model.learning_rate)
    for group in recorded["params"]:
        for param_id in group:
            assert param_id in decoder_ids or param_id in prior_ids
            assert param_id not in encoder_ids
            assert param_id not in classifier_ids

    by_epoch = {epoch: (train, val) for epoch, train, val in updates}
    assert 3 in by_epoch
    train_metrics, val_metrics = by_epoch[3]
    assert "joint_objective" in train_metrics
    assert "total_loss" in train_metrics
    assert "total_loss" in val_metrics
    assert "kl" in val_metrics


@pytest.mark.parametrize(
    "mode, expect_optimizer, expect_prior_group, expect_collect_calls, expect_stats, expected_epochs",
    [
        ("decoder_only", True, False, 0, False, 1),
        ("decoder_prior", True, True, 0, False, 1),
        ("prior_em_only", False, False, 1, True, 1),
        ("prior_em_decoder", True, False, 1, True, 2),
    ],
)
def test_decoder_refine_modes(monkeypatch, mode, expect_optimizer, expect_prior_group, expect_collect_calls, expect_stats, expected_epochs):
    X, y, schema = _toy_dataset()

    recorded_groups: list[dict[str, list[list[int]] | list[float]]] = []
    tracking_flag = [False]
    collect_counter = [0]
    changes: dict[str, object] = {}

    original_adam = suave_model.Adam

    def recording_adam(params, *args, **kwargs):
        optimizer = original_adam(params, *args, **kwargs)
        if tracking_flag[0]:
            recorded_groups.append(
                {
                    "params": [
                        [id(param) for param in group["params"]]
                        for group in optimizer.param_groups
                    ],
                    "lrs": [group["lr"] for group in optimizer.param_groups],
                }
            )
        return optimizer

    monkeypatch.setattr(suave_model, "Adam", recording_adam)

    original_collect = SUAVE._collect_posterior_statistics

    def tracking_collect(self: SUAVE, *args, **kwargs):
        result = original_collect(self, *args, **kwargs)
        if tracking_flag[0]:
            collect_counter[0] += 1
        return result

    monkeypatch.setattr(SUAVE, "_collect_posterior_statistics", tracking_collect)

    original_refine = SUAVE._refine_decoder_after_joint

    def instrumented_refine(
        self: SUAVE, *args, **kwargs
    ) -> tuple[int, dict[str, torch.Tensor] | None]:
        decoder_ids = {id(param) for param in self._decoder.parameters()}
        prior_params = self._prior_parameters_for_optimizer()
        prior_ids = {id(param) for param in prior_params}
        collect_counter[0] = 0
        recorded_groups.clear()
        tracking_flag[0] = True
        prior_before = [param.detach().clone() for param in prior_params]
        try:
            epochs, stats = original_refine(self, *args, **kwargs)
        finally:
            tracking_flag[0] = False
        prior_after = [param.detach().clone() for param in prior_params]
        changes.update(
            {
                "decoder_ids": decoder_ids,
                "prior_ids": prior_ids,
                "epochs": epochs,
                "stats": stats,
                "optimizer_groups": list(recorded_groups),
                "collect_calls": collect_counter[0],
                "prior_deltas": [
                    float((after - before).abs().sum().item())
                    for before, after in zip(prior_before, prior_after)
                ],
            }
        )
        return epochs, stats

    monkeypatch.setattr(SUAVE, "_refine_decoder_after_joint", instrumented_refine)

    model = SUAVE(
        schema=schema,
        latent_dim=3,
        n_components=2,
        batch_size=2,
        decoder_refine_mode=mode,
    )

    model.fit(
        X,
        y,
        warmup_epochs=1,
        head_epochs=1,
        finetune_epochs=1,
        decoder_refine_epochs=1,
        early_stop_patience=1,
    )

    assert changes.get("epochs") == expected_epochs
    stats = changes.get("stats")
    if expect_stats:
        assert isinstance(stats, dict)
    else:
        assert stats is None

    optimizer_groups = changes.get("optimizer_groups") or []
    decoder_ids = changes.get("decoder_ids", set())
    prior_ids = changes.get("prior_ids", set())

    if expect_optimizer:
        assert optimizer_groups, "Expected optimizer to run during refinement"
        recorded = optimizer_groups[-1]
        groups = recorded["params"]
        if expect_prior_group:
            assert any(
                any(param_id in prior_ids for param_id in group) for group in groups
            ), "Prior parameters should be optimised in this mode"
        else:
            for group in groups:
                for param_id in group:
                    assert param_id in decoder_ids
                    assert param_id not in prior_ids
    else:
        assert not optimizer_groups, "Optimizer should not run in this mode"

    collect_calls = int(changes.get("collect_calls", 0))
    if expect_collect_calls:
        assert collect_calls >= expect_collect_calls
    else:
        assert collect_calls == 0

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
    def _assert_state_equal(actual: Mapping[str, object], expected: Mapping[str, object]) -> None:
        assert set(actual.keys()) == set(expected.keys())
        for key in actual:
            lhs = actual[key]
            rhs = expected[key]
            if isinstance(lhs, torch.Tensor):
                assert isinstance(rhs, torch.Tensor)
                assert torch.equal(lhs, rhs)
            elif isinstance(lhs, Mapping):
                assert isinstance(rhs, Mapping)
                _assert_state_equal(lhs, rhs)
            else:
                assert lhs == rhs

    _assert_state_equal(restored_states[-1], captured_states[0])
    assert model._joint_val_metrics is not None
    assert model._joint_val_metrics["joint_objective"] == pytest.approx(1.5)
    assert model.classification_loss_weight == pytest.approx(1.0)
