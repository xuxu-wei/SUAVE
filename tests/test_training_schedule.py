import numpy as np
import pandas as pd

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
    assert model._joint_val_metrics is None or "nll" in model._joint_val_metrics

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
