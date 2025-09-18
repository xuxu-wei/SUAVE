"""Minimal reproduction of the TensorFlow HI-VAE general example using SUAVE.

This script mirrors the structure of ``third_party/hivae_tf/hivae/examples/
hivae_general_example.py`` but relies exclusively on the native PyTorch
implementation.  It creates a small mixed-type dataset, trains SUAVE in
``behaviour="hivae"`` mode and reports reconstruction quality.  When the
TensorFlow reference implementation is available the script also prints the
baseline scores for a quick side-by-side comparison.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from suave.model import SUAVE
from suave.types import Schema


def _make_mock_dataset(n_samples: int, *, random_state: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    data = {
        "age": rng.normal(loc=60.0, scale=12.0, size=n_samples),
        "gender": rng.integers(0, 2, size=n_samples),
        "cholesterol": rng.lognormal(mean=math.log(180.0), sigma=0.15, size=n_samples),
        "num_admissions": rng.poisson(lam=1.8, size=n_samples),
    }
    frame = pd.DataFrame(data)
    frame["age"] = frame["age"].clip(lower=18.0)
    frame["cholesterol"] = frame["cholesterol"].clip(lower=90.0)
    frame.loc[rng.random(size=n_samples) < 0.15, "cholesterol"] = np.nan
    return frame


def _schema() -> Schema:
    return Schema(
        {
            "age": {"type": "real"},
            "gender": {"type": "cat", "n_classes": 2},
            "cholesterol": {"type": "pos"},
            "num_admissions": {"type": "count"},
        }
    )


def _fit_suave(train: pd.DataFrame, *, epochs: int) -> SUAVE:
    model = SUAVE(
        schema=_schema(), behaviour="hivae", latent_dim=8, hidden_dims=(64, 32)
    )
    model.fit(train, epochs=epochs, batch_size=min(64, len(train)))
    return model


def _reconstruction_error(original: pd.DataFrame, reconstructed: pd.DataFrame) -> float:
    numeric_cols = [
        col for col, spec in _schema().to_dict().items() if spec["type"] != "cat"
    ]
    diff = original[numeric_cols] - reconstructed[numeric_cols]
    return float(np.nanmean(diff.to_numpy(dtype=np.float32) ** 2))


def _reconstruct(model: SUAVE, X: pd.DataFrame) -> pd.DataFrame:
    return model.impute(X, only_missing=False)


def _maybe_run_tensorflow_baseline(
    train: pd.DataFrame, *, epochs: int
) -> dict[str, Any] | None:
    try:
        from third_party.hivae_tf.hivae import hivae as tf_hivae
    except ImportError:
        return None

    types_description = [
        ("age", "pos", 1, None),
        ("gender", "cat", 2, 2),
        ("cholesterol", "pos", 1, None),
        ("num_admissions", "count", 1, None),
    ]
    network_dict = {
        "batch_size": min(64, len(train)),
        "model_name": "model_HIVAE_inputDropout",
        "dim_z": 6,
        "dim_y": 6,
        "dim_s": 6,
    }

    temp_dir = Path("./.hivae_tf_cache")
    temp_dir.mkdir(exist_ok=True)
    baseline = tf_hivae.hivae(
        types_description,
        network_dict,
        results_path=temp_dir / "results",
        network_path=temp_dir / "networks",
        verbosity_level=0,
    )
    mask = train.notna().astype(float)
    baseline.fit(train.fillna(train.mean()), epochs=epochs, true_missing_mask=mask)
    recon = baseline.predict(train.fillna(train.mean()), true_missing_mask=mask)[1]
    recon_df = pd.DataFrame(recon, columns=train.columns, index=train.index)
    mse = _reconstruction_error(train, recon_df)
    return {"mse": mse, "path": temp_dir}


def main(epochs: int) -> None:
    train = _make_mock_dataset(300)
    test = _make_mock_dataset(120, random_state=1)

    model = _fit_suave(train, epochs=epochs)
    samples = model.sample(5)
    reconstruction = _reconstruct(model, test)
    suave_mse = _reconstruction_error(test, reconstruction)

    print("SUAVE HI-VAE example")
    print("====================")
    print(f"Training samples : {len(train)}")
    print(f"Test samples     : {len(test)}")
    print(f"Reconstruction MSE (numeric cols): {suave_mse:.4f}")
    print("Synthetic preview:\n", samples.head())

    baseline = _maybe_run_tensorflow_baseline(train, epochs=epochs)
    if baseline is None:
        print(
            "\n[optional] Install third_party HI-VAE TensorFlow code to compare baselines."
        )
    else:
        print("\nTensorFlow baseline")
        print(f"Reconstruction MSE (numeric cols): {baseline['mse']:.4f}")
        print(f"Artifacts stored under {baseline['path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SUAVE HI-VAE example.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs for both models.",
    )
    args = parser.parse_args()
    main(args.epochs)
