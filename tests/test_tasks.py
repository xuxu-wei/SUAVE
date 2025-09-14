import numpy as np
import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from suave.suave import SUAVE
from suave.utils import set_random_seed


def generate_dataset(difficulty):
    rng = np.random.default_rng(0)
    if difficulty == "simple":
        n = 60
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        logits = x1 + 0.5 * x2
        prob = 1 / (1 + np.exp(-logits))
        y1 = (prob > 0.5).astype(int)
        X = np.column_stack([x1, x2]).astype(np.float32)
        Y = y1[:, None].astype(np.int64)
        task_classes = [2]
    elif difficulty == "medium":
        n = 100
        numeric = rng.normal(size=(n, 3))
        categorical = rng.integers(0, 4, size=n)
        cat_onehot = np.eye(4)[categorical]
        X = np.concatenate([numeric, cat_onehot], axis=1).astype(np.float32)
        logits = numeric[:, 0] ** 2 + np.sin(numeric[:, 1]) + cat_onehot[:, 1] - cat_onehot[:, 2]
        prob = 1 / (1 + np.exp(-logits))
        y1 = (prob > 0.5).astype(int)
        y2_raw = numeric[:, 0] + numeric[:, 1] * numeric[:, 2] + categorical
        y2 = pd.qcut(y2_raw, 3, labels=False)
        Y = np.column_stack([y1, y2]).astype(np.int64)
        task_classes = [2, 3]
    elif difficulty == "hard":
        n = 200
        num1 = rng.normal(size=n)
        num2 = rng.exponential(size=n)
        num3 = rng.uniform(-2, 2, size=n)
        cat = rng.integers(0, 3, size=n)
        cat_onehot = np.eye(3)[cat]
        bin_feat = rng.binomial(1, 0.3, size=n)
        nonlinear = np.sin(num1) * num3 + np.log1p(num2)
        X = np.column_stack([num1, num2, num3, cat_onehot, bin_feat, nonlinear]).astype(np.float32)
        # random missingness
        mask = rng.random(X.shape) < 0.1
        X[mask] = np.nan
        # non-random missingness conditioned on bin_feat
        X[bin_feat == 1, 0] = np.nan
        # mean imputation
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        logits1 = 0.8 * num1 - 1.2 * num3 + 0.5 * bin_feat + cat_onehot[:, 1] - cat_onehot[:, 2]
        prob1 = 1 / (1 + np.exp(-logits1))
        thresh = np.quantile(prob1, 0.7)
        y1 = (prob1 > thresh).astype(int)
        raw2 = num1 * num3 + np.cos(num2) + cat
        y2 = pd.qcut(raw2, 3, labels=False)
        Y = np.column_stack([y1, y2]).astype(np.int64)
        task_classes = [2, 3]
    else:
        raise ValueError("unknown difficulty")
    return X, Y, task_classes


@pytest.mark.parametrize("difficulty", ["simple", "medium", "hard"])
def test_suave_on_synthetic_tasks(difficulty):
    set_random_seed(0)
    X, Y, task_classes = generate_dataset(difficulty)
    model = SUAVE(
        input_dim=X.shape[1],
        task_classes=task_classes,
        latent_dim=5,
        vae_depth=1,
        predictor_depth=1,
        batch_size=16,
        validation_split=0.2,
        use_lr_scheduler=False,
    )
    epochs = {"simple": 5, "medium": 10, "hard": 20}[difficulty]
    model.fit(X, Y, epochs=epochs, patience=5, verbose=False, early_stopping=False)
    total_loss, recon_loss, kl_loss, task_loss, aucs = model.eval_loss(X, Y)
    print(f"{difficulty} task AUCs={aucs} recon={recon_loss:.4f}")
    assert np.isfinite(recon_loss)
