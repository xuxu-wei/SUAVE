import numpy as np
import os
import sys
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from suave.suave import SUAVE
from suave.utils import set_random_seed


def generate_dataset(difficulty):
    rng = np.random.default_rng(0)
    if difficulty == "simple":
        n = 200
        base = rng.normal(size=(n, 5))
        noise = rng.normal(scale=0.1, size=(n, 5))
        x1, x2, x3, x4, x5 = base.T
        x6 = x1 + noise[:, 0]
        x7 = 0.5 * x2 - 0.3 * x3 + noise[:, 1]
        x8 = 0.2 * x4 + 0.8 * x5 + noise[:, 2]
        x9 = -x1 + x5 + noise[:, 3]
        x10 = x2 - x4 + 0.5 * x3 + noise[:, 4]
        X = np.column_stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]).astype(np.float32)
        logits = 2 * x1 + 1.5 * x2 - x3 + 0.5 * x4
        prob = 1 / (1 + np.exp(-logits))
        y1 = (prob > 0.5).astype(int)
        Y = y1[:, None].astype(np.int64)
        task_classes = [2]
    elif difficulty == "medium":
        n = 1000
        numeric = rng.normal(size=(n, 5))
        categorical = rng.integers(0, 5, size=n)
        cat_onehot = np.eye(5)[categorical]
        derived1 = numeric[:, 0] * numeric[:, 1]
        derived2 = np.sin(numeric[:, 2])
        derived3 = np.log1p(np.abs(numeric[:, 3]))
        derived4 = numeric[:, 4] * categorical
        derived5 = numeric[:, 1] + categorical
        noise = rng.normal(size=(n, 5))
        X = np.column_stack([
            numeric,
            cat_onehot,
            derived1,
            derived2,
            derived3,
            derived4,
            derived5,
            noise,
        ]).astype(np.float32)
        mask = rng.random(X.shape) < 0.1
        X[mask] = np.nan
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        logits = derived1 + derived2 + cat_onehot[:, 1] - cat_onehot[:, 2]
        prob = 1 / (1 + np.exp(-logits))
        y1 = (prob > 0.5).astype(int)
        y2_raw = numeric[:, 0] + derived3 + categorical
        y2 = pd.qcut(y2_raw, 3, labels=False)
        Y = np.column_stack([y1, y2]).astype(np.int64)
        task_classes = [2, 3]
    elif difficulty == "hard":
        n = 5000
        num = rng.normal(size=(n, 5))
        exp = rng.exponential(size=(n, 3))
        uniform = rng.uniform(-3, 3, size=(n, 2))
        heavy = rng.gamma(2.0, 2.0, size=n)
        cat = rng.integers(0, 5, size=n)
        cat_onehot = np.eye(5)[cat]
        bin_feat = rng.binomial(1, 0.3, size=n)
        count_feat = rng.poisson(3, size=n)
        derived1 = 0.5 * num[:, 0] + uniform[:, 0] + rng.normal(scale=0.1, size=n)
        derived2 = num[:, 1] * uniform[:, 1] + np.log1p(exp[:, 0])
        derived3 = np.sin(num[:, 2]) - exp[:, 1]
        derived4 = num[:, 3] ** 2 + count_feat
        derived5 = np.tanh(num[:, 4] + heavy)
        derived6 = derived1 * derived2
        derived7 = np.abs(num[:, 0] - uniform[:, 1])
        derived8 = exp[:, 2] / (1 + np.abs(num[:, 3]))
        derived9 = (cat == 3).astype(float) * num[:, 2]
        derived10 = bin_feat * heavy
        noise = rng.normal(size=(n, 15))
        dup = num[:, [0, 1]]
        dup_count = count_feat[:, None]
        const = np.ones((n, 4))
        X = np.column_stack([
            num,
            exp,
            uniform,
            heavy[:, None],
            cat_onehot,
            bin_feat[:, None],
            count_feat[:, None],
            derived1,
            derived2,
            derived3,
            derived4,
            derived5,
            derived6,
            derived7,
            derived8,
            derived9,
            derived10,
            noise,
            dup,
            dup_count,
            const,
        ]).astype(np.float32)
        assert X.shape[1] == 50
        mask = rng.random(X.shape) < 0.1
        X[mask] = np.nan
        X[bin_feat == 1, 0] = np.nan
        X[cat == 2, 1] = np.nan
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        logits1 = 0.7 * num[:, 0] - 1.1 * derived3 + 0.5 * bin_feat + cat_onehot[:, 2] - cat_onehot[:, 3]
        prob1 = 1 / (1 + np.exp(-logits1))
        thresh = np.quantile(prob1, 0.9)
        y1 = (prob1 > thresh).astype(int)
        raw2 = derived2 + np.cos(heavy) + cat + 0.1 * count_feat
        y2 = pd.qcut(raw2, 4, labels=False)
        Y = np.column_stack([y1, y2]).astype(np.int64)
        task_classes = [2, 4]
    else:
        raise ValueError("unknown difficulty")
    return X, Y, task_classes


def evaluate_linear(X_train, Y_train, X_test, Y_test, task_classes):
    aucs = []
    for k, classes in enumerate(task_classes):
        clf = LogisticRegression(max_iter=200)
        clf.fit(X_train, Y_train[:, k])
        proba = clf.predict_proba(X_test)
        if classes == 2:
            auc = roc_auc_score(Y_test[:, k], proba[:, 1])
        else:
            auc = roc_auc_score(Y_test[:, k], proba, multi_class="ovr")
        aucs.append(auc)
    return aucs


def evaluate_autogluon(X_train, Y_train, X_test, Y_test, task_classes):
    from autogluon.tabular import TabularPredictor

    columns = [f"f{i}" for i in range(X_train.shape[1])]
    aucs = []
    for k, classes in enumerate(task_classes):
        train_df = pd.DataFrame(X_train, columns=columns)
        train_df["label"] = Y_train[:, k]
        test_df = pd.DataFrame(X_test, columns=columns)
        test_df["label"] = Y_test[:, k]
        problem = "binary" if classes == 2 else "multiclass"
        predictor = TabularPredictor(
            label="label", problem_type=problem, eval_metric="roc_auc" if classes == 2 else "roc_auc_ovr"
        ).fit(train_df, time_limit=10, presets="medium_quality_faster_train", verbosity=0)
        proba = predictor.predict_proba(test_df)
        if classes == 2:
            auc = roc_auc_score(test_df["label"], proba[1])
        else:
            auc = roc_auc_score(test_df["label"], proba, multi_class="ovr")
        aucs.append(auc)
    return aucs


def evaluate_sklearn_baselines(X_train, Y_train, X_test, Y_test, task_classes):
    models = {
        "RandomForest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1),
        "SVM": lambda: SVC(probability=True, gamma="scale"),
    }
    results = {}
    for name, ctor in models.items():
        aucs = []
        for k, classes in enumerate(task_classes):
            clf = ctor()
            clf.fit(X_train, Y_train[:, k])
            proba = clf.predict_proba(X_test)
            if classes == 2:
                auc = roc_auc_score(Y_test[:, k], proba[:, 1])
            else:
                auc = roc_auc_score(Y_test[:, k], proba, multi_class="ovr")
            aucs.append(auc)
        results[name] = aucs
    return results


@pytest.mark.parametrize("difficulty", ["simple", "medium", "hard"])
def test_suave_on_synthetic_tasks(difficulty):
    set_random_seed(0)
    X, Y, task_classes = generate_dataset(difficulty)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    params = {
        "simple": dict(latent_dim=5, vae_depth=1, predictor_depth=1, batch_size=32),
        "medium": dict(latent_dim=8, vae_depth=1, predictor_depth=1, batch_size=64),
        "hard": dict(latent_dim=10, vae_depth=2, predictor_depth=2, batch_size=128),
    }[difficulty]
    model = SUAVE(
        input_dim=X_train.shape[1],
        task_classes=task_classes,
        validation_split=0.1,
        use_lr_scheduler=False,
        **params,
    )
    epochs = {"simple": 5, "medium": 10, "hard": 15}[difficulty]
    model.fit(X_train, Y_train, epochs=epochs, patience=5, verbose=False, early_stopping=False)
    total_loss, recon_loss, kl_loss, task_loss, aucs = model.eval_loss(X_test, Y_test)
    print(f"{difficulty} task test AUCs={aucs} recon={recon_loss:.4f}")
    lin_aucs = evaluate_linear(X_train, Y_train, X_test, Y_test, task_classes)
    try:
        auto_aucs = evaluate_autogluon(X_train, Y_train, X_test, Y_test, task_classes)
        baseline_names = ["AutoGluon"]
        baseline_aucs = [auto_aucs]
    except Exception as err:
        print(f"AutoGluon unavailable: {err}. Using sklearn baselines instead.")
        baseline_dict = evaluate_sklearn_baselines(X_train, Y_train, X_test, Y_test, task_classes)
        baseline_names = list(baseline_dict.keys())
        baseline_aucs = list(baseline_dict.values())
    rows = [["SUAVE", *aucs], ["Linear", *lin_aucs]]
    for name, model_aucs in zip(baseline_names, baseline_aucs):
        rows.append([name, *model_aucs])
    columns = ["Model"] + [f"task{i+1}_AUC" for i in range(len(task_classes))]
    table = pd.DataFrame(rows, columns=columns)
    print(table.to_string(index=False))
    assert np.isfinite(recon_loss)
