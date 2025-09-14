import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from suave.sklearn import SuaveClassifier

try:
    from autogluon.tabular import TabularPredictor
    HAS_AG = True
except Exception:  # pragma: no cover
    HAS_AG = False

rng = np.random.default_rng(0)


def _insert_missing(X, mcar=0.0, mnar_indices=None, y=None):
    """Insert random (MCAR) and non-random missingness."""
    X = X.copy()
    if mcar > 0:
        mask = rng.random(X.shape) < mcar
        X[mask] = np.nan
    if mnar_indices is not None and y is not None:
        for idx in mnar_indices:
            mask = (y == 1) & (rng.random(len(y)) < 0.5)
            X[mask, idx] = np.nan
    return X


def generate_simple():
    n = 200
    X = rng.normal(size=(n, 10))
    y1 = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + rng.normal(scale=0.1, size=n) > 0).astype(int)
    w = X[:, 3] + 0.5 * X[:, 4] - X[:, 5]
    bins = np.quantile(w, [0, 1/3, 2/3, 1])
    y2 = np.digitize(w, bins[1:-1])
    Y = np.column_stack([y1, y2])
    X = _insert_missing(X, mcar=0.05)
    return X, Y, [2, 3]


def generate_medium():
    n = 1000
    part1 = rng.normal(size=(n, 10))
    part2 = rng.uniform(-3, 3, size=(n, 5))
    part3 = rng.exponential(scale=1.0, size=(n, 5))
    X = np.hstack([part1, part2, part3])
    x1, x2, x3, x4, x5, x6, x7, x8 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7]
    derived = np.column_stack([
        x1 * x2,
        np.sin(x3),
        x4 ** 2,
        (x5 > 0).astype(float),
        x6 + rng.normal(scale=0.1, size=n),
    ])
    noise = rng.normal(size=(n, 5))
    X = np.hstack([X, derived, noise])
    y1_score = x1 * x2 + np.sin(x3) + np.log(np.abs(x4) + 1) + rng.normal(scale=0.1, size=n)
    thresh = np.quantile(y1_score, 0.7)
    y1 = (y1_score > thresh).astype(int)
    y2_score = x5 ** 2 - x6 * x7 + np.sin(x8)
    bins = np.quantile(y2_score, [0, 1/3, 2/3, 1])
    y2 = np.digitize(y2_score, bins[1:-1])
    Y = np.column_stack([y1, y2])
    X = _insert_missing(X, mcar=0.1, mnar_indices=[1], y=y1)
    return X, Y, [2, 3]


def generate_hard():
    n = 3000
    cont_norm = rng.normal(size=(n, 10))
    cont_log = rng.lognormal(mean=0.0, sigma=1.0, size=(n, 5)) * 10
    cont_uni = rng.uniform(-5, 5, size=(n, 5))
    cont_exp = rng.exponential(scale=1.0, size=(n, 5))
    cont = np.hstack([cont_norm, cont_log, cont_uni, cont_exp])  # 25
    pois = rng.poisson(lam=3, size=(n, 5))
    bino = rng.binomial(n=10, p=0.3, size=(n, 5))
    cat = rng.integers(0, 4, size=(n, 5))
    base = np.hstack([cont, pois, bino, cat])  # 40
    d1 = base[:, 0] * base[:, 1]
    d2 = np.sin(base[:, 2])
    d3 = np.log(np.abs(base[:, 3]) + 1)
    d4 = (base[:, 4] > base[:, 5]).astype(float)
    d5 = base[:, 6] ** 2
    derived = np.column_stack([d1, d2, d3, d4, d5])
    noise = rng.normal(size=(n, 5))
    X = np.hstack([base, derived, noise])
    s1 = base[:, 0] * base[:, 1] + np.sin(base[:, 2]) - np.log(np.abs(base[:, 3]) + 1) + base[:, 4] ** 2
    t1 = np.quantile(s1, 0.9)
    y1 = (s1 > t1).astype(int)
    s2 = base[:, 5] * base[:, 6] - np.sin(base[:, 7]) + np.log1p(base[:, 8] ** 2) - base[:, 9]
    bins = np.quantile(s2, [0, 0.25, 0.5, 0.75, 1])
    y2 = np.digitize(s2, bins[1:-1])
    y3 = ((cont_norm[:, 0] > 0) ^ (cont_uni[:, 0] > 0)).astype(int)
    Y = np.column_stack([y1, y2, y3])
    X = _insert_missing(X, mcar=0.1, mnar_indices=[0, 1], y=y1)
    return X, Y, [2, 4, 2]


def _model_factory(name):
    if name == "Linear":
        clf = LogisticRegression(max_iter=1000)
    elif name == "SVM":
        clf = SVC(kernel="rbf", max_iter=200)
    elif name == "KNN":
        clf = KNeighborsClassifier()
    elif name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError(name)
    return Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler(with_mean=False) if name in ["SVM", "KNN", "Linear"] else "passthrough"), ("clf", clf)])


def benchmark_models(X_train, X_test, y_train, y_test, task_classes):
    model_names = ["Linear", "SVM", "KNN", "RandomForest"]
    if HAS_AG:
        model_names.append("AutoGluon")
    results = {m: [] for m in model_names}
    for t, n_class in enumerate(task_classes):
        ytr, yte = y_train[:, t], y_test[:, t]
        for m in model_names:
            if m == "AutoGluon":
                train_df = pd.DataFrame(X_train)
                train_df["label"] = ytr
                test_df = pd.DataFrame(X_test)
                predictor = TabularPredictor(label="label", verbosity=0).fit(train_df, time_limit=10)
                proba = predictor.predict_proba(test_df).to_numpy()
            else:
                model = _model_factory(m)
                Xtr, ysub = X_train, ytr
                if m == "SVM" and len(Xtr) > 1000:
                    idx = rng.choice(len(Xtr), 1000, replace=False)
                    Xtr, ysub = Xtr[idx], ytr[idx]
                model.fit(Xtr, ysub)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)
                else:
                    scores = model.decision_function(X_test)
                    if scores.ndim == 1:
                        proba = 1/ (1 + np.exp(-scores))
                    else:
                        exp_s = np.exp(scores)
                        proba = exp_s / exp_s.sum(axis=1, keepdims=True)
            if proba.ndim == 1:
                auc = roc_auc_score(yte, proba)
            elif proba.shape[1] == 2:
                auc = roc_auc_score(yte, proba[:, 1])
            else:
                auc = roc_auc_score(yte, proba, multi_class="ovr", average="macro")
            results[m].append(auc)
    return {m: float(np.mean(v)) for m, v in results.items() if v}


def run_task(generator, name):
    X, Y, task_classes = generator()
    for _ in range(10):
        rs = rng.integers(0, 1_000_000)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=int(rs))
        if all(len(np.unique(y_train[:, i])) > 1 for i in range(Y.shape[1])):
            break
    else:
        raise ValueError("Failed to generate split with all classes present")
    bench = benchmark_models(X_train, X_test, y_train, y_test, task_classes)
    imputer = SimpleImputer()
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(X_train))
    Xte = scaler.transform(imputer.transform(X_test))
    model = SuaveClassifier(input_dim=Xtr.shape[1], task_classes=task_classes, latent_dim=16, vae_depth=2, predictor_depth=2, batch_size=256, gamma_task=1.0)
    model.fit(Xtr, y_train, epochs=100, patience=5, verbose=False)
    suave_auc = model.score(Xte, y_test).mean()
    _, recon_loss, _, _, _ = model.eval_loss(Xte, y_test)
    bench["SUAVE"] = suave_auc
    return bench, recon_loss


def test_benchmarks():
    tasks = [(generate_simple, "simple"), (generate_medium, "medium"), (generate_hard, "hard")]
    rows = []
    recon_rows = []
    for gen, name in tasks:
        res, recon = run_task(gen, name)
        for model, auc in res.items():
            rows.append({"dataset": name, "model": model, "auc": auc})
        recon_rows.append({"dataset": name, "recon": recon})
    df = pd.DataFrame(rows)
    table = df.pivot(index="model", columns="dataset", values="auc")
    print("\nBenchmark AUCs:\n", table.to_markdown())
    recon_df = pd.DataFrame(recon_rows)
    print("\nSUAVE Reconstruction Loss:\n", recon_df.to_markdown(index=False))

