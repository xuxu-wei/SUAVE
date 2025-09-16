# Repository Map

## Core Modules
- `suave/suave.py`: defines the `SUAVE` model class integrating a VAE with a multi-task predictor.
- `suave/sklearn.py`: provides `SuaveClassifier`, a scikit-learn compatible wrapper exposing `fit`, `predict`, and `score`.

## Training and Evaluation Entrypoints
- `SuaveClassifier.fit`: trains SUAVE models.
- `SuaveClassifier.score` / `predict_proba`: evaluation utilities used in benchmarks.
- `tools/benchmark.py`: full hard-task benchmark and regression guard maximizing SUAVE performance across missingness variants.
- `tests/test_benchmarks_smoke.py`: lightweight smoke-test benchmark verifying the wiring for the above APIs.

## Benchmark Dependencies
- `numpy`, `pandas`
- `scikit-learn`: `train_test_split`, `Pipeline`, `StandardScaler`, `SimpleImputer`,
  `LogisticRegression`, `SVC`, `KNeighborsClassifier`, `RandomForestClassifier`,
  and metrics (`roc_auc_score`, `average_precision_score`, `accuracy_score`, `f1_score`).
- `optuna` for hyperparameter search.
- Optional: `autogluon.tabular.TabularPredictor` when available.
