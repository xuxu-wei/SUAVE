[![中文](https://img.shields.io/badge/README-中文-blue)](https://github.com/xuxu-wei/SUAVE/blob/main/README-CN.md)![GitHub Repo stars](https://img.shields.io/github/stars/xuxu-wei/SUAVE)![Pepy Total Downloads](https://img.shields.io/pepy/dt/suave-ml)![PyPI - Version](https://img.shields.io/pypi/v/suave-ml)![PyPI - Status](https://img.shields.io/pypi/status/suave-ml)![PyPI - License](https://img.shields.io/pypi/l/suave-ml) 


# SUAVE: Supervised, Unified, Augmented Variational Embedding

SUAVE is a schema-first variational autoencoder for mixed tabular data that unifies generative modelling and supervised prediction. The project draws direct inspiration from HI-VAE and related research on hierarchical latent variable models while modernising the workflow around explicit schemas, staged training, and probability calibration.

## Key features

- **Schema-driven inputs.** Users declare every column through `Schema`, giving the model explicit knowledge of data types and category counts before training begins.
- **Staged optimisation.** Training follows a warm-up → classifier head → joint fine-tuning → decoder refinement schedule with KL annealing for stable convergence.
- **Transparent automation.** Heuristic defaults adapt batch sizes, learning rates, and schedule lengths using dataset statistics while keeping explicit overrides intact.
- **Mask-aware generative decoding.** Normalisation utilities and decoder heads propagate feature-wise masks so missing values remain consistent across real, categorical, positive, count, and ordinal variables.
- **Built-in calibration and evaluation.** Temperature scaling, Brier score, expected calibration error, and additional metrics are available for trustworthy downstream decisions.

## Installation

The package targets Python 3.9+ with PyTorch as its primary dependency. **It is recommended** to install the suitable PyTorch version for your system environment before installing this package. Please refer to the [official PyTorch guide](https://pytorch.org/get-started/locally/) for installation instructions. For example, on Windows, you can use the following pip command to install the version of PyTorch corresponding to CUDA 12.1:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install suave-ml
```
## Core API surface

| Method | Purpose |
| ------ | ------- |
| `fit(X, y=None, **schedule)` | Train the generative model (and classifier head when labels are supplied) using staged optimisation with internal validation splits. |
| `predict(X, attr=None, **options)` | Return class labels or attribute-level predictions; unsupervised models require `attr` to target a feature. |
| `predict_proba(X, attr=None, **options)` | Produce calibrated class probabilities or posterior predictive distributions for categorical/ordinal attributes with caching to avoid repeated encoder passes. |
| `predict_confidence_interval(X, attr, **options)` | Summarise posterior predictive distributions for real/positive/count attributes (point estimate + interval bounds, optional samples). |
| `calibrate(X, y)` | Learn temperature scaling parameters on held-out logits and reuse them for later predictions. |
| `encode(X, return_components=False)` | Map data to the latent space; optionally expose mixture assignments and component statistics. |
| `sample(n, conditional=False, y=None)` | Generate synthetic samples, optionally conditioned on class labels. |
| `impute(X, only_missing=True)` | Reconstruct missing or masked cells and merge them back into the original frame. |
| `save(path)` / `SUAVE.load(path)` | Persist and restore model weights, schema metadata, and calibration state for deployment. |



## Quick start

```python
import pandas as pd
from suave import SUAVE, SchemaInferencer

# 1. Load data and review the suggested schema interactively
train_X = pd.read_csv("data/train_features.csv")
train_y = pd.read_csv("data/train_labels.csv")["label"]
schema_result = SchemaInferencer().infer(train_X, mode="interactive")  # launches the UI
schema = schema_result.schema

# 2. Fit the model with the reviewed schema
model = SUAVE(schema=schema)
model.fit(train_X, train_y)

# 3. Generate predictions
probabilities = model.predict_proba(train_X.tail(5))
labels = model.predict(train_X.tail(5))
```

If you skip step 1, ``SUAVE.fit`` automatically infers a schema using
``mode="info"`` so you can still prototype quickly. The interactive review is
recommended for production datasets because it highlights columns that deserve a
manual check.

For an end-to-end demonstration, see  [`examples/demo-mimic_mortality_supervised.ipynb`](examples/demo-mimic_mortality_supervised.ipynb).

## API overview

The following snippets highlight the most common workflows. Each method accepts pandas DataFrames or NumPy arrays unless stated otherwise.

### Schema definition

```python
from suave.types import Schema

schema = Schema(
    {
        "age": {"type": "real"},
        "gender": {"type": "cat", "n_classes": 2},
        "lactate": {"type": "pos"},
        "icu_visits": {"type": "count"},
    }
)
```

Schemas can be updated with new columns and validated against incoming data:

```python
schema.update({"qsofa": {"type": "ordinal", "n_classes": 4}})
schema.require_columns(["age", "gender", "qsofa"])
```

Schema inference can also be automated and optionally reviewed via the browser
assistant:

```python
from suave import SchemaInferencer

result = SchemaInferencer().infer(train_X, mode="interactive")  # launches the UI
schema = result.schema
```

The ``interactive`` mode opens a lightweight GUI to confirm types and edit
flags. Use ``mode="info"`` to obtain diagnostics without the GUI or omit the
``schema`` entirely when constructing ``SUAVE`` to let ``fit`` infer it
automatically.

### Model fitting

```python
from suave import SUAVE

model = SUAVE(schema=schema, latent_dim=32, beta=1.5)
model.fit(train_X,train_y)
```

The final decoder refinement stage defaults to the warm-up length and can be
disabled by setting ``decoder_refine_epochs=0`` when a classifier-only focus is
desired.

When ``behaviour="unsupervised"`` the ``y`` argument is optional and the schedule collapses to the warm-up phase because the classifier head and decoder refinement stages are disabled:

```python
unsupervised = SUAVE(schema=schema, behaviour="unsupervised")
unsupervised.fit(train_X, epochs=50)
```

### Probability prediction

```python
from suave import data as suave_data

# Class-level predictions (supervised behaviour)
proba = model.predict_proba(test_X)
preds = model.predict(test_X)

# Attribute-level posterior queries
mask = suave_data.build_missing_mask(test_X)
gender_probs = model.predict_proba(test_X, attr="gender", mask=mask)
glucose_point = model.predict(test_X, attr="glucose")
glucose_samples = model.predict(test_X, attr="glucose", mode="sample", L=128)

# Continuous attributes with interval estimates
age_stats = model.predict_confidence_interval(test_X, "age", L=256)
```

Classifier probabilities are cached per input fingerprint to avoid redundant encoder passes during repeated evaluations. Providing `attr` switches to the generative decoder so you can recover posterior predictive distributions for individual features; pass `mask` when operating on imputed frames to preserve the original missingness pattern. Continuous attributes expose summary statistics via `predict_confidence_interval`, while `mode="sample"` on `predict` returns Monte Carlo draws. In unsupervised mode, specify `attr` explicitly because the classifier head is disabled.

**Supervised vs. unsupervised prediction behaviour**

- ``predict`` and ``predict_proba`` without ``attr`` require a fitted classifier head (the default supervised behaviour). Calling either method after training without labels raises an error because the logits cache cannot be populated.
- Supplying ``attr`` activates the generative decoder in both behaviours. ``predict_proba`` expects categorical or ordinal attributes, whereas ``predict`` falls back to ``predict_confidence_interval`` for real/positive/count features.
- `predict_confidence_interval` always operates on the decoder (thus requires ``attr``) and is limited to real/positive/count attributes. It returns posterior summaries in both modes and is the recommended entry point for continuous features when label heads are absent.
- In ``behaviour="unsupervised"`` the classifier head is disabled; therefore, ``predict`` and ``predict_proba`` must include ``attr`` and will return decoder-driven outputs exclusively.
- Passing ``mask`` for decoder-backed methods ensures masked cells stay hidden; omit it when raw ``NaN`` markers are present in ``X``.

### Calibration and evaluation

```python
model.calibrate(val_X, val_y)
calibrated = model.predict_proba(test_X)
```

Temperature scaling is trained on held-out logits and automatically reused for subsequent predictions.

```python
from suave.evaluate import compute_auroc, compute_auprc, compute_brier, compute_ece

auroc = compute_auroc(proba, val_y.to_numpy())
auprc = compute_auprc(proba, val_y.to_numpy())
brier = compute_brier(proba, val_y.to_numpy())
ece = compute_ece(proba, val_y.to_numpy(), n_bins=15)
```

Each helper validates probability shapes, performs necessary conversions for binary tasks, and returns `numpy.nan` when inputs are degenerate.

### Synthetic data quality

```python
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from suave.evaluate import (
    evaluate_trtr,
    evaluate_tstr,
    classifier_two_sample_test,
    mutual_information_feature,
    rbf_mmd,
    simple_membership_inference,
)

# Compare real-vs-real and synthetic-vs-real transfer
tstr_scores = evaluate_tstr((X_syn, y_syn), (X_test, y_test), LogisticRegression)
trtr_scores = evaluate_trtr((X_train, y_train), (X_test, y_test), LogisticRegression)

# Run the classifier two-sample test (C2ST) on full feature matrices
real_matrix = real_features.values
synthetic_matrix = synthetic_features.values
c2st = classifier_two_sample_test(
    real_matrix,
    synthetic_matrix,
    model_factories={
        "xgboost": lambda: XGBClassifier(random_state=0),
        "logistic": lambda: LogisticRegression(max_iter=200),
    },
    random_state=0,
    n_bootstrap=200,
)

# Inspect per-feature distribution alignment
mmd_labs, mmd_labs_p = rbf_mmd(
    real_labs.values, synthetic_labs.values, random_state=0, n_permutations=200
)
mi_unit = mutual_information_feature(real_unit.values, synthetic_unit.values)

# Audit membership privacy leakage
attack = simple_membership_inference(train_probs, train_labels, test_probs, test_labels)
```

The `evaluate_tstr`/`evaluate_trtr` pair supports model-agnostic baselines for benchmarking synthetic cohorts. `classifier_two_sample_test` accepts a mapping of estimator factories—by default we pair an XGBoost endpoint with a logistic regression sensitivity check—while the RBF-MMD, energy distance (dimension-normalised Euclidean + Hamming with optional permutation `p`-values), and mutual information helpers quantify per-feature fidelity. Low C2ST AUCs (≈`0.5`), low MMD/energy distance (≈`0.0`), and near-zero mutual information indicate strong alignment; larger values call for manual inspection. The membership attack reports AUROC and accuracy for separating training members from held-out data, highlighting potential privacy leakage.

### Latent representations

```python
z = model.encode(test_X)
components = model.encode(test_X, return_components=True)
```

The second form exposes mixture assignments and component-specific statistics for downstream analysis.

### Latent-feature correlations

```python
from suave.plots import plot_feature_latent_correlation_bubble

fig, ax = plot_feature_latent_correlation_bubble(model, train_X, targets=train_y)
```

The helper draws a bubble chart sized by the absolute Spearman
correlation and coloured by the (adjusted) `p`-value, saving the figure
when ``output_path`` is provided (for example,
``outputs/latent_correlations.png``).

### Sampling

```python
synthetic = model.sample(100)
conditional = model.sample(50, conditional=True, y=preds[:50])
```

Generated frames are automatically denormalised back into the original feature space, including categorical decoding.

### Imputation

```python
# Fill only the entries that SUAVE marked as missing during normalisation
completed = model.impute(test_X, only_missing=True)

# The same API works in unsupervised mode when no labels are provided
unsup_completed = unsupervised.impute(test_X, only_missing=True)
```

`impute` runs the decoder on masked cells (including unseen categorical levels and out-of-range ordinals) and merges the
reconstructed values back into the input frame so downstream consumers receive fully populated features.

### Persistence

```python
path = model.save("artifacts/sepsis.suave")
restored = SUAVE.load(path)
restored.predict_proba(test_X)
```

Model artefacts include schema metadata, learned parameters, and calibration state for reproducible deployment.

**Community feedback and pull requests are welcome!**