# SUAVE: Supervised, Unified, Augmented Variational Embedding

SUAVE is a schema-first variational autoencoder for mixed tabular data that unifies generative modelling and supervised prediction. The project draws direct inspiration from HI-VAE and related research on hierarchical latent variable models while modernising the workflow around explicit schemas, staged training, and probability calibration.

## Key features

- **Schema-driven inputs.** Users declare every column through `Schema`, giving the model explicit knowledge of data types and category counts before training begins.
- **Staged optimisation.** Training follows a warm-up → classifier head → joint fine-tuning schedule with KL annealing for stable convergence.
- **Transparent automation.** Heuristic defaults adapt batch sizes, learning rates, and schedule lengths using dataset statistics while keeping explicit overrides intact.
- **Mask-aware generative decoding.** Normalisation utilities and decoder heads propagate feature-wise masks so missing values remain consistent across real, categorical, positive, count, and ordinal variables.
- **Built-in calibration and evaluation.** Temperature scaling, Brier score, expected calibration error, and additional metrics are available for trustworthy downstream decisions.

## Core API surface

| Method | Purpose |
| ------ | ------- |
| `fit(X, y=None, **schedule)` | Train the generative model (and classifier head when labels are supplied) using staged optimisation with internal validation splits. |
| `predict(X)` | Return the most likely class label after optional calibration. |
| `predict_proba(X)` | Produce calibrated class probabilities with caching to avoid repeated encoder passes. |
| `calibrate(X, y)` | Learn temperature scaling parameters on held-out logits and reuse them for later predictions. |
| `encode(X, return_components=False)` | Map data to the latent space; optionally expose mixture assignments and component statistics. |
| `sample(n, conditional=False, y=None)` | Generate synthetic samples, optionally conditioned on class labels. |
| `impute(X, only_missing=True)` | Reconstruct missing or masked cells and merge them back into the original frame. |
| `save(path)` / `SUAVE.load(path)` | Persist and restore model weights, schema metadata, and calibration state for deployment. |

## Installation

```bash
pip install -r requirements.txt
```

The package targets Python 3.9+ with PyTorch as its primary dependency.

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

For an end-to-end demonstration, see [`examples/sepsis_minimal.py`](examples/sepsis_minimal.py).

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
model.fit(train_X, train_y, warmup_epochs=20, head_epochs=5, finetune_epochs=10)
```

When ``behaviour="unsupervised"`` the ``y`` argument is optional and the schedule collapses to the warm-up phase:

```python
unsupervised = SUAVE(schema=schema, behaviour="unsupervised")
unsupervised.fit(train_X, epochs=50)
```

### Probability prediction

```python
proba = model.predict_proba(test_X)
preds = model.predict(test_X)
```

Probabilities are cached per input fingerprint to avoid redundant encoder passes during repeated evaluations.

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

### Latent representations

```python
z = model.encode(test_X)
components = model.encode(test_X, return_components=True)
```

The second form exposes mixture assignments and component-specific statistics for downstream analysis.

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

## Roadmap

- Expand automatic schema tooling while preserving explicit overrides.
- Add counterfactual sampling helpers that leverage posterior component assignments.
- Integrate model interpretability reports into the training loop.

Community feedback and pull requests are welcome!
