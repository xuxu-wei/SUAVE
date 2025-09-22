# SUAVE

SUAVE is a schema-first variational autoencoder for mixed tabular data that unifies generative modeling and supervised prediction. The project draws direct inspiration from HI-VAE and related research on hierarchical latent variable models while modernising the workflow around explicit schemas, staged training, and probability calibration.

## Design philosophy

- **Schema-driven inputs.** Users must declare every feature through :class:`~suave.types.Schema`, ensuring the model has explicit knowledge of mixed data types and category counts before training.【F:suave/types.py†L22-L115】
- **Staged optimisation.** Training follows a warm-up → classifier head → joint fine-tuning schedule with KL annealing, mirroring best practices from HI-VAE-style objectives for stable convergence.【F:suave/model.py†L889-L1010】
- **Transparent automation.** Optional ``auto_parameters`` heuristics adapt batch sizes and schedule lengths using dataset statistics while keeping overrides explicit in method signatures.【F:suave/model.py†L73-L128】【F:suave/model.py†L1039-L1091】
- **Mask-aware generative decoding.** Normalisation utilities and decoder heads propagate feature-wise masks so missing data is handled consistently across real, categorical, positive, count, and ordinal variables.【F:suave/data.py†L15-L143】【F:suave/model.py†L1123-L1244】
- **Built-in calibration and evaluation.** Temperature scaling, Brier score, expected calibration error, and other metrics are exposed for reliable downstream decision making.【F:suave/model.py†L2603-L2682】【F:suave/evaluate.py†L1-L200】

## Installation

```bash
pip install -r requirements.txt
```

The package targets Python 3.9+ with PyTorch as its primary dependency.

## Quick start

```python
import pandas as pd
from suave import SUAVE, Schema

# 1. Declare the schema
schema = Schema(
    {
        "age": {"type": "real"},
        "sofa": {"type": "count"},
        "gender": {"type": "cat", "n_classes": 2},
    }
)

# 2. Load data and fit the model
train_X = pd.read_csv("data/train_features.csv")
train_y = pd.read_csv("data/train_labels.csv")["label"]
model = SUAVE(schema=schema)
model.fit(train_X, train_y)

# 3. Generate predictions
probabilities = model.predict_proba(train_X.tail(5))
labels = model.predict(train_X.tail(5))
```

For an end-to-end demonstration, see [`examples/sepsis_minimal.py`](examples/sepsis_minimal.py).

## API overview

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

### Calibration

```python
model.calibrate(val_X, val_y)
calibrated = model.predict_proba(test_X)
```

Temperature scaling is trained on held-out logits and automatically reused for subsequent predictions.

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

### Evaluation utilities

```python
from suave.evaluate import compute_auroc, compute_auprc, compute_brier, compute_ece

auroc = compute_auroc(proba, val_y.to_numpy())
auprc = compute_auprc(proba, val_y.to_numpy())
brier = compute_brier(proba, val_y.to_numpy())
ece = compute_ece(proba, val_y.to_numpy(), n_bins=15)
```

Each helper validates probability shapes, performs necessary conversions for binary tasks, and returns ``numpy.nan`` when inputs are degenerate.

## Roadmap

- Expand automatic schema tooling while preserving explicit overrides.
- Add counterfactual sampling helpers that leverage posterior component assignments.
- Integrate model interpretability reports into the training loop.

Community feedback and pull requests are welcome!
