[![Static Badge](https://img.shields.io/badge/%E5%88%87%E6%8D%A2-%E4%B8%AD%E6%96%87%E7%89%88%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3-1082C3?style=flat)](使用说明-中文版.md)

# SUAVE

SUAVE is an integrated approach to clinical tabular modelling built around a
single low-dimensional latent variable ``z``. A multi-distribution decoder
accommodates mixed feature types, reconstruction is aware of missing values and
sample generation can be class conditioned (enabling an optional CVAE variant).
The model includes a deployment-friendly calibration link and explicit hooks for
interpretability so that the same ``z`` supports reconstruction, synthesis and
downstream tasks.

``z`` acts as the information hub for three workflows:

1. **Prediction** – a classifier consumes ``z`` (and encoder features) to output
   calibrated probabilities.
2. **Generation** – decoding samples from ``z`` produces missing-free synthetic
   tables for TSTR evaluation.
3. **Interpretation** – correlations between ``z`` and clinical covariates help
   reveal latent factors.

This probability-first design natively handles heterogeneous variables,
missingness and class imbalance, providing a compact and interpretable approach
that is particularly suited to clinical research.

## Installation

```bash
pip install suave-ml
```

## Quick start

```python
import numpy as np
from suave import SUAVE

X_train = np.random.randn(100, 10)
y_train = np.random.randint(0, 2, size=100)
X_test = np.random.randn(20, 10)

model = SUAVE(input_dim=X_train.shape[1])
model.fit(X_train, y_train, epochs=5)
proba = model.predict_proba(X_test)
synthetic = model.generate(5)
z = model.latent(X_test)
```

## API sketch

* ``fit(X, y, epochs=20)`` – train the model.
* ``predict_proba(X)`` / ``predict(X)`` – obtain calibrated predictions.
* ``generate(n, conditional=None, seed=None)`` – create synthetic samples.
* ``latent(X)`` – extract latent representations.

## License

BSD-3-Clause
