"""Core model definitions for the minimal SUAVE package."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import data as data_utils
from .modules.calibrate import TemperatureScaler
from .sampling import sample as sampling_stub
from .types import Schema

LOGGER = logging.getLogger(__name__)


class SUAVE:
    """Minimal SUAVE model stub.

    Parameters
    ----------
    schema:
        Optional :class:`Schema` describing the dataset. If not provided during
        initialisation it must be supplied to :meth:`fit`.
    latent_dim:
        Dimensionality of the latent representation. Default is ``32``.
    beta:
        Weighting factor for the KL term. Default is ``1.5``.
    hidden_dims:
        Shape of the encoder and decoder multilayer perceptrons. Default is
        ``(256, 128)``.
    dropout:
        Dropout probability applied inside neural modules. Default is ``0.1``.
    learning_rate:
        Optimiser learning rate. Default is ``1e-3``.
    val_split:
        Validation split ratio used inside :meth:`fit`. Default is ``0.2``.
    stratify:
        Whether to preserve class balance when creating the validation split.
    random_state:
        Seed controlling deterministic behaviour in helper utilities.

    Examples
    --------
    >>> import pandas as pd
    >>> from suave.model import SUAVE
    >>> from suave.types import Schema
    >>> X = pd.DataFrame({"age": [1.0, 2.0], "gender": [0, 1]})
    >>> y = pd.Series([0, 1])
    >>> schema = Schema({"age": {"type": "real"}, "gender": {"type": "cat", "n_classes": 2}})
    >>> model = SUAVE(schema=schema)
    >>> _ = model.fit(X, y)
    >>> proba = model.predict_proba(X)
    >>> proba.shape
    (2, 2)
    """

    def __init__(
        self,
        schema: Optional[Schema] = None,
        *,
        latent_dim: int = 32,
        beta: float = 1.5,
        hidden_dims: Iterable[int] = (256, 128),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        val_split: float = 0.2,
        stratify: bool = True,
        random_state: int = 0,
    ) -> None:
        self.schema = schema
        self.latent_dim = latent_dim
        self.beta = beta
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.stratify = stratify
        self.random_state = random_state

        self._is_fitted = False
        self._is_calibrated = False
        self._classes: np.ndarray | None = None
        self._normalization_stats: dict[str, dict[str, float | list[str]]] = {}
        self._temperature_scaler = TemperatureScaler()

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame | np.ndarray,
        *,
        schema: Optional[Schema] = None,
        epochs: int = 1,
    ) -> "SUAVE":
        """Train the minimal SUAVE model.

        Parameters
        ----------
        X:
            Training features with shape ``(n_samples, n_features)``.
        y:
            Training targets with shape ``(n_samples,)``.
        schema:
            Optional schema overriding the instance-level schema.
        epochs:
            Number of placeholder epochs to simulate. Each epoch logs a short
            message and sleeps for a few milliseconds.

        Returns
        -------
        SUAVE
            The fitted model (``self``) for fluent-style chaining.

        Examples
        --------
        >>> model = SUAVE(schema=schema)
        >>> model.fit(X, y, epochs=2)
        SUAVE(...)
        """

        if schema is not None:
            self.schema = schema
        if self.schema is None:
            raise ValueError("A schema must be provided to fit the model")

        LOGGER.info("Starting fit: n_samples=%s, n_features=%s", len(X), X.shape[1])
        X_train, X_val, y_train, y_val = data_utils.split_train_val(
            X, y, val_split=self.val_split, stratify=self.stratify
        )
        LOGGER.info("Train/val split: %s/%s", len(X_train), len(X_val))

        missing_mask = data_utils.build_missing_mask(X_train)
        LOGGER.debug(
            "Missing mask summary: %s missing values", int(missing_mask.sum().sum())
        )

        X_train, stats = data_utils.standardize(X_train, self.schema)
        self._normalization_stats = stats
        _ = data_utils.standardize(X_val, self.schema)

        y_train_array = np.asarray(y_train)
        self._classes = np.unique(y_train_array)
        if self._classes.size == 0:
            raise ValueError("Training targets must contain at least one class")

        for epoch in range(epochs):
            LOGGER.info("Epoch %s/%s - placeholder training step", epoch + 1, epochs)
            time.sleep(0.01)

        self._is_fitted = True
        LOGGER.info("Fit complete")
        return self

    # ------------------------------------------------------------------
    # Prediction utilities
    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return placeholder class probabilities.

        Parameters
        ----------
        X:
            Input features with shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples, n_classes)`` containing uniform
            probabilities across the observed classes.

        Examples
        --------
        >>> proba = model.predict_proba(X)
        >>> proba.sum(axis=1)
        array([1., 1.])
        """

        if not self._is_fitted or self._classes is None:
            raise RuntimeError("Model must be fitted before calling predict_proba")
        n_samples = len(X)
        n_classes = len(self._classes)
        probabilities = np.full((n_samples, n_classes), 1.0 / n_classes)
        if self._is_calibrated:
            logits = np.log(probabilities + 1e-8)
            logits = self._temperature_scaler(logits)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the most likely class for each sample."""

        probabilities = self.predict_proba(X)
        indices = probabilities.argmax(axis=1)
        return self._classes[indices]

    # ------------------------------------------------------------------
    # Calibration utilities
    # ------------------------------------------------------------------
    def calibrate(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "SUAVE":
        """Apply temperature scaling using placeholder logits."""

        if not self._is_fitted:
            raise RuntimeError("Fit must be called before calibrate")
        if len(X) != len(y):
            raise ValueError("X and y must have matching first dimensions")
        self._is_calibrated = True
        self._temperature_scaler = TemperatureScaler(temperature=1.0)
        return self

    # ------------------------------------------------------------------
    # Latent utilities and sampling
    # ------------------------------------------------------------------
    def encode(self, X: pd.DataFrame) -> np.ndarray:
        """Return zero latent representations matching ``latent_dim``."""

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before encoding data")
        return np.zeros((len(X), self.latent_dim))

    def sample(
        self, n_samples: int, conditional: bool = False, y: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Generate placeholder samples using :mod:`suave.sampling`."""

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before sampling")
        n_features = len(self.schema.feature_names) if self.schema else 0
        return sampling_stub(n_samples, n_features, conditional=conditional, y=y)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Serialise minimal model state to ``path``."""

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        path = Path(path)
        state = {
            "schema": self.schema.to_dict() if self.schema else None,
            "classes": self._classes.tolist() if self._classes is not None else None,
            "normalization": self._normalization_stats,
        }
        path.write_text(json.dumps(state))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SUAVE":
        """Load a model saved with :meth:`save`."""

        data = json.loads(Path(path).read_text())
        schema_dict = data.get("schema") or {}
        schema = Schema(schema_dict) if schema_dict else None
        model = cls(schema=schema)
        classes = data.get("classes")
        if classes is not None:
            model._classes = np.array(classes)
            model._is_fitted = True
        model._normalization_stats = data.get("normalization", {})
        return model

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            "SUAVE(latent_dim={latent_dim}, beta={beta}, hidden_dims={hidden_dims}, "
            "dropout={dropout}, learning_rate={learning_rate})"
        ).format(
            latent_dim=self.latent_dim,
            beta=self.beta,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
        )
