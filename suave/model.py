"""Core model definitions for the minimal SUAVE package."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import CategoricalDtype
import torch
from torch import Tensor
from torch.optim import Adam
from tqdm.auto import tqdm

from . import data as data_utils
from .modules.calibrate import TemperatureScaler
from .modules.decoder import Decoder
from .modules.encoder import EncoderMLP
from .modules.heads import ClassificationHead
from .modules import losses
from .sampling import sample as sampling_stub
from .types import Schema

LOGGER = logging.getLogger(__name__)


class SUAVE:
    """HI-VAE inspired model for mixed tabular data.

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
    batch_size:
        Mini-batch size used inside :meth:`fit`. Default is ``128``.
    kl_warmup_epochs:
        Number of epochs over which to linearly anneal the KL term.
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
        batch_size: int = 128,
        kl_warmup_epochs: int = 10,
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
        self.batch_size = batch_size
        self.kl_warmup_epochs = kl_warmup_epochs
        self.val_split = val_split
        self.stratify = stratify
        self.random_state = random_state

        self._is_fitted = False
        self._is_calibrated = False
        self._classes: np.ndarray | None = None
        self._norm_stats_per_col: dict[str, dict[str, float | list[str]]] = {}
        self._temperature_scaler = TemperatureScaler()
        self._temperature_scaler_state: dict[str, float | bool] | None = None
        self._device: torch.device | None = None
        self._encoder: EncoderMLP | None = None
        self._decoder: Decoder | None = None
        self._classifier: ClassificationHead | None = None
        self._feature_layout: dict[str, dict[str, int]] = {"real": {}, "cat": {}}
        self._class_to_index: dict[object, int] | None = None
        self._cached_logits: np.ndarray | None = None
        self._cached_probabilities: np.ndarray | None = None

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
        batch_size: Optional[int] = None,
        kl_warmup_epochs: Optional[int] = None,
    ) -> "SUAVE":
        """Optimise the HI-VAE encoder/decoder using the ELBO objective.

        Parameters
        ----------
        X:
            Training features with shape ``(n_samples, n_features)``.
        y:
            Training targets with shape ``(n_samples,)``.
        schema:
            Optional schema overriding the instance-level schema.
        epochs:
            Number of optimisation epochs for the ELBO objective.
        batch_size:
            Overrides the batch size specified during initialisation.
        kl_warmup_epochs:
            Overrides the KL warm-up epochs specified during initialisation.

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
        self.schema.require_columns(X.columns)
        unsupported = [
            column
            for column in self.schema.feature_names
            if self.schema[column].type not in {"real", "cat"}
        ]
        if unsupported:
            joined = ", ".join(unsupported)
            raise ValueError(
                f"Columns {joined} use unsupported feature types. "
                "Enable positive/count/ordinal heads in a future release."
            )

        LOGGER.info("Starting fit: n_samples=%s, n_features=%s", len(X), X.shape[1])
        X_train, X_val, y_train, y_val = data_utils.split_train_val(
            X, y, val_split=self.val_split, stratify=self.stratify
        )
        LOGGER.info("Train/val split: %s/%s", len(X_train), len(X_val))

        missing_mask = data_utils.build_missing_mask(X_train)
        LOGGER.debug(
            "Missing mask summary: %s missing values", int(missing_mask.sum().sum())
        )

        X_train_std, stats = data_utils.standardize(X_train, self.schema)
        self._norm_stats_per_col = stats
        _ = data_utils.standardize(X_val, self.schema)

        y_train_array = np.asarray(y_train)
        self._classes = np.unique(y_train_array)
        if self._classes.size == 0:
            raise ValueError("Training targets must contain at least one class")
        self._class_to_index = {
            cls: idx for idx, cls in enumerate(self._classes.tolist())
        }
        train_target_indices = self._map_targets_to_indices(y_train_array)
        class_counts = np.bincount(
            train_target_indices, minlength=self._classes.size
        ).astype(np.float32)
        total_count = float(class_counts.sum())
        if total_count <= 0:
            raise ValueError("Training targets must contain at least one class")
        class_weights = total_count / (len(class_counts) * class_counts)

        batch_size = batch_size or self.batch_size
        kl_warmup_epochs = kl_warmup_epochs or self.kl_warmup_epochs
        encoder_inputs, data_tensors, mask_tensors = self._prepare_training_tensors(
            X_train_std, missing_mask
        )

        device = self._select_device()
        encoder_inputs = encoder_inputs.to(device)
        for feature_type in data_tensors:
            for column, tensor in data_tensors[feature_type].items():
                data_tensors[feature_type][column] = tensor.to(device)
                mask_tensors[feature_type][column] = mask_tensors[feature_type][
                    column
                ].to(device)

        torch.manual_seed(self.random_state)
        self._encoder = EncoderMLP(
            encoder_inputs.size(-1),
            self.latent_dim,
            hidden=self.hidden_dims,
            dropout=self.dropout,
        ).to(device)
        self._decoder = Decoder(
            self.latent_dim,
            self.schema,
            hidden=self.hidden_dims,
            dropout=self.dropout,
        ).to(device)
        self._classifier = ClassificationHead(
            self.latent_dim,
            self._classes.size,
            class_weight=class_weights,
        ).to(device)
        parameters = (
            list(self._encoder.parameters())
            + list(self._decoder.parameters())
            + list(self._classifier.parameters())
        )
        optimizer = Adam(parameters, lr=self.learning_rate)

        y_train_tensor = torch.from_numpy(train_target_indices.astype(np.int64)).to(
            device
        )

        n_samples = encoder_inputs.size(0)
        effective_batch = min(batch_size, n_samples)
        n_batches = max(1, math.ceil(n_samples / effective_batch))
        warmup_steps = max(1, kl_warmup_epochs * n_batches)
        global_step = 0

        progress = tqdm(range(epochs), desc="Training", leave=False)
        for epoch in progress:
            permutation = torch.randperm(n_samples, device=device)
            epoch_loss = 0.0
            for start in range(0, n_samples, effective_batch):
                batch_indices = permutation[start : start + effective_batch]
                batch_input = encoder_inputs[batch_indices]
                batch_data = {
                    key: {
                        column: tensor[batch_indices]
                        for column, tensor in tensors.items()
                    }
                    for key, tensors in data_tensors.items()
                }
                batch_masks = {
                    key: {
                        column: tensor[batch_indices]
                        for column, tensor in tensors.items()
                    }
                    for key, tensors in mask_tensors.items()
                }

                mu_z, logvar_z = self._encoder(batch_input)
                z = self._reparameterize(mu_z, logvar_z)
                decoder_out = self._decoder(
                    z, batch_data, self._norm_stats_per_col, batch_masks
                )
                recon_terms = decoder_out["log_px"]
                global_step += 1
                beta_scale = losses.kl_warmup(global_step, warmup_steps, self.beta)
                kl = losses.kl_normal(mu_z, logvar_z) * beta_scale
                elbo_value = losses.elbo(recon_terms, kl)
                logits = self._classifier(z)
                batch_targets = y_train_tensor[batch_indices]
                classification_loss = self._classifier.loss(logits, batch_targets)
                loss = -elbo_value.mean() + classification_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            progress.set_postfix({"loss": epoch_loss / n_batches})

        self._is_fitted = True
        self._is_calibrated = False
        self._temperature_scaler = TemperatureScaler()
        self._temperature_scaler_state = None
        self._cached_logits = None
        self._cached_probabilities = None
        LOGGER.info("Fit complete")
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_device(self) -> torch.device:
        """Return the computation device, warning if CUDA is unavailable."""

        if self._device is not None:
            return self._device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            LOGGER.warning("CUDA not available; falling back to CPU")
            self._device = torch.device("cpu")
        return self._device

    def _prepare_training_tensors(
        self,
        X: pd.DataFrame,
        mask: pd.DataFrame,
        *,
        update_layout: bool = True,
    ) -> Tuple[
        Tensor,
        Dict[str, Dict[str, Tensor]],
        Dict[str, Dict[str, Tensor]],
    ]:
        """Convert normalised dataframes into tensors for optimisation."""

        n_samples = len(X)
        value_parts: list[Tensor] = []
        mask_parts: list[Tensor] = []
        real_data: Dict[str, Tensor] = {}
        cat_data: Dict[str, Tensor] = {}
        real_masks: Dict[str, Tensor] = {}
        cat_masks: Dict[str, Tensor] = {}

        feature_layout = {"real": {}, "cat": {}}

        for column in self.schema.real_features:
            values = pd.to_numeric(X[column], errors="coerce").to_numpy(
                dtype=np.float32
            )
            values = np.nan_to_num(values, nan=0.0).reshape(n_samples, 1)
            tensor = torch.from_numpy(values)
            missing = mask[column].to_numpy().astype(bool)
            observed = (~missing).astype(np.float32).reshape(n_samples, 1)
            mask_tensor = torch.from_numpy(observed)
            real_data[column] = tensor
            real_masks[column] = mask_tensor
            value_parts.append(tensor)
            mask_parts.append(mask_tensor)
            feature_layout["real"][column] = 1

        for column in self.schema.categorical_features:
            series = X[column]
            if not isinstance(series.dtype, CategoricalDtype):
                series = series.astype("category")
            codes = series.cat.codes.to_numpy()
            n_classes = int(self.schema[column].n_classes or series.cat.categories.size)
            if n_classes <= 0:
                raise ValueError(f"Categorical column '{column}' must define classes")
            onehot = np.zeros((n_samples, n_classes), dtype=np.float32)
            valid = codes >= 0
            indices = np.where(valid)[0]
            if indices.size:
                onehot[indices, codes[indices]] = 1.0
            tensor = torch.from_numpy(onehot)
            missing = mask[column].to_numpy().astype(bool)
            observed = (~missing).astype(np.float32).reshape(n_samples, 1)
            mask_tensor = torch.from_numpy(observed)
            cat_data[column] = tensor
            cat_masks[column] = mask_tensor
            value_parts.append(tensor)
            mask_parts.append(mask_tensor)
            feature_layout["cat"][column] = n_classes

        if not value_parts:
            raise ValueError("No supported features present in the training data")

        encoder_inputs = torch.cat(value_parts + mask_parts, dim=1).float()
        data_tensors = {"real": real_data, "cat": cat_data}
        mask_tensors = {"real": real_masks, "cat": cat_masks}
        if update_layout:
            self._feature_layout = feature_layout
        return encoder_inputs, data_tensors, mask_tensors

    @staticmethod
    def _reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample latent variables using the reparameterisation trick."""

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _apply_training_normalization(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply stored normalisation statistics to ``X``."""

        if self.schema is None:
            raise RuntimeError("Schema is required for normalisation")
        self.schema.require_columns(X.columns)
        normalised = X.reset_index(drop=True).copy()
        for column in self.schema.real_features:
            stats = self._norm_stats_per_col.get(column, {})
            mean = float(stats.get("mean", 0.0))
            std = float(stats.get("std", 1.0))
            if std <= 0:
                std = 1.0
            values = pd.to_numeric(normalised[column], errors="coerce")
            normalised[column] = (values - mean) / max(std, 1e-6)
        for column in self.schema.categorical_features:
            stats = self._norm_stats_per_col.get(column, {})
            categories = stats.get("categories")
            if categories is not None:
                normalised[column] = pd.Categorical(
                    normalised[column], categories=categories
                )
            else:
                normalised[column] = normalised[column].astype("category")
        return normalised

    def _prepare_inference_inputs(self, X: pd.DataFrame) -> Tensor:
        """Return encoder inputs for inference using stored statistics."""

        if self.schema is None:
            raise RuntimeError("Schema must be defined for inference")
        aligned = X.reset_index(drop=True)
        mask = data_utils.build_missing_mask(aligned)
        normalised = self._apply_training_normalization(aligned)
        encoder_inputs, _, _ = self._prepare_training_tensors(
            normalised, mask, update_layout=False
        )
        return encoder_inputs.float()

    def _map_targets_to_indices(self, targets: Iterable[object]) -> np.ndarray:
        """Convert raw targets into class indices."""

        if self._class_to_index is None:
            raise RuntimeError("Model classes are not initialised")
        target_array = np.asarray(targets).reshape(-1)
        indices: list[int] = []
        for value in target_array:
            if value not in self._class_to_index:
                raise ValueError(f"Unknown class '{value}' encountered")
            indices.append(self._class_to_index[value])
        return np.asarray(indices, dtype=np.int64)

    def _compute_logits(self, X: pd.DataFrame) -> np.ndarray:
        """Run the classifier head on ``X`` and cache the logits."""

        if not self._is_fitted or self._encoder is None or self._classifier is None:
            raise RuntimeError("Model must be fitted before computing logits")
        device = self._select_device()
        encoder_inputs = self._prepare_inference_inputs(X).to(device)
        was_encoder_training = self._encoder.training
        was_classifier_training = self._classifier.training
        self._encoder.eval()
        self._classifier.eval()
        with torch.no_grad():
            mu, _ = self._encoder(encoder_inputs)
            logits_tensor = self._classifier(mu)
        if was_encoder_training:
            self._encoder.train()
        if was_classifier_training:
            self._classifier.train()
        logits = logits_tensor.cpu().numpy()
        self._cached_logits = logits
        return logits

    @staticmethod
    def _logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities with numerical stability."""

        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 2:
            raise ValueError("logits must be a 2D array")
        stabilised = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(stabilised)
        normaliser = probabilities.sum(axis=1, keepdims=True)
        normaliser[normaliser == 0.0] = 1.0
        return probabilities / normaliser

    def _infer_latent_statistics(self, X: pd.DataFrame) -> tuple[Tensor, Tensor]:
        """Return posterior parameters for ``X`` using the trained encoder."""

        if not self._is_fitted or self._encoder is None:
            raise RuntimeError("Model must be fitted before encoding data")
        device = self._select_device()
        encoder_inputs = self._prepare_inference_inputs(X).to(device)
        was_training = self._encoder.training
        self._encoder.eval()
        with torch.no_grad():
            mu, logvar = self._encoder(encoder_inputs)
        if was_training:
            self._encoder.train()
        return mu, logvar

    # ------------------------------------------------------------------
    # Prediction utilities
    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated class probabilities for ``X``.

        Parameters
        ----------
        X:
            Input features with shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples, n_classes)`` containing the calibrated
            probabilities produced by the classifier head.

        Examples
        --------
        >>> proba = model.predict_proba(X)
        >>> proba.sum(axis=1)
        array([1., 1.])
        """

        if not self._is_fitted or self._classes is None:
            raise RuntimeError("Model must be fitted before calling predict_proba")
        logits = self._compute_logits(X)
        if self._is_calibrated:
            logits = self._temperature_scaler.transform(logits)
        probabilities = self._logits_to_probabilities(logits)
        self._cached_probabilities = probabilities
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
        """Fit the temperature scaler using logits from ``X``."""

        if not self._is_fitted or self._classes is None:
            raise RuntimeError("Fit must be called before calibrate")
        if len(X) != len(y):
            raise ValueError("X and y must have matching first dimensions")
        logits = self._compute_logits(X)
        probabilities = self._logits_to_probabilities(logits)
        target_indices = self._map_targets_to_indices(y)
        self._temperature_scaler.fit(logits, target_indices)
        self._temperature_scaler_state = self._temperature_scaler.state_dict()
        self._is_calibrated = True
        self._cached_probabilities = probabilities
        return self

    # ------------------------------------------------------------------
    # Latent utilities and sampling
    # ------------------------------------------------------------------
    def encode(self, X: pd.DataFrame) -> np.ndarray:
        """Return posterior means of the latent representation for ``X``.

        Parameters
        ----------
        X:
            Input features with shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples, latent_dim)`` containing the latent
            posterior means produced by the trained encoder.

        Examples
        --------
        >>> latents = model.encode(X)
        >>> latents.shape
        (len(X), model.latent_dim)
        """

        if not self._is_fitted or self._encoder is None:
            raise RuntimeError("Model must be fitted before encoding data")

        device = self._select_device()
        encoder_inputs = self._prepare_inference_inputs(X).to(device)
        n_samples = encoder_inputs.size(0)
        if n_samples == 0:
            return np.empty((0, self.latent_dim), dtype=np.float32)

        effective_batch = min(self.batch_size, n_samples)
        if effective_batch <= 0:
            effective_batch = n_samples

        latent_batches: list[Tensor] = []
        was_training = self._encoder.training
        self._encoder.eval()
        with torch.no_grad():
            for start in range(0, n_samples, effective_batch):
                end = min(start + effective_batch, n_samples)
                batch_inputs = encoder_inputs[start:end]
                mu, _ = self._encoder(batch_inputs)
                latent_batches.append(mu.cpu())
        if was_training:
            self._encoder.train()

        return torch.cat(latent_batches, dim=0).numpy()

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
            "normalization": self._norm_stats_per_col,
            "temperature_scaler": self._temperature_scaler_state,
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
        model._norm_stats_per_col = data.get("normalization", {})
        scaler_state = data.get("temperature_scaler")
        if scaler_state:
            model._temperature_scaler_state = scaler_state
            model._temperature_scaler.load_state_dict(scaler_state)
            model._is_calibrated = True
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
