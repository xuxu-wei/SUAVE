"""Core model definitions for the minimal SUAVE package."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import warnings
from pathlib import Path
import pickle
from collections import OrderedDict
from typing import Any, Dict, Iterable, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import CategoricalDtype
from pandas.util import hash_pandas_object
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter
from torch.distributions import Categorical
from torch.optim import Adam
from tqdm.auto import tqdm

from . import data as data_utils
from .modules.calibrate import TemperatureScaler
from .modules.decoder import Decoder
from .modules.encoder import EncoderMLP
from .modules.heads import ClassificationHead
from .modules import losses, distributions as dist_utils
from .modules.prior import PriorMean
from . import sampling as sampling_utils
from .types import Schema, ColumnSpec
from .evaluate import compute_brier, compute_ece
from .defaults import (
    parse_heuristic_hyperparameters,
    recommend_hyperparameters,
    serialise_heuristic_hyperparameters,
)

LOGGER = logging.getLogger(__name__)


_DEFAULT_LATENT_DIM = 32
_DEFAULT_N_COMPONENTS = 1
_DEFAULT_BETA = 1.5
_DEFAULT_HIDDEN_DIMS = (256, 128)
_DEFAULT_HEAD_HIDDEN_DIMS: tuple[int, ...] = ()
_DEFAULT_DROPOUT = 0.1
_DEFAULT_LEARNING_RATE = 1e-3
_DEFAULT_BATCH_SIZE = 128
_DEFAULT_KL_WARMUP_EPOCHS = 10
_DEFAULT_VAL_SPLIT = 0.2
_DEFAULT_STRATIFY = True
_DEFAULT_RANDOM_STATE = 0
_DEFAULT_GUMBEL_TEMPERATURE = 1.0
_DEFAULT_WARMUP_EPOCHS = 10
_DEFAULT_HEAD_EPOCHS = 5
_DEFAULT_FINETUNE_EPOCHS = 10
_DEFAULT_JOINT_DECODER_LR_SCALE = 0.1
_DEFAULT_EARLY_STOP_PATIENCE = 5

_BEHAVIOUR_ALIASES = {"suave": "supervised", "hivae": "unsupervised"}


def _normalise_behaviour(value: str) -> str:
    """Return the canonical behaviour label for ``value``."""

    normalised = value.lower()
    return _BEHAVIOUR_ALIASES.get(normalised, normalised)


class SUAVE:
    """
    Model for mixed tabular data with supervised and unsupervised branches.

    This estimator couples a VAE-style latent model for heterogeneous features
    with an optional supervised classification head. It supports two operating
    modes:

    - ``behaviour="supervised"``: encoder/decoder + classifier head.
    - ``behaviour="unsupervised"``: encoder/decoder only (head is disabled).

    Parameters
    ----------
    schema : Schema, optional
        Dataset description including column types and cardinalities.  When not
        supplied during initialisation, a schema must be provided to
        :meth:`fit`.
    behaviour : {"supervised", "unsupervised"}, default "supervised"
        Selects the feature set exposed by the estimator.  ``"supervised"``
        enables the classification head whereas ``"unsupervised"`` activates the
        generative-only workflow.
    latent_dim : int, optional
        Dimensionality of the latent representation shared between the encoder
        and decoder networks.  When ``None`` the value is selected
        heuristically during :meth:`fit` based on dataset statistics.
    n_components : int, default 1
        Number of mixture components parameterising the hierarchical latent
        prior.  ``1`` corresponds to a standard Gaussian encoder.
    beta : float, default 1.5
        Weight applied to the KL divergence term in the evidence lower bound
        objective.
    hidden_dims : Iterable[int], optional
        Width of each hidden layer used in the encoder and decoder multilayer
        perceptrons.  ``None`` defers to the heuristic defaults chosen at
        training time.
    head_hidden_dims : Iterable[int], default ()
        Width of optional hidden layers inserted into the classification head.
        Each hidden layer follows a ``Linear → ReLU → Dropout`` pattern before
        the final ``n_classes`` projection.
    dropout : float, optional
        Dropout probability applied inside the neural modules.  When omitted a
        dataset-size-aware default is selected during :meth:`fit`.
    learning_rate : float, optional
        Learning rate for the Adam optimiser driving all optimisation stages.
        ``None`` enables heuristic selection at training time.
    batch_size : int, optional
        Mini-batch size consumed by :meth:`fit` and downstream inference
        utilities.  Leaving this ``None`` delegates the choice to the
        heuristics.
    warmup_epochs : int, optional
        Number of epochs dedicated to ELBO-only optimisation before the
        classifier head is trained.  ``None`` activates heuristic scheduling.
    kl_warmup_epochs : int, optional
        Number of epochs used to linearly anneal the KL divergence weight during
        the warm-up phase.  ``None`` activates heuristic scheduling.
    head_epochs : int, optional
        Number of epochs allocated to the classifier-head-only optimisation
        stage.  ``None`` activates heuristic scheduling.
    finetune_epochs : int, optional
        Duration of the joint fine-tuning phase performed after the head stage.
        ``None`` activates heuristic scheduling.
    early_stop_patience : int, optional
        Patience of the validation early-stopping monitor used during joint
        fine-tuning before the best checkpoint is restored.  ``None`` activates
        heuristic scheduling.
    joint_decoder_lr_scale : float, default 0.1
        Multiplicative factor applied to the decoder and prior learning rates
        during joint fine-tuning relative to the encoder rate.
    val_split : float, default 0.2
        Ratio of samples assigned to the internal validation split constructed
        inside :meth:`fit`.
    stratify : bool, default True
        Whether to preserve label balance when generating the validation split.
    random_state : int, default 0
        Deterministic seed used by helper utilities that rely on randomness.
    gumbel_temperature : float, default 1.0
        Temperature applied by the straight-through Gumbel-Softmax estimator
        when sampling mixture assignments during training.
    tau_start : float, default 1.0
        Initial temperature used by the unsupervised mixture sampler prior to
        annealing.
    tau_min : float, default 1e-3
        Final temperature reached after annealing in the unsupervised branch,
        leading to near-deterministic component assignments.
    tau_decay : float, default 0.01
        Linear decrement applied to the temperature at each epoch when
        ``behaviour="unsupervised"``.

    Attributes
    ----------
    `auto_hyperparameters_` : dict[str, object] or None
        Description of heuristically selected hyperparameters after calling
        :meth:`fit`.  ``None`` when every configurable value was supplied by the
        user.

    See Also
    --------
    `fit` : Optimize encoder/decoder and (when supervised) the classifier head.
    `calibrate` : Temperature-scale logits on held-out data for better calibration.
    `predict_proba` : Calibrated class probabilities (or posterior predictive for an attribute).
    `predict` : Class labels (supervised) or attribute predictions/samples.
    `predict_confidence_interval` : Posterior predictive mean/median and CI for real/pos/count.
    `encode` : Latent posterior means (optionally component assignments/params).
    `impute` : Single-pass decoder imputation of missing entries.
    `sample` : Draw synthetic rows (optionally conditional on labels in supervised mode).
    `save` : Serialize a trained model to disk.
    `load` : Restore a serialized model and make it ready for inference.

    --------
    Training schedule (how the epoch knobs interact)
    --------
    The full training routine is divided into stages. Understanding the schedule
    clarifies how ``warmup_epochs``, ``kl_warmup_epochs``, ``head_epochs`` and
    ``finetune_epochs`` work together.

    1) Warm-up (ELBO-only):
       Controlled by ``warmup_epochs``.
       - Only the encoder/decoder/prior are optimized against the ELBO
         (reconstruction - β·KL). The classifier head is **not** used here.
       - ``kl_warmup_epochs`` controls the *annealing of β* **within** this
         warm-up stage: β is linearly ramped from ~0 to the configured ``beta``
         across ``kl_warmup_epochs * n_batches`` steps.
       - Practical implication:
         * If ``kl_warmup_epochs <= warmup_epochs``, β typically reaches its
           target by the end of warm-up.
         * If ``kl_warmup_epochs > warmup_epochs``, β may still be < target at
           warm-up end (intentional if you want a gentler KL schedule).
         * If ``warmup_epochs == 0``, no ELBO warm-up or KL annealing occurs.

    2) Head-only (supervised mode only):
       Controlled by ``head_epochs`` (ignored in unsupervised mode).
       - The classifier head is trained **alone** on cached latent means; the
         encoder/decoder are frozen. This isolates the head’s optimization.

    3) Joint fine-tuning (supervised mode only):
       Controlled by ``finetune_epochs`` (ignored in unsupervised mode).
       - Encoder, decoder, prior and head are optimized **together**.
       - ``joint_decoder_lr_scale`` scales decoder/prior LR relative to the
         encoder/head (often < 1 to stabilize updates).
       - Early stopping is applied with ``early_stop_patience`` on validation
         metrics; the best checkpoint is restored at the end.

    Visual timeline (supervised):
        [ ELBO warm-up ]  ->  [ head-only ]  ->  [ joint fine-tune ]
        warmup_epochs         head_epochs          finetune_epochs

    Visual timeline (unsupervised):
        [ ELBO warm-up only ]
        warmup_epochs
        (``head_epochs`` and ``finetune_epochs`` are ignored)

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
        behaviour: Literal["supervised", "unsupervised"] = "supervised",
        latent_dim: Optional[int] = None,
        n_components: int = _DEFAULT_N_COMPONENTS,
        beta: float = _DEFAULT_BETA,
        hidden_dims: Optional[Iterable[int]] = None,
        head_hidden_dims: Iterable[int] = _DEFAULT_HEAD_HIDDEN_DIMS,
        dropout: Optional[float] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        warmup_epochs: Optional[int] = None,
        kl_warmup_epochs: Optional[int] = None,
        head_epochs: Optional[int] = None,
        finetune_epochs: Optional[int] = None,
        early_stop_patience: Optional[int] = None,
        joint_decoder_lr_scale: float = _DEFAULT_JOINT_DECODER_LR_SCALE,
        val_split: float = _DEFAULT_VAL_SPLIT,
        stratify: bool = _DEFAULT_STRATIFY,
        random_state: int = _DEFAULT_RANDOM_STATE,
        gumbel_temperature: float = _DEFAULT_GUMBEL_TEMPERATURE,
        tau_start: float = 1.0,
        tau_min: float = 1e-3,
        tau_decay: float = 0.01,
    ) -> None:
        self.schema = schema

        behaviour_normalised = _normalise_behaviour(str(behaviour))
        if behaviour_normalised not in {"supervised", "unsupervised"}:
            raise ValueError("behaviour must be either 'supervised' or 'unsupervised'")
        self.behaviour = behaviour_normalised

        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = int(n_components)
        self.beta = float(beta)

        latent_dim_user = latent_dim is not None
        latent_dim_value = (
            _DEFAULT_LATENT_DIM if latent_dim is None else int(latent_dim)
        )
        if latent_dim_value <= 0:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = latent_dim_value

        if hidden_dims is None:
            hidden_dims_value = tuple(int(dim) for dim in _DEFAULT_HIDDEN_DIMS)
            hidden_dims_user = False
        else:
            hidden_dims_value = tuple(int(dim) for dim in hidden_dims)
            hidden_dims_user = True
        if any(dim <= 0 for dim in hidden_dims_value):
            raise ValueError("hidden_dims must contain positive integers")
        self.hidden_dims = hidden_dims_value

        head_hidden_dims = tuple(int(dim) for dim in head_hidden_dims)
        if any(dim <= 0 for dim in head_hidden_dims):
            raise ValueError("head_hidden_dims must contain positive integers")
        self.head_hidden_dims = head_hidden_dims

        dropout_user = dropout is not None
        dropout_value = float(_DEFAULT_DROPOUT if dropout is None else dropout)
        if dropout_value < 0 or dropout_value >= 1:
            raise ValueError("dropout must lie in the [0, 1) range")
        self.dropout = dropout_value

        learning_rate_user = learning_rate is not None
        learning_rate_value = float(
            _DEFAULT_LEARNING_RATE if learning_rate is None else learning_rate
        )
        if learning_rate_value <= 0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = learning_rate_value

        batch_size_user = batch_size is not None
        batch_size_value = int(
            _DEFAULT_BATCH_SIZE if batch_size is None else batch_size
        )
        if batch_size_value <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size_value

        kl_user = kl_warmup_epochs is not None
        kl_value = int(
            _DEFAULT_KL_WARMUP_EPOCHS if kl_warmup_epochs is None else kl_warmup_epochs
        )
        if kl_value < 0:
            raise ValueError("kl_warmup_epochs must be non-negative")
        self.kl_warmup_epochs = kl_value

        warmup_user = warmup_epochs is not None
        warmup_value = int(
            _DEFAULT_WARMUP_EPOCHS if warmup_epochs is None else warmup_epochs
        )
        if warmup_value < 0:
            raise ValueError("warmup_epochs must be non-negative")
        self.warmup_epochs = warmup_value

        head_user = head_epochs is not None
        head_value = int(_DEFAULT_HEAD_EPOCHS if head_epochs is None else head_epochs)
        if head_value < 0:
            raise ValueError("head_epochs must be non-negative")
        self.head_epochs = head_value

        finetune_user = finetune_epochs is not None
        finetune_value = int(
            _DEFAULT_FINETUNE_EPOCHS if finetune_epochs is None else finetune_epochs
        )
        if finetune_value < 0:
            raise ValueError("finetune_epochs must be non-negative")
        self.finetune_epochs = finetune_value

        patience_user = early_stop_patience is not None
        patience_value = int(
            _DEFAULT_EARLY_STOP_PATIENCE
            if early_stop_patience is None
            else early_stop_patience
        )
        if patience_value < 0:
            raise ValueError("early_stop_patience must be non-negative")
        self.early_stop_patience = patience_value

        if joint_decoder_lr_scale <= 0:
            raise ValueError("joint_decoder_lr_scale must be positive")
        self.joint_decoder_lr_scale = float(joint_decoder_lr_scale)

        self.val_split = float(val_split)
        self.stratify = bool(stratify)
        self.random_state = int(random_state)

        if gumbel_temperature <= 0:
            raise ValueError("gumbel_temperature must be positive")
        self.gumbel_temperature = float(gumbel_temperature)

        self.auto_hyperparameters_: dict[str, int | float | tuple[int, ...]] | None = (
            None
        )
        self._auto_configured: dict[str, bool] = {
            "latent_dim": not latent_dim_user,
            "hidden_dims": not hidden_dims_user,
            "head_hidden_dims": False,
            "dropout": not dropout_user,
            "learning_rate": not learning_rate_user,
            "batch_size": not batch_size_user,
            "kl_warmup_epochs": not kl_user,
            "warmup_epochs": not warmup_user,
            "head_epochs": not head_user,
            "finetune_epochs": not finetune_user,
            "early_stop_patience": not patience_user,
        }

        self._tau_start: float | None = None
        self._tau_min: float | None = None
        self._tau_decay: float | None = None
        self._inference_tau: float = 1.0
        if self.behaviour == "unsupervised":
            if tau_start <= 0.0:
                raise ValueError("tau_start must be positive for unsupervised mode")
            if tau_min <= 0.0:
                raise ValueError("tau_min must be positive for unsupervised mode")
            if tau_min > tau_start:
                raise ValueError("tau_min cannot exceed tau_start")
            if tau_decay < 0.0:
                raise ValueError("tau_decay must be non-negative")
            self._tau_start = float(tau_start)
            self._tau_min = float(tau_min)
            self._tau_decay = float(tau_decay)
            self._inference_tau = float(tau_min)

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
        self._feature_layout: dict[str, dict[str, int]] = {
            "real": {},
            "pos": {},
            "count": {},
            "cat": {},
            "ordinal": {},
        }
        self._class_to_index: dict[object, int] | None = None
        self._cached_logits: np.ndarray | None = None
        self._cached_probabilities: np.ndarray | None = None
        self._logits_cache_key: str | None = None
        self._probability_cache_key: str | None = None
        self._warmup_val_history: list[dict[str, float]] = []
        self._joint_val_metrics: dict[str, float] | None = None
        self._train_latent_mu: Tensor | None = None
        self._train_latent_logvar: Tensor | None = None
        self._train_target_indices: np.ndarray | None = None
        self._train_component_logits: Tensor | None = None
        self._train_component_mu: Tensor | None = None
        self._train_component_logvar: Tensor | None = None
        self._train_component_probs: Tensor | None = None
        self._prior_mean_layer: PriorMean | None = None

        self._reset_prior_parameters()

    def _reset_prior_parameters(self) -> None:
        """Initialise trainable tensors for the mixture prior parameters."""

        zeros_logits = torch.zeros(self.n_components, dtype=torch.float32)
        zeros_full = torch.zeros(
            self.n_components, self.latent_dim, dtype=torch.float32
        )

        if self.behaviour == "unsupervised":
            self._prior_component_logits = Parameter(zeros_logits, requires_grad=False)
            self._prior_component_mu = None
            self._prior_component_logvar = Parameter(zeros_full, requires_grad=False)
            self._prior_mean_layer = PriorMean(self.n_components, self.latent_dim)
        else:
            self._prior_component_logits = Parameter(
                zeros_logits.clone(), requires_grad=True
            )
            self._prior_component_mu = Parameter(zeros_full.clone(), requires_grad=True)
            self._prior_component_logvar = Parameter(
                zeros_full.clone(), requires_grad=True
            )
            self._prior_mean_layer = None

    def _move_prior_parameters_to_device(self, device: torch.device) -> None:
        """Ensure mixture prior parameters live on ``device`` as trainable tensors."""

        def _wrap(
            tensor: Tensor | Parameter | None, *, trainable: bool
        ) -> Parameter | None:
            if tensor is None:
                return None
            data = tensor.detach().to(device=device, dtype=torch.float32)
            return Parameter(data, requires_grad=trainable)

        trainable = self.behaviour == "supervised"
        self._prior_component_logits = _wrap(
            self._prior_component_logits, trainable=trainable
        )
        self._prior_component_logvar = _wrap(
            self._prior_component_logvar, trainable=trainable
        )
        if self._prior_component_mu is not None:
            self._prior_component_mu = _wrap(
                self._prior_component_mu, trainable=trainable
            )
        if self._prior_mean_layer is not None:
            self._prior_mean_layer.to(device)

    def _prior_parameters_for_optimizer(self) -> list[Parameter]:
        """Return the list of trainable prior parameters."""

        if self.behaviour == "unsupervised":
            if self._prior_mean_layer is None:
                raise RuntimeError("Prior mean layer is not initialised")
            return list(self._prior_mean_layer.parameters())
        params: list[Parameter] = []
        for param in (
            self._prior_component_logits,
            self._prior_component_mu,
            self._prior_component_logvar,
        ):
            if isinstance(param, Parameter) and param.requires_grad:
                params.append(param)
        return params

    def _prior_component_logits_tensor(self) -> Tensor:
        """Return the tensor of prior logits on the active device."""

        if self._prior_component_logits is None:
            raise RuntimeError("Prior logits are not initialised")
        return self._prior_component_logits

    def _prior_component_means_tensor(self) -> Tensor:
        """Return the matrix of prior means for each mixture component."""

        if self.behaviour == "unsupervised":
            if self._prior_mean_layer is None:
                raise RuntimeError("Prior mean layer is not initialised")
            return self._prior_mean_layer.component_means()
        if self._prior_component_mu is None:
            raise RuntimeError("Prior component means are not initialised")
        return self._prior_component_mu

    def _prior_component_logvar_tensor(self) -> Tensor:
        """Return the diagonal log-variance of the prior distribution."""

        if self._prior_component_logvar is None:
            raise RuntimeError("Prior log-variance is not initialised")
        return self._prior_component_logvar

    def _gumbel_temperature_for_epoch(self, epoch: int) -> float:
        """Return the annealed Gumbel-Softmax temperature for ``epoch``."""

        if self.behaviour != "unsupervised" or self._tau_start is None:
            return 1.0
        tau_start = float(self._tau_start)
        tau_min = float(self._tau_min) if self._tau_min is not None else tau_start
        tau_decay = float(self._tau_decay) if self._tau_decay is not None else 0.0
        temperature = tau_start - tau_decay * float(epoch)
        if temperature < tau_min:
            return tau_min
        return temperature

    def _gumbel_temperature_for_epoch(self, epoch: int) -> float:
        """Return the annealed Gumbel-Softmax temperature for ``epoch``."""

        if self.behaviour != "unsupervised" or self._tau_start is None:
            return 1.0
        tau_start = float(self._tau_start)
        tau_min = float(self._tau_min) if self._tau_min is not None else tau_start
        tau_decay = float(self._tau_decay) if self._tau_decay is not None else 0.0
        temperature = tau_start - tau_decay * float(epoch)
        if temperature < tau_min:
            return tau_min
        return temperature

    def _configure_with_auto_defaults(
        self,
        encoder_inputs: Tensor,
        *,
        n_train_samples: int,
        class_counts: np.ndarray | None,
        overrides: dict[str, bool],
        batch_size: int,
        kl_warmup_epochs: int,
        warmup_epochs: int,
        head_epochs: int,
        finetune_epochs: int,
        early_stop_patience: int,
    ) -> tuple[int, int, int, int, int, int, bool]:
        """Return updated hyperparameters using automatic heuristics."""

        heuristic_targets = (
            "latent_dim",
            "hidden_dims",
            "dropout",
            "learning_rate",
            "batch_size",
            "kl_warmup_epochs",
            "warmup_epochs",
            "head_epochs",
            "finetune_epochs",
            "early_stop_patience",
        )
        needs_heuristic = any(
            self._auto_configured.get(name, False) for name in heuristic_targets
        )
        if not needs_heuristic:
            self.auto_hyperparameters_ = None
            return (
                int(batch_size),
                int(kl_warmup_epochs),
                int(warmup_epochs),
                int(head_epochs),
                int(finetune_epochs),
                int(early_stop_patience),
                False,
            )

        input_dim = int(encoder_inputs.size(-1))
        n_train = max(int(n_train_samples), 1)
        recommendations = recommend_hyperparameters(
            input_dim=input_dim,
            n_train_samples=n_train,
            class_counts=class_counts,
        )
        self.auto_hyperparameters_ = recommendations.to_dict()

        reset_prior = False

        latent_dim = int(recommendations.latent_dim)
        if self._auto_configured.get("latent_dim", False) and latent_dim > 0:
            if latent_dim != self.latent_dim:
                LOGGER.info(
                    "Auto-configuring latent_dim=%s (previously %s)",
                    latent_dim,
                    self.latent_dim,
                )
                self.latent_dim = latent_dim
                reset_prior = True

        hidden_dims = tuple(int(v) for v in recommendations.hidden_dims)
        if self._auto_configured.get("hidden_dims", False) and hidden_dims:
            hidden_tuple = tuple(int(v) for v in hidden_dims)
            if hidden_tuple and hidden_tuple != self.hidden_dims:
                LOGGER.info(
                    "Auto-configuring hidden_dims=%s (previously %s)",
                    hidden_tuple,
                    self.hidden_dims,
                )
                self.hidden_dims = hidden_tuple

        dropout = float(recommendations.dropout)
        if self._auto_configured.get("dropout", False):
            dropout = float(np.clip(dropout, 0.0, 0.95))
            if not math.isclose(
                dropout, float(self.dropout), rel_tol=1e-6, abs_tol=1e-6
            ):
                LOGGER.info(
                    "Auto-configuring dropout=%.3f (previously %.3f)",
                    dropout,
                    self.dropout,
                )
                self.dropout = dropout

        learning_rate = float(recommendations.learning_rate)
        if self._auto_configured.get("learning_rate", False) and learning_rate > 0:
            if not math.isclose(
                learning_rate, float(self.learning_rate), rel_tol=1e-9, abs_tol=1e-12
            ):
                LOGGER.info(
                    "Auto-configuring learning_rate=%.2e (previously %.2e)",
                    learning_rate,
                    self.learning_rate,
                )
                self.learning_rate = learning_rate

        if self._auto_configured.get("batch_size", False) and not overrides.get(
            "batch_size", False
        ):
            recommended_batch = int(max(1, recommendations.batch_size))
            if recommended_batch != self.batch_size:
                LOGGER.info(
                    "Auto-configuring batch_size=%s (previously %s)",
                    recommended_batch,
                    self.batch_size,
                )
                self.batch_size = recommended_batch
            batch_size = self.batch_size
        elif overrides.get("batch_size", False):
            self.batch_size = int(batch_size)

        if self._auto_configured.get("kl_warmup_epochs", False) and not overrides.get(
            "kl_warmup_epochs", False
        ):
            recommended_kl = int(max(1, recommendations.kl_warmup_epochs))
            if recommended_kl != self.kl_warmup_epochs:
                LOGGER.info(
                    "Auto-configuring kl_warmup_epochs=%s (previously %s)",
                    recommended_kl,
                    self.kl_warmup_epochs,
                )
                self.kl_warmup_epochs = recommended_kl
            kl_warmup_epochs = self.kl_warmup_epochs
        elif overrides.get("kl_warmup_epochs", False):
            self.kl_warmup_epochs = int(kl_warmup_epochs)

        if self._auto_configured.get("warmup_epochs", False) and not overrides.get(
            "warmup_epochs", False
        ):
            recommended_warmup = int(max(1, recommendations.warmup_epochs))
            if recommended_warmup != self.warmup_epochs:
                LOGGER.info(
                    "Auto-configuring warmup_epochs=%s (previously %s)",
                    recommended_warmup,
                    self.warmup_epochs,
                )
                self.warmup_epochs = recommended_warmup
            warmup_epochs = self.warmup_epochs
        elif overrides.get("warmup_epochs", False):
            self.warmup_epochs = int(warmup_epochs)

        if self._auto_configured.get("head_epochs", False) and not overrides.get(
            "head_epochs", False
        ):
            recommended_head = int(max(1, recommendations.head_epochs))
            if recommended_head != self.head_epochs:
                LOGGER.info(
                    "Auto-configuring head_epochs=%s (previously %s)",
                    recommended_head,
                    self.head_epochs,
                )
                self.head_epochs = recommended_head
            head_epochs = self.head_epochs
        elif overrides.get("head_epochs", False):
            self.head_epochs = int(head_epochs)

        if self._auto_configured.get("finetune_epochs", False) and not overrides.get(
            "finetune_epochs", False
        ):
            recommended_finetune = int(max(1, recommendations.finetune_epochs))
            if recommended_finetune != self.finetune_epochs:
                LOGGER.info(
                    "Auto-configuring finetune_epochs=%s (previously %s)",
                    recommended_finetune,
                    self.finetune_epochs,
                )
                self.finetune_epochs = recommended_finetune
            finetune_epochs = self.finetune_epochs
        elif overrides.get("finetune_epochs", False):
            self.finetune_epochs = int(finetune_epochs)

        if self._auto_configured.get(
            "early_stop_patience", False
        ) and not overrides.get("early_stop_patience", False):
            recommended_patience = int(max(1, recommendations.early_stop_patience))
            if recommended_patience != self.early_stop_patience:
                LOGGER.info(
                    "Auto-configuring early_stop_patience=%s (previously %s)",
                    recommended_patience,
                    self.early_stop_patience,
                )
                self.early_stop_patience = recommended_patience
            early_stop_patience = self.early_stop_patience
        elif overrides.get("early_stop_patience", False):
            self.early_stop_patience = int(early_stop_patience)

        return (
            int(batch_size),
            int(kl_warmup_epochs),
            int(warmup_epochs),
            int(head_epochs),
            int(finetune_epochs),
            int(early_stop_patience),
            reset_prior,
        )

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series | pd.DataFrame | np.ndarray] = None,
        *,
        schema: Optional[Schema] = None,
        epochs: int | None = None,
        batch_size: Optional[int] = None,
        kl_warmup_epochs: Optional[int] = None,
        warmup_epochs: Optional[int] = None,
        head_epochs: Optional[int] = None,
        finetune_epochs: Optional[int] = None,
        joint_decoder_lr_scale: Optional[float] = None,
        early_stop_patience: Optional[int] = None,
    ) -> "SUAVE":
        """Optimise the encoder, decoder and optional classifier head.

        Parameters
        ----------
        X : pandas.DataFrame
            Training features with shape ``(n_samples, n_features)``. Columns
            must align with the schema provided at construction or supplied via
            :paramref:`schema`.
        y : pandas.Series or pandas.DataFrame or numpy.ndarray, optional
            Target labels aligned with ``X``. Required when
            ``behaviour='supervised'``; ignored by the unsupervised workflow.
        schema : Schema, optional
            Schema overriding the instance-level schema. Useful when the model
            was instantiated without a schema and the information becomes
            available only during fitting.
        epochs : int, optional
            Deprecated alias for :paramref:`warmup_epochs`. Retained for
            backwards compatibility. When provided alongside explicit schedule
            overrides, the specific overrides take precedence.
        batch_size : int, optional
            Overrides the batch size specified at initialisation time.
        kl_warmup_epochs : int, optional
            Number of epochs spent annealing the KL divergence weight.
        warmup_epochs : int, optional
            Duration of the ELBO-only warm start. Defaults to the instance
            configuration. In unsupervised mode this value controls the entire
            training duration.
        head_epochs : int, optional
            Length of the classifier-head-only stage. Ignored when
            ``behaviour='unsupervised'``.
        finetune_epochs : int, optional
            Duration of the joint fine-tuning stage. Ignored in unsupervised
            mode.
        joint_decoder_lr_scale : float, optional
            Decoder/prior learning-rate multiplier to use during joint
            fine-tuning.
        early_stop_patience : int, optional
            Override for the validation early-stopping patience used during the
            joint fine-tuning phase.

        Returns
        -------
        SUAVE
            The fitted estimator (``self``) to support method chaining.

        Raises
        ------
        ValueError
            If the schema is missing, targets are omitted in supervised mode or
            schedule overrides are invalid.

        See Also
        --------
        SUAVE.calibrate : Fit the temperature scaler using held-out logits.
        SUAVE.predict_proba : Produce calibrated probabilities after fitting.

        Examples
        --------
        >>> model = SUAVE(schema=schema)
        >>> model.fit(X, y, warmup_epochs=2, head_epochs=1, finetune_epochs=1)
        SUAVE(...)
        """

        if schema is not None:
            self.schema = schema
        if self.schema is None:
            raise ValueError("A schema must be provided to fit the model")
        self.schema.require_columns(X.columns)
        if self.behaviour == "supervised" and y is None:
            raise ValueError("Targets must be provided when behaviour='supervised'")

        LOGGER.info("Starting fit: n_samples=%s, n_features=%s", len(X), X.shape[1])
        stratify_split = (
            self.stratify and self.behaviour == "supervised" and y is not None
        )
        split_targets = y if y is not None else np.zeros(len(X), dtype=np.int64)
        X_train, X_val, y_train, y_val = data_utils.split_train_val(
            X,
            split_targets,
            val_split=self.val_split,
            stratify=stratify_split,
        )
        LOGGER.info("Train/val split: %s/%s", len(X_train), len(X_val))

        epochs_override = epochs is not None
        warmup_override = warmup_epochs is not None or epochs_override
        head_override = head_epochs is not None
        finetune_override = finetune_epochs is not None
        batch_override = batch_size is not None
        kl_override = kl_warmup_epochs is not None
        patience_override = early_stop_patience is not None

        schedule_warmup = (
            self.warmup_epochs if warmup_epochs is None else int(warmup_epochs)
        )
        if epochs_override and warmup_epochs is None:
            schedule_warmup = int(epochs)
        schedule_head = self.head_epochs if head_epochs is None else int(head_epochs)
        schedule_finetune = (
            self.finetune_epochs if finetune_epochs is None else int(finetune_epochs)
        )
        schedule_lr_scale = (
            self.joint_decoder_lr_scale
            if joint_decoder_lr_scale is None
            else float(joint_decoder_lr_scale)
        )
        schedule_patience = (
            self.early_stop_patience
            if early_stop_patience is None
            else int(early_stop_patience)
        )
        for name, value in {
            "warmup_epochs": schedule_warmup,
            "head_epochs": schedule_head,
            "finetune_epochs": schedule_finetune,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if schedule_lr_scale <= 0:
            raise ValueError("joint_decoder_lr_scale must be positive")
        if schedule_patience < 0:
            raise ValueError("early_stop_patience must be non-negative")

        if self.behaviour == "unsupervised":
            schedule_head = 0
            schedule_finetune = 0

        self._joint_val_metrics = None

        batch_size = batch_size or self.batch_size
        kl_warmup_epochs = kl_warmup_epochs or self.kl_warmup_epochs

        missing_mask = data_utils.build_missing_mask(X_train)
        X_train_std, stats = data_utils.standardize(X_train, self.schema)
        missing_mask = missing_mask | X_train_std.isna()
        self._norm_stats_per_col = stats

        encoder_inputs, data_tensors, mask_tensors = self._prepare_training_tensors(
            X_train_std, missing_mask
        )

        val_missing_mask = data_utils.build_missing_mask(X_val)
        X_val_std = self._apply_training_normalization(X_val)
        val_missing_mask = val_missing_mask | X_val_std.isna()
        val_encoder_inputs, val_data_tensors, val_mask_tensors = (
            self._prepare_training_tensors(
                X_val_std, val_missing_mask, update_layout=False
            )
        )

        class_weights: np.ndarray | None = None
        y_train_tensor: Tensor | None = None
        y_val_tensor: Tensor | None = None
        class_counts: np.ndarray | None = None
        if self.behaviour == "supervised":
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
            y_train_tensor = torch.from_numpy(train_target_indices.astype(np.int64))
            if y_val is not None:
                val_indices = self._map_targets_to_indices(np.asarray(y_val))
                y_val_tensor = torch.from_numpy(val_indices.astype(np.int64))
        else:
            self._classes = None
            self._class_to_index = None

        (
            batch_size,
            kl_warmup_epochs,
            schedule_warmup,
            schedule_head,
            schedule_finetune,
            schedule_patience,
            _reset_prior,
        ) = self._configure_with_auto_defaults(
            encoder_inputs,
            n_train_samples=encoder_inputs.size(0),
            class_counts=class_counts,
            overrides={
                "batch_size": batch_override,
                "kl_warmup_epochs": kl_override,
                "warmup_epochs": warmup_override,
                "head_epochs": head_override,
                "finetune_epochs": finetune_override,
                "early_stop_patience": patience_override,
            },
            batch_size=batch_size,
            kl_warmup_epochs=kl_warmup_epochs,
            warmup_epochs=schedule_warmup,
            head_epochs=schedule_head,
            finetune_epochs=schedule_finetune,
            early_stop_patience=schedule_patience,
        )

        if _reset_prior:
            self._reset_prior_parameters()

        if self.behaviour == "unsupervised":
            schedule_head = 0
            schedule_finetune = 0

        device = self._select_device()
        self._move_prior_parameters_to_device(device)
        encoder_inputs = encoder_inputs.to(device)
        val_encoder_inputs = val_encoder_inputs.to(device)

        def _nested_to_device(
            nested: dict[str, dict[str, Tensor]],
        ) -> dict[str, dict[str, Tensor]]:
            return {
                key: {column: tensor.to(device) for column, tensor in tensors.items()}
                for key, tensors in nested.items()
            }

        data_tensors = _nested_to_device(data_tensors)
        mask_tensors = _nested_to_device(mask_tensors)
        val_data_tensors = _nested_to_device(val_data_tensors)
        val_mask_tensors = _nested_to_device(val_mask_tensors)

        torch.manual_seed(self.random_state)
        self._encoder = EncoderMLP(
            encoder_inputs.size(-1),
            self.latent_dim,
            hidden=self.hidden_dims,
            dropout=self.dropout,
            n_components=self.n_components,
        ).to(device)
        self._decoder = Decoder(
            self.latent_dim,
            self.schema,
            hidden=self.hidden_dims,
            dropout=self.dropout,
            n_components=self.n_components,
        ).to(device)
        self._classifier = None

        if y_train_tensor is not None:
            y_train_tensor = y_train_tensor.to(device)
        if y_val_tensor is not None:
            y_val_tensor = y_val_tensor.to(device)

        warmup_history = self._run_warmup_phase(
            schedule_warmup,
            encoder_inputs,
            data_tensors,
            mask_tensors,
            val_encoder_inputs,
            val_data_tensors,
            val_mask_tensors,
            batch_size=batch_size,
            kl_warmup_epochs=kl_warmup_epochs,
        )

        if self.behaviour == "unsupervised" and self._tau_start is not None:
            self._inference_tau = warmup_history.get("final_temperature", 1.0)

        posterior_stats = self._collect_posterior_statistics(
            encoder_inputs,
            batch_size=batch_size,
            temperature=(
                self._inference_tau if self.behaviour == "unsupervised" else None
            ),
        )

        if self.behaviour == "supervised":
            assert self._classes is not None
            self._classifier = ClassificationHead(
                self.latent_dim,
                self._classes.size,
                class_weight=class_weights,
                dropout=self.dropout,
                hidden_dims=self.head_hidden_dims,
            ).to(device)
            self._train_head_phase(
                posterior_stats["mean"],
                y_train_tensor,
                epochs=schedule_head,
                batch_size=batch_size,
            )

            self._run_joint_finetune(
                schedule_finetune,
                encoder_inputs,
                data_tensors,
                mask_tensors,
                val_encoder_inputs,
                val_data_tensors,
                val_mask_tensors,
                batch_size=batch_size,
                lr_scale=schedule_lr_scale,
                early_stop_patience=schedule_patience,
                y_train_tensor=y_train_tensor,
                y_val_tensor=y_val_tensor,
            )

            cache_stats = self._collect_posterior_statistics(
                encoder_inputs,
                batch_size=batch_size,
                temperature=(
                    self._inference_tau if self.behaviour == "unsupervised" else None
                ),
            )
        else:
            cache_stats = posterior_stats
            history = warmup_history.get("history", [])
            self._joint_val_metrics = history[-1] if history else None
        self._cache_training_statistics(
            cache_stats,
            y_train_tensor.cpu().numpy() if y_train_tensor is not None else None,
        )

        self.warmup_epochs = schedule_warmup
        self.head_epochs = schedule_head
        self.finetune_epochs = schedule_finetune
        self.joint_decoder_lr_scale = schedule_lr_scale
        self.early_stop_patience = schedule_patience

        self._is_fitted = True
        self._is_calibrated = False
        self._temperature_scaler = TemperatureScaler()
        self._temperature_scaler_state = None
        self._cached_logits = None
        self._cached_probabilities = None
        self._logits_cache_key = None
        self._probability_cache_key = None
        self._warmup_val_history = warmup_history.get("history", [])

        LOGGER.info("Fit complete")
        return self

    def _run_warmup_phase(
        self,
        warmup_epochs: int,
        encoder_inputs: Tensor,
        data_tensors: dict[str, dict[str, Tensor]],
        mask_tensors: dict[str, dict[str, Tensor]],
        val_inputs: Tensor,
        val_data_tensors: dict[str, dict[str, Tensor]],
        val_mask_tensors: dict[str, dict[str, Tensor]],
        *,
        batch_size: int,
        kl_warmup_epochs: int,
    ) -> dict[str, Any]:
        """Execute the ELBO warm-start stage and record validation metrics."""

        history: list[dict[str, float]] = []
        device = encoder_inputs.device
        if warmup_epochs <= 0:
            temperature = (
                self._gumbel_temperature_for_epoch(0)
                if self.behaviour == "unsupervised"
                else None
            )
            metrics = self._compute_elbo_on_dataset(
                val_inputs,
                val_data_tensors,
                val_mask_tensors,
                batch_size=batch_size,
                temperature=temperature,
            )
            if metrics:
                history.append(metrics)
            final_temperature = temperature if temperature is not None else 1.0
            return {"final_temperature": float(final_temperature), "history": history}

        parameters = list(self._encoder.parameters())
        parameters.extend(self._decoder.parameters())
        parameters.extend(self._prior_parameters_for_optimizer())
        optimizer = Adam(parameters, lr=self.learning_rate)

        n_samples = encoder_inputs.size(0)
        effective_batch = min(batch_size, n_samples) if n_samples else batch_size
        n_batches = max(1, math.ceil(max(n_samples, 1) / max(effective_batch, 1)))
        warmup_steps = max(1, kl_warmup_epochs * n_batches)
        global_step = 0

        final_temperature = self._gumbel_temperature_for_epoch(0)
        description = (
            "Warm-start" if self.behaviour == "supervised" else "unsupervised training"
        )
        progress = tqdm(range(warmup_epochs), desc=description, leave=False)
        for epoch in progress:
            temperature = (
                self._gumbel_temperature_for_epoch(epoch)
                if self.behaviour == "unsupervised"
                else None
            )
            if temperature is not None:
                final_temperature = temperature
            permutation = (
                torch.randperm(n_samples, device=device)
                if n_samples
                else torch.tensor([], device=device, dtype=torch.long)
            )
            epoch_loss = 0.0
            for start in range(0, n_samples, max(effective_batch, 1)):
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

                global_step += 1
                beta_scale = losses.kl_warmup(global_step, warmup_steps, self.beta)
                outputs = self._forward_elbo_batch(
                    batch_input,
                    batch_data,
                    batch_masks,
                    beta_scale=beta_scale,
                    temperature=temperature,
                )
                loss = outputs["loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

            progress.set_postfix({"loss": epoch_loss / n_batches})
            metrics = self._compute_elbo_on_dataset(
                val_inputs,
                val_data_tensors,
                val_mask_tensors,
                batch_size=batch_size,
                temperature=temperature,
            )
            if metrics:
                history.append(metrics)

        final_temperature = final_temperature if final_temperature is not None else 1.0
        return {"final_temperature": float(final_temperature), "history": history}

    def _forward_elbo_batch(
        self,
        batch_input: Tensor,
        batch_data: dict[str, dict[str, Tensor]],
        batch_masks: dict[str, dict[str, Tensor]],
        *,
        beta_scale: float,
        temperature: float | None,
    ) -> dict[str, Tensor]:
        """Return ELBO components for a mini-batch."""

        component_logits, component_mu, component_logvar = self._encoder(batch_input)
        component_probs = torch.softmax(component_logits, dim=-1)
        posterior_weights = component_probs
        if temperature is not None and self.behaviour == "unsupervised":
            gumbel_tau = max(float(temperature), 1e-6)
            posterior_weights = F.gumbel_softmax(
                component_logits, tau=gumbel_tau, hard=False, dim=-1
            )

        categorical_kl = losses.kl_categorical(
            component_logits, self._prior_component_logits_tensor()
        )

        if self.behaviour == "supervised":
            assignment_tau = max(float(self.gumbel_temperature), 1e-6)
            assignments = F.gumbel_softmax(
                component_logits, tau=assignment_tau, hard=True
            )
            component_indices = assignments.argmax(dim=-1)
            selected_mu = self._gather_component_parameters(
                component_mu, component_indices
            )
            selected_logvar = self._gather_component_parameters(
                component_logvar, component_indices
            )
            latent = self._reparameterize(selected_mu, selected_logvar)
            decoder_out = self._decoder(
                latent,
                assignments,
                batch_data,
                self._norm_stats_per_col,
                batch_masks,
            )
            recon_sum = losses.sum_reconstruction_terms(decoder_out["log_px"])
            prior_means = self._prior_component_means_tensor()
            prior_logvars = self._prior_component_logvar_tensor()
            prior_mu = prior_means.index_select(0, component_indices)
            prior_logvar = prior_logvars.index_select(0, component_indices)
            gaussian_kl = losses.kl_normal_vs_normal(
                selected_mu, selected_logvar, prior_mu, prior_logvar
            )
            total_kl_unscaled = categorical_kl + gaussian_kl
            total_kl = beta_scale * total_kl_unscaled
            elbo_value = recon_sum - total_kl
            loss = -elbo_value.mean()
            reconstruction = recon_sum
        else:
            z_samples = self._reparameterize(component_mu, component_logvar)
            component_log_px: list[Tensor] = []
            for component_idx in range(self.n_components):
                component_assignments = F.one_hot(
                    torch.full(
                        (z_samples.size(0),),
                        component_idx,
                        device=z_samples.device,
                        dtype=torch.long,
                    ),
                    num_classes=self.n_components,
                ).float()
                decoder_out = self._decoder(
                    z_samples[:, component_idx, :],
                    component_assignments,
                    batch_data,
                    self._norm_stats_per_col,
                    batch_masks,
                )
                recon_sum = losses.sum_reconstruction_terms(decoder_out["log_px"])
                component_log_px.append(recon_sum)
            component_log_px_tensor = torch.stack(component_log_px, dim=-1)
            gaussian_kl = losses.kl_normal_mixture(
                component_mu,
                component_logvar,
                self._prior_component_means_tensor(),
                self._prior_component_logvar_tensor(),
                posterior_weights,
            )
            total_kl_unscaled = categorical_kl + gaussian_kl
            total_kl = beta_scale * total_kl_unscaled
            reconstruction = (posterior_weights * component_log_px_tensor).sum(dim=-1)
            elbo_value = reconstruction - total_kl
            loss = -elbo_value.mean()
            latent = (posterior_weights.unsqueeze(-1) * z_samples).sum(dim=1)

        return {
            "loss": loss,
            "reconstruction": reconstruction,
            "categorical_kl": categorical_kl,
            "gaussian_kl": gaussian_kl,
            "total_kl": total_kl,
            "kl_unscaled": total_kl_unscaled,
            "latent": latent,
            "component_logits": component_logits,
            "component_mu": component_mu,
            "component_logvar": component_logvar,
        }

    def _collect_posterior_statistics(
        self,
        encoder_inputs: Tensor,
        *,
        batch_size: int,
        temperature: float | None,
    ) -> dict[str, Tensor]:
        """Return posterior summaries for ``encoder_inputs``."""

        n_samples = encoder_inputs.size(0)
        effective_batch = min(batch_size, n_samples) if n_samples else batch_size

        was_training = self._encoder.training
        self._encoder.eval()
        means: list[Tensor] = []
        logvars: list[Tensor] = []
        probs: list[Tensor] = []
        logits: list[Tensor] = []
        mu_params: list[Tensor] = []
        logvar_params: list[Tensor] = []
        with torch.no_grad():
            for start in range(0, n_samples, max(effective_batch, 1)):
                end = start + effective_batch
                batch_input = encoder_inputs[start:end]
                component_logits, component_mu, component_logvar = self._encoder(
                    batch_input
                )
                posterior_mean, posterior_logvar, posterior_probs = (
                    self._mixture_posterior_statistics(
                        component_logits,
                        component_mu,
                        component_logvar,
                        temperature=temperature,
                    )
                )
                means.append(posterior_mean.detach().cpu())
                logvars.append(posterior_logvar.detach().cpu())
                probs.append(posterior_probs.detach().cpu())
                logits.append(component_logits.detach().cpu())
                mu_params.append(component_mu.detach().cpu())
                logvar_params.append(component_logvar.detach().cpu())

        if was_training:
            self._encoder.train()

        return {
            "mean": (
                torch.cat(means, dim=0) if means else torch.empty(0, self.latent_dim)
            ),
            "logvar": (
                torch.cat(logvars, dim=0)
                if logvars
                else torch.empty(0, self.latent_dim)
            ),
            "probs": (
                torch.cat(probs, dim=0) if probs else torch.empty(0, self.n_components)
            ),
            "component_logits": (
                torch.cat(logits, dim=0)
                if logits
                else torch.empty(0, self.n_components)
            ),
            "component_mu": (
                torch.cat(mu_params, dim=0)
                if mu_params
                else torch.empty(0, self.n_components, self.latent_dim)
            ),
            "component_logvar": (
                torch.cat(logvar_params, dim=0)
                if logvar_params
                else torch.empty(0, self.n_components, self.latent_dim)
            ),
        }

    def _train_head_phase(
        self,
        latent_mu: Tensor,
        y_train_tensor: Tensor | None,
        *,
        epochs: int,
        batch_size: int,
    ) -> None:
        """Train the classification head on cached latent representations."""

        if self._classifier is None or y_train_tensor is None or epochs <= 0:
            return

        device = y_train_tensor.device
        latent_mu = latent_mu.to(device)
        n_samples = latent_mu.size(0)
        if n_samples == 0:
            return

        effective_batch = min(batch_size, n_samples)
        n_batches = max(1, math.ceil(n_samples / effective_batch))
        optimizer = Adam(self._classifier.parameters(), lr=self.learning_rate)

        encoder_requires_grad = [
            param.requires_grad for param in self._encoder.parameters()
        ]
        decoder_requires_grad = [
            param.requires_grad for param in self._decoder.parameters()
        ]
        for param in self._encoder.parameters():
            param.requires_grad = False
        for param in self._decoder.parameters():
            param.requires_grad = False

        was_training = self._classifier.training
        self._classifier.train()
        progress = tqdm(range(epochs), desc="Head", leave=False)
        for _ in progress:
            permutation = torch.randperm(n_samples, device=device)
            epoch_loss = 0.0
            for start in range(0, n_samples, effective_batch):
                batch_indices = permutation[start : start + effective_batch]
                logits = self._classifier(latent_mu[batch_indices])
                targets = y_train_tensor[batch_indices]
                loss = self._classifier.loss(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            progress.set_postfix({"loss": epoch_loss / n_batches})

        if not was_training:
            self._classifier.eval()
        for param, flag in zip(self._encoder.parameters(), encoder_requires_grad):
            param.requires_grad = flag
        for param, flag in zip(self._decoder.parameters(), decoder_requires_grad):
            param.requires_grad = flag

    def _run_joint_finetune(
        self,
        finetune_epochs: int,
        encoder_inputs: Tensor,
        data_tensors: dict[str, dict[str, Tensor]],
        mask_tensors: dict[str, dict[str, Tensor]],
        val_inputs: Tensor,
        val_data_tensors: dict[str, dict[str, Tensor]],
        val_mask_tensors: dict[str, dict[str, Tensor]],
        *,
        batch_size: int,
        lr_scale: float,
        early_stop_patience: int,
        y_train_tensor: Tensor | None,
        y_val_tensor: Tensor | None,
    ) -> None:
        """Fine-tune all modules jointly with early stopping."""

        if finetune_epochs <= 0:
            return

        device = encoder_inputs.device
        encoder_params = list(self._encoder.parameters())
        decoder_params = list(self._decoder.parameters())
        prior_params = self._prior_parameters_for_optimizer()
        param_groups: list[dict[str, Any]] = []
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": self.learning_rate})
        if self._classifier is not None:
            param_groups.append(
                {"params": self._classifier.parameters(), "lr": self.learning_rate}
            )
        scaled_lr = self.learning_rate * lr_scale
        if decoder_params:
            param_groups.append({"params": decoder_params, "lr": scaled_lr})
        if prior_params:
            param_groups.append({"params": prior_params, "lr": scaled_lr})
        if not param_groups:
            return

        optimizer = Adam(param_groups)
        self._encoder.train()
        self._decoder.train()
        if self._classifier is not None:
            self._classifier.train()

        n_samples = encoder_inputs.size(0)
        effective_batch = min(batch_size, n_samples) if n_samples else batch_size
        n_batches = max(1, math.ceil(max(n_samples, 1) / max(effective_batch, 1)))

        best_state: dict[str, Any] | None = None
        best_metrics: dict[str, float] | None = None
        patience_counter = 0
        progress = tqdm(range(finetune_epochs), desc="Joint fine-tune", leave=False)
        for epoch in progress:
            permutation = (
                torch.randperm(n_samples, device=device)
                if n_samples
                else torch.tensor([], device=device, dtype=torch.long)
            )
            epoch_loss = 0.0
            for start in range(0, n_samples, max(effective_batch, 1)):
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
                outputs = self._forward_elbo_batch(
                    batch_input,
                    batch_data,
                    batch_masks,
                    beta_scale=self.beta,
                    temperature=(
                        self._inference_tau
                        if self.behaviour == "unsupervised"
                        else None
                    ),
                )
                loss = outputs["loss"]
                if self._classifier is not None and y_train_tensor is not None:
                    logits = self._classifier(outputs["latent"])
                    targets = y_train_tensor[batch_indices]
                    loss = loss + self._classifier.loss(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

            metrics = self._compute_validation_scores(
                val_inputs,
                val_data_tensors,
                val_mask_tensors,
                batch_size=batch_size,
                temperature=(
                    self._inference_tau if self.behaviour == "unsupervised" else None
                ),
                y_val_tensor=y_val_tensor,
            )
            progress.set_postfix(
                {"loss": epoch_loss / n_batches, "nll": metrics.get("nll")}
            )

            if best_metrics is None or self._is_better_metrics(metrics, best_metrics):
                best_metrics = metrics
                best_state = self._capture_model_state()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > early_stop_patience:
                    break

        if best_state is not None:
            self._restore_model_state(best_state, device)
            self._joint_val_metrics = best_metrics

    def _compute_elbo_on_dataset(
        self,
        encoder_inputs: Tensor,
        data_tensors: dict[str, dict[str, Tensor]],
        mask_tensors: dict[str, dict[str, Tensor]],
        *,
        batch_size: int,
        temperature: float | None,
    ) -> dict[str, float]:
        """Evaluate reconstruction and KL terms on a dataset."""

        n_samples = encoder_inputs.size(0)
        if n_samples == 0:
            return {}
        effective_batch = min(batch_size, n_samples)

        encoder_training = self._encoder.training
        decoder_training = self._decoder.training
        self._encoder.eval()
        self._decoder.eval()

        total_recon = 0.0
        total_cat_kl = 0.0
        total_gauss_kl = 0.0
        with torch.no_grad():
            for start in range(0, n_samples, effective_batch):
                end = start + effective_batch
                batch_input = encoder_inputs[start:end]
                batch_data = {
                    key: {
                        column: tensor[start:end] for column, tensor in tensors.items()
                    }
                    for key, tensors in data_tensors.items()
                }
                batch_masks = {
                    key: {
                        column: tensor[start:end] for column, tensor in tensors.items()
                    }
                    for key, tensors in mask_tensors.items()
                }
                outputs = self._forward_elbo_batch(
                    batch_input,
                    batch_data,
                    batch_masks,
                    beta_scale=self.beta,
                    temperature=temperature,
                )
                total_recon += float(outputs["reconstruction"].sum().item())
                total_cat_kl += float(outputs["categorical_kl"].sum().item())
                total_gauss_kl += float(outputs["gaussian_kl"].sum().item())

        if encoder_training:
            self._encoder.train()
        if decoder_training:
            self._decoder.train()

        nll = ((total_cat_kl + total_gauss_kl) * self.beta - total_recon) / n_samples
        return {
            "nll": float(nll),
            "reconstruction": float(total_recon / n_samples),
            "categorical_kl": float(total_cat_kl / n_samples),
            "gaussian_kl": float(total_gauss_kl / n_samples),
        }

    def _compute_validation_scores(
        self,
        encoder_inputs: Tensor,
        data_tensors: dict[str, dict[str, Tensor]],
        mask_tensors: dict[str, dict[str, Tensor]],
        *,
        batch_size: int,
        temperature: float | None,
        y_val_tensor: Tensor | None,
    ) -> dict[str, float]:
        """Compute validation NLL, Brier score and ECE."""

        metrics = self._compute_elbo_on_dataset(
            encoder_inputs,
            data_tensors,
            mask_tensors,
            batch_size=batch_size,
            temperature=temperature,
        )
        if not metrics:
            return {}
        metrics = dict(metrics)
        metrics.setdefault("brier", float("nan"))
        metrics.setdefault("ece", float("nan"))

        if self._classifier is None or y_val_tensor is None:
            return metrics

        was_training = self._classifier.training
        self._classifier.eval()
        posterior = self._collect_posterior_statistics(
            encoder_inputs,
            batch_size=batch_size,
            temperature=temperature,
        )
        latent_mu = posterior["mean"].to(encoder_inputs.device)
        with torch.no_grad():
            logits = self._classifier(latent_mu)
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        targets = y_val_tensor.detach().cpu().numpy()
        try:
            metrics["brier"] = float(compute_brier(probabilities, targets))
        except ValueError:
            metrics["brier"] = float("nan")
        try:
            metrics["ece"] = float(compute_ece(probabilities, targets))
        except ValueError:
            metrics["ece"] = float("nan")
        if was_training:
            self._classifier.train()
        return metrics

    @staticmethod
    def _is_better_metrics(candidate: dict[str, float], best: dict[str, float]) -> bool:
        """Return ``True`` when ``candidate`` improves upon ``best``."""

        for key in ("nll", "brier", "ece"):
            cand = float(candidate.get(key, float("inf")))
            best_val = float(best.get(key, float("inf")))
            if not math.isfinite(cand):
                cand = float("inf")
            if not math.isfinite(best_val):
                best_val = float("inf")
            if cand < best_val - 1e-8:
                return True
            if cand > best_val + 1e-8:
                return False
        return False

    def _capture_model_state(self) -> dict[str, Any]:
        """Capture trainable module states for early stopping."""

        state: dict[str, Any] = {
            "encoder": self._state_dict_to_cpu(self._encoder),
            "decoder": self._state_dict_to_cpu(self._decoder),
            "classifier": self._state_dict_to_cpu(self._classifier),
            "prior_logits": self._tensor_to_cpu(self._prior_component_logits_tensor()),
            "prior_logvar": self._tensor_to_cpu(self._prior_component_logvar_tensor()),
            "prior_mu": self._tensor_to_cpu(self._prior_component_means_tensor()),
            "inference_tau": float(self._inference_tau),
        }
        if self._prior_mean_layer is not None:
            state["prior_mean_state"] = self._state_dict_to_cpu(self._prior_mean_layer)
        return state

    def _restore_model_state(self, state: dict[str, Any], device: torch.device) -> None:
        """Restore trainable parameters from ``state``."""

        encoder_state = state.get("encoder")
        if encoder_state is not None and self._encoder is not None:
            self._encoder.load_state_dict(encoder_state)
        decoder_state = state.get("decoder")
        if decoder_state is not None and self._decoder is not None:
            self._decoder.load_state_dict(decoder_state)
        classifier_state = state.get("classifier")
        if classifier_state is not None and self._classifier is not None:
            self._classifier.load_state_dict(classifier_state)

        prior_logits = state.get("prior_logits")
        if prior_logits is not None and self._prior_component_logits is not None:
            self._prior_component_logits.data.copy_(prior_logits.to(device))
        prior_logvar = state.get("prior_logvar")
        if prior_logvar is not None and self._prior_component_logvar is not None:
            self._prior_component_logvar.data.copy_(prior_logvar.to(device))
        prior_mu = state.get("prior_mu")
        if prior_mu is not None and self._prior_component_mu is not None:
            self._prior_component_mu.data.copy_(prior_mu.to(device))
        if self._prior_mean_layer is not None and "prior_mean_state" in state:
            self._prior_mean_layer.load_state_dict(state["prior_mean_state"])

        inference_tau = state.get("inference_tau")
        if inference_tau is not None:
            self._inference_tau = float(inference_tau)

    def _cache_training_statistics(
        self,
        stats: dict[str, Tensor],
        train_target_indices: np.ndarray | None,
    ) -> None:
        """Persist posterior summaries for downstream tasks."""

        self._train_latent_mu = (
            stats.get("mean", torch.empty(0, self.latent_dim)).clone().detach()
        )
        self._train_latent_logvar = (
            stats.get("logvar", torch.empty(0, self.latent_dim)).clone().detach()
        )
        self._train_component_logits = (
            stats.get("component_logits", torch.empty(0, self.n_components))
            .clone()
            .detach()
        )
        self._train_component_mu = (
            stats.get(
                "component_mu", torch.empty(0, self.n_components, self.latent_dim)
            ).clone()
        ).detach()
        self._train_component_logvar = (
            stats.get(
                "component_logvar", torch.empty(0, self.n_components, self.latent_dim)
            ).clone()
        ).detach()
        self._train_component_probs = (
            stats.get("probs", torch.empty(0, self.n_components)).clone().detach()
        )
        if train_target_indices is not None:
            self._train_target_indices = np.asarray(
                train_target_indices, dtype=np.int64
            )
        else:
            self._train_target_indices = None

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

    @property
    def device(self) -> torch.device:
        """Expose the computation device used by the estimator."""

        return self._select_device()

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
        pos_data: Dict[str, Tensor] = {}
        count_data: Dict[str, Tensor] = {}
        cat_data: Dict[str, Tensor] = {}
        ordinal_data: Dict[str, Tensor] = {}
        real_masks: Dict[str, Tensor] = {}
        pos_masks: Dict[str, Tensor] = {}
        count_masks: Dict[str, Tensor] = {}
        cat_masks: Dict[str, Tensor] = {}
        ordinal_masks: Dict[str, Tensor] = {}

        feature_layout = {"real": {}, "pos": {}, "count": {}, "cat": {}, "ordinal": {}}

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

        for column in self.schema.positive_features:
            values = pd.to_numeric(X[column], errors="coerce").to_numpy(
                dtype=np.float32
            )
            values = np.nan_to_num(values, nan=0.0).reshape(n_samples, 1)
            tensor = torch.from_numpy(values)
            missing = mask[column].to_numpy().astype(bool)
            observed = (~missing).astype(np.float32).reshape(n_samples, 1)
            mask_tensor = torch.from_numpy(observed)
            pos_data[column] = tensor
            pos_masks[column] = mask_tensor
            value_parts.append(tensor)
            mask_parts.append(mask_tensor)
            feature_layout["pos"][column] = 1

        for column in self.schema.count_features:
            values = pd.to_numeric(X[column], errors="coerce").to_numpy(
                dtype=np.float32
            )
            values = np.nan_to_num(values, nan=0.0).reshape(n_samples, 1)
            tensor = torch.from_numpy(values)
            missing = mask[column].to_numpy().astype(bool)
            observed = (~missing).astype(np.float32).reshape(n_samples, 1)
            mask_tensor = torch.from_numpy(observed)
            count_data[column] = tensor
            count_masks[column] = mask_tensor
            value_parts.append(tensor)
            mask_parts.append(mask_tensor)
            feature_layout["count"][column] = 1

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

        for column in self.schema.ordinal_features:
            series = pd.to_numeric(X[column], errors="coerce")
            n_classes = int(self.schema[column].n_classes or 0)
            thermo = np.zeros((n_samples, n_classes), dtype=np.float32)
            valid_mask = series.notna().to_numpy()
            valid_indices = np.where(valid_mask)[0]
            if valid_indices.size:
                values = series.iloc[valid_indices].astype(int).to_numpy()
                values = np.clip(values, 0, n_classes - 1)
                for row, value in zip(valid_indices, values):
                    thermo[row, : value + 1] = 1.0
            tensor = torch.from_numpy(thermo)
            missing = mask[column].to_numpy().astype(bool)
            observed = (~missing).astype(np.float32).reshape(n_samples, 1)
            mask_tensor = torch.from_numpy(observed)
            ordinal_data[column] = tensor
            ordinal_masks[column] = mask_tensor
            value_parts.append(tensor)
            mask_parts.append(mask_tensor)
            feature_layout["ordinal"][column] = n_classes

        if not value_parts:
            raise ValueError("No supported features present in the training data")

        encoder_inputs = torch.cat(value_parts + mask_parts, dim=1).float()
        data_tensors = {
            "real": real_data,
            "pos": pos_data,
            "count": count_data,
            "cat": cat_data,
            "ordinal": ordinal_data,
        }
        mask_tensors = {
            "real": real_masks,
            "pos": pos_masks,
            "count": count_masks,
            "cat": cat_masks,
            "ordinal": ordinal_masks,
        }
        if update_layout:
            self._feature_layout = feature_layout
        return encoder_inputs, data_tensors, mask_tensors

    @staticmethod
    def _reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample latent variables using the reparameterisation trick."""

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _gather_component_parameters(params: Tensor, indices: Tensor) -> Tensor:
        """Select component-specific parameters from ``params`` using ``indices``."""

        if params.dim() != 3:
            raise ValueError("params must have shape (batch, components, latent)")
        if indices.dim() != 1:
            raise ValueError("indices must be a 1D tensor")
        gather_indices = indices.view(-1, 1, 1).expand(-1, 1, params.size(-1))
        gathered = params.gather(1, gather_indices)
        return gathered.squeeze(1)

    @staticmethod
    def _mixture_posterior_statistics(
        logits: Tensor, mu: Tensor, logvar: Tensor, *, temperature: float | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return mean, log-variance and component probabilities for a mixture."""

        if temperature is not None:
            if temperature <= 0.0:
                raise ValueError("temperature must be positive")
            scale = max(float(temperature), 1e-6)
            scaled_logits = logits / scale
        else:
            scaled_logits = logits
        probs = torch.softmax(scaled_logits, dim=-1)
        weighted_mu = (probs.unsqueeze(-1) * mu).sum(dim=1)
        second_moment = (probs.unsqueeze(-1) * (torch.exp(logvar) + mu.pow(2))).sum(
            dim=1
        )
        var = torch.clamp(second_moment - weighted_mu.pow(2), min=1e-6)
        logvar_mean = torch.log(var)
        return weighted_mu, logvar_mean, probs

    def _draw_latent_samples(
        self,
        n_samples: int,
        *,
        conditional: bool,
        targets: Optional[Iterable[object] | np.ndarray],
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Return latent samples and mixture assignments for decoding."""

        if conditional:
            return self._draw_conditional_latents(n_samples, targets, device)
        latents, component_indices = sampling_utils.sample_mixture_latents(
            self._prior_component_logits_tensor().detach(),
            self._prior_component_means_tensor().detach(),
            self._prior_component_logvar_tensor().detach(),
            n_samples,
            device=device,
        )
        assignments = F.one_hot(
            component_indices.to(torch.long), num_classes=self.n_components
        ).float()
        return latents, assignments

    def _draw_conditional_latents(
        self,
        n_samples: int,
        targets: Optional[Iterable[object] | np.ndarray],
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Sample latent variables and assignments conditioned on class labels."""

        if targets is None:
            raise ValueError("Targets must be provided when conditional=True")

        if (
            self._train_component_mu is None
            or self._train_component_logvar is None
            or self._train_component_probs is None
        ):
            raise RuntimeError(
                "Latent posterior statistics are unavailable for sampling"
            )
        if self._class_to_index is None or self._train_target_indices is None:
            raise RuntimeError(
                "Conditional sampling requires supervised targets from the training data"
            )

        target_array = np.asarray(targets).reshape(-1)
        if target_array.shape[0] != n_samples:
            raise ValueError(
                "y must have length equal to n_samples when conditional=True"
            )

        latents = torch.zeros((n_samples, self.latent_dim), device=device)
        assignments = torch.zeros((n_samples, self.n_components), device=device)
        rng = np.random.default_rng()
        for row, raw_label in enumerate(target_array):
            label = raw_label.item() if isinstance(raw_label, np.generic) else raw_label
            if label not in self._class_to_index:
                raise ValueError(
                    f"Unknown class '{label}' supplied for conditional sampling"
                )
            class_index = self._class_to_index[label]
            candidate_indices = np.where(self._train_target_indices == class_index)[0]
            if candidate_indices.size == 0:
                raise ValueError(
                    f"No training samples available for class '{label}' to condition on"
                )
            selected = int(rng.choice(candidate_indices))
            component_probs = self._train_component_probs[selected].to(device)
            component_dist = Categorical(probs=component_probs)
            component_idx = int(component_dist.sample())
            mu = (
                self._train_component_mu[selected, component_idx]
                .unsqueeze(0)
                .to(device)
            )
            logvar = (
                self._train_component_logvar[selected, component_idx]
                .unsqueeze(0)
                .to(device)
            )
            latent_sample = self._reparameterize(mu, logvar)
            latents[row] = latent_sample.squeeze(0)
            assignments[row, component_idx] = 1.0
        return latents, assignments

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
        for column in self.schema.positive_features:
            stats = self._norm_stats_per_col.get(column, {})
            mean_log = float(stats.get("mean_log", 0.0))
            std_log = float(stats.get("std_log", 1.0))
            values = pd.to_numeric(normalised[column], errors="coerce")
            finite = values[values.notna()]
            if not finite.empty and (finite < -1.0 + 1e-6).any():
                raise ValueError(
                    f"Column '{column}' of type 'pos' must be >= -1 to apply log1p"
                )
            log_values = np.log1p(values)
            normalised[column] = (log_values - mean_log) / max(std_log, 1e-6)
        for column in self.schema.count_features:
            stats = self._norm_stats_per_col.get(column, {})
            offset = float(stats.get("offset", 0.0))
            values = pd.to_numeric(normalised[column], errors="coerce")
            finite = values[values.notna()]
            if not finite.empty and (finite < 0).any():
                raise ValueError(
                    f"Column '{column}' of type 'count' must be non-negative"
                )
            shifted = values + offset
            normalised[column] = np.log(shifted)
        for column in self.schema.categorical_features:
            stats = self._norm_stats_per_col.get(column, {})
            categories = stats.get("categories")
            if categories is not None:
                normalised[column] = pd.Categorical(
                    normalised[column], categories=categories
                )
            else:
                normalised[column] = normalised[column].astype("category")
        for column in self.schema.ordinal_features:
            stats = self._norm_stats_per_col.get(column, {})
            original = normalised[column]
            series = pd.to_numeric(original, errors="coerce")
            n_classes = int(
                stats.get(
                    "n_classes",
                    (
                        self.schema[column].n_classes
                        if self.schema[column].n_classes
                        else 0
                    ),
                )
            )
            if n_classes:
                out_of_range = series.notna() & ~series.between(0, n_classes - 1)
            else:
                out_of_range = pd.Series(False, index=series.index)
            invalid_coercion = original.notna() & series.isna()
            if invalid_coercion.any() or out_of_range.any():
                range_text = (
                    f"[0, {n_classes - 1}]"
                    if n_classes
                    else "the configured ordinal range"
                )
                warnings.warn(
                    (
                        f"Column '{column}' contains ordinal values outside {range_text} "
                        "or non-numeric entries; they will be treated as missing."
                    ),
                    UserWarning,
                    stacklevel=2,
                )
                series[invalid_coercion] = np.nan
                series[out_of_range] = np.nan
            normalised[column] = series
        return normalised

    def _prepare_inference_inputs(self, X: pd.DataFrame) -> Tensor:
        """Return encoder inputs for inference using stored statistics."""

        if self.schema is None:
            raise RuntimeError("Schema must be defined for inference")
        aligned = X.reset_index(drop=True)
        mask = data_utils.build_missing_mask(aligned)
        normalised = self._apply_training_normalization(aligned)
        mask = mask | normalised.isna()
        encoder_inputs, _, _ = self._prepare_training_tensors(
            normalised, mask, update_layout=False
        )
        return encoder_inputs.float()

    def _resolve_attribute(self, attr: str | int) -> tuple[str, ColumnSpec, int]:
        """Return the schema specification for ``attr`` and its positional index."""

        if self.schema is None:
            raise RuntimeError("Schema must be defined to resolve attributes")
        feature_names = list(self.schema.feature_names)
        if not feature_names:
            raise RuntimeError("Schema does not define any features")
        if isinstance(attr, str):
            if attr not in self.schema:
                raise KeyError(f"Attribute '{attr}' not present in the schema")
            name = attr
            index = feature_names.index(name)
        else:
            try:
                index = int(attr)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise TypeError("attr must be a string name or integer index") from exc
            if index < 0 or index >= len(feature_names):
                raise IndexError(
                    "attr index out of range for schema with "
                    f"{len(feature_names)} features"
                )
            name = feature_names[index]
        spec = self.schema[name]
        return name, spec, index

    @staticmethod
    def _feature_bucket(feature_type: str) -> str:
        """Return the decoder bucket key associated with ``feature_type``."""

        mapping = {
            "real": "real",
            "pos": "pos",
            "count": "count",
            "cat": "cat",
            "ordinal": "ordinal",
        }
        if feature_type not in mapping:
            raise ValueError(f"Unsupported feature type '{feature_type}'")
        return mapping[feature_type]

    def _normalize_inputs(
        self,
        X: pd.DataFrame,
        attr_name: str,
        mask: pd.DataFrame | np.ndarray | Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return encoder inputs and attribute tensors with ``attr_name`` masked."""

        if self.schema is None:
            raise RuntimeError("Schema must be defined before normalisation")
        feature_names = list(self.schema.feature_names)
        aligned = X.loc[:, feature_names].copy()
        aligned_reset = aligned.reset_index(drop=True)

        if mask is None:
            mask_frame = data_utils.build_missing_mask(aligned_reset)
        elif isinstance(mask, pd.DataFrame):
            missing_cols = [col for col in feature_names if col not in mask.columns]
            if missing_cols:
                raise KeyError(
                    "Mask dataframe missing required columns: "
                    + ", ".join(missing_cols)
                )
            mask_frame = mask.loc[:, feature_names].astype(bool).reset_index(drop=True)
        else:
            mask_array = np.asarray(mask)
            if mask_array.shape != aligned_reset.shape:
                raise ValueError("mask must have the same shape as the feature matrix")
            mask_frame = pd.DataFrame(mask_array, columns=feature_names).astype(bool)

        mask_frame[attr_name] = True
        aligned_reset[attr_name] = np.nan
        normalised = self._apply_training_normalization(aligned_reset)
        mask_frame = (mask_frame | normalised.isna()).reset_index(drop=True)

        encoder_inputs, data_tensors, mask_tensors = self._prepare_training_tensors(
            normalised, mask_frame, update_layout=False
        )

        feature_type = self.schema[attr_name].type
        bucket = self._feature_bucket(feature_type)
        attr_data = data_tensors[bucket][attr_name].float()
        attr_mask = mask_tensors[bucket][attr_name].float()
        if attr_mask.numel():
            attr_mask.zero_()
        return encoder_inputs.float(), attr_data, attr_mask

    def _posterior_draws(
        self,
        component_logits: Tensor,
        component_mu: Tensor,
        component_logvar: Tensor,
        L: int,
    ) -> tuple[Tensor, Tensor]:
        """Sample mixture assignments and latent variables from the posterior."""

        if L <= 0:
            raise ValueError("L must be a positive integer")
        if component_logits.ndim != 2:
            raise ValueError("component_logits must have shape (batch, components)")
        batch, components = component_logits.shape
        if components != self.n_components:
            raise ValueError(
                "component dimension mismatch between encoder outputs and model"
            )
        categorical = Categorical(logits=component_logits)
        indices = categorical.sample((L,))
        assignments = F.one_hot(indices, num_classes=self.n_components).float()

        mu_expanded = component_mu.unsqueeze(0).expand(L, -1, -1, -1)
        logvar_expanded = component_logvar.unsqueeze(0).expand(L, -1, -1, -1)
        gather_idx = (
            indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.latent_dim)
        )
        selected_mu = mu_expanded.gather(2, gather_idx).squeeze(2)
        selected_logvar = logvar_expanded.gather(2, gather_idx).squeeze(2)
        std = torch.exp(0.5 * selected_logvar)
        eps = torch.randn_like(std)
        z_samples = selected_mu + eps * std
        return assignments, z_samples

    @staticmethod
    def _expand_observation(tensor: Tensor, L: int) -> Tensor:
        """Tile ``tensor`` ``L`` times along the batch dimension."""

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)
        batch = tensor.size(0)
        trailing = tensor.size()[1:]
        expanded = tensor.unsqueeze(0).expand((L, batch, *trailing))
        return expanded.reshape(L * batch, -1)

    def _posterior_predictive_params(
        self,
        encoder_inputs: Tensor,
        attr_name: str,
        attr_type: str,
        attr_data: Tensor,
        attr_mask: Tensor,
        L: int,
    ) -> dict[str, Tensor]:
        """Return decoder parameters for ``attr_name`` under posterior sampling."""

        batch = encoder_inputs.size(0)
        if batch == 0:
            return {"params": {}, "probs": torch.empty(0)}

        component_logits, component_mu, component_logvar = self._encoder(encoder_inputs)
        assignments, z_samples = self._posterior_draws(
            component_logits, component_mu, component_logvar, L
        )
        latents = z_samples.view(L * batch, self.latent_dim)
        assignments_flat = assignments.view(L * batch, self.n_components)
        hidden = self._decoder.backbone(latents)
        y_full = self._decoder.y_projection(hidden)
        y_slice = y_full[:, self._decoder._y_slices[attr_name]]
        data_expanded = self._expand_observation(attr_data, L)
        mask_expanded = self._expand_observation(attr_mask, L)
        stats = self._norm_stats_per_col.get(attr_name, {})
        head = self._decoder.heads[attr_name]
        head_out = head(
            y=y_slice,
            s=assignments_flat,
            x=data_expanded,
            norm_stats=stats,
            mask=mask_expanded,
        )
        params = head_out.get("params", {})
        probs = params.get("probs") if isinstance(params, dict) else None
        return {"params": params, "probs": probs, "L": L, "batch": batch}

    def _draw_predictive_samples(
        self,
        attr_type: str,
        params: dict[str, Tensor],
        L: int,
        batch: int,
        device: torch.device,
    ) -> Tensor:
        """Sample from the attribute decoder given posterior draws."""

        if batch == 0:
            return torch.empty((L, 0), device=device)
        if attr_type == "real":
            mean = params.get("mean")
            var = params.get("var")
            if mean is None or var is None:
                raise RuntimeError("Gaussian head did not provide mean/var")
            samples_flat = dist_utils.sample_gaussian(mean, var)
        elif attr_type == "pos":
            mean_log = params.get("mean_log")
            var_log = params.get("var_log")
            if mean_log is None or var_log is None:
                raise RuntimeError(
                    "Log-normal head did not provide mean/var parameters"
                )
            samples_flat = dist_utils.sample_lognormal(mean_log, var_log)
        elif attr_type == "count":
            rate = params.get("rate")
            if rate is None:
                raise RuntimeError("Poisson head did not provide rate parameters")
            samples_flat = dist_utils.sample_poisson(rate)
        else:
            raise ValueError(f"Unsupported attribute type '{attr_type}' for sampling")
        return samples_flat.view(L, batch, -1).squeeze(-1)

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

    def _ensure_classifier_available(self, caller: str) -> None:
        """Raise an informative error if classifier-dependent APIs are used."""

        if self.behaviour == "unsupervised":
            self._cached_logits = None
            self._cached_probabilities = None
            self._logits_cache_key = None
            self._probability_cache_key = None
            raise RuntimeError(
                f"{caller} is unavailable when behaviour='unsupervised'; this mode focuses "
                "on generative modelling and does not expose classifier outputs."
            )

    def _compute_logits(
        self, X: pd.DataFrame, *, cache_key: str | None = None
    ) -> np.ndarray:
        """Run the classifier head on ``X`` and cache the logits."""

        self._ensure_classifier_available("Logit computation")
        if not self._is_fitted or self._encoder is None or self._classifier is None:
            raise RuntimeError("Model must be fitted before computing logits")
        key = cache_key or self._fingerprint_inputs(X)
        if self._cached_logits is not None and self._logits_cache_key == key:
            return self._cached_logits
        device = self._select_device()
        encoder_inputs = self._prepare_inference_inputs(X).to(device)
        was_encoder_training = self._encoder.training
        was_classifier_training = self._classifier.training
        self._encoder.eval()
        self._classifier.eval()
        with torch.no_grad():
            logits_enc, mu_enc, logvar_enc = self._encoder(encoder_inputs)
            posterior_mean, _, _ = self._mixture_posterior_statistics(
                logits_enc,
                mu_enc,
                logvar_enc,
                temperature=(
                    self._inference_tau if self.behaviour == "unsupervised" else None
                ),
            )
            logits_tensor = self._classifier(posterior_mean)
        if was_encoder_training:
            self._encoder.train()
        if was_classifier_training:
            self._classifier.train()
        logits = logits_tensor.cpu().numpy()
        self._cached_logits = logits
        self._cached_probabilities = None
        self._probability_cache_key = None
        self._logits_cache_key = key
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

    @staticmethod
    def _fingerprint_inputs(frame: pd.DataFrame) -> str:
        """Return a stable fingerprint for ``frame`` used to manage caches."""

        hashed = hash_pandas_object(frame, index=True).to_numpy()
        digest = hashlib.blake2b(hashed.tobytes(), digest_size=16)
        column_signature = "|".join(map(str, frame.columns))
        return (
            f"{frame.shape[0]}x{frame.shape[1]}|{column_signature}|{digest.hexdigest()}"
        )

    def impute(
        self,
        X: pd.DataFrame,
        *,
        only_missing: bool = True,
        assignment_strategy: Literal["soft", "hard", "sample"] | None = None,
    ) -> pd.DataFrame:
        """Fill missing entries in ``X`` using the trained decoder.

        Parameters
        ----------
        X : pandas.DataFrame
            DataFrame containing the features declared in the training schema.
            Missing entries (``NaN``) are imputed using the posterior mean of
            the latent representation followed by a single decoder pass.  The
            method works in both ``behaviour="supervised"`` and
            ``behaviour="unsupervised"`` modes.
        only_missing : bool, default True
            When ``True`` only the originally missing entries are replaced by
            their reconstructions.  When ``False`` the returned dataframe
            contains the full decoder output for all features.
        assignment_strategy : {"soft", "hard", "sample"}, optional
            Strategy used to map posterior mixture responsibilities to decoder
            assignments.  ``"soft"`` (default in the unsupervised branch) uses
            posterior probabilities as continuous weights. ``"hard"`` (default in
            the supervised branch) selects the argmax component, yielding
            sharper but less uncertainty-aware reconstructions. ``"sample"`` draws
            a relaxed Gumbel-Softmax sample which preserves stochasticity while
            still supplying soft weights to the decoder.

        Returns
        -------
        pandas.DataFrame
            Copy of ``X`` where the missing entries have been replaced by the
            decoded reconstructions.  Observed values are preserved.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from suave.model import SUAVE
        >>> from suave.types import Schema
        >>> data = pd.DataFrame({"real": [1.0, np.nan, 3.0], "cat": [0, 1, 0]})
        >>> schema = Schema({"real": {"type": "real"}, "cat": {"type": "cat", "n_classes": 2}})
        >>> model = SUAVE(schema=schema, behaviour="unsupervised")
        >>> _ = model.fit(data, epochs=1, batch_size=3)
        >>> imputed = model.impute(data)
        >>> imputed.isna().sum().sum()
        0
        """

        if not self._is_fitted or self._encoder is None or self._decoder is None:
            raise RuntimeError("Model must be fitted before calling impute")
        if self.schema is None:
            raise RuntimeError("Schema must be defined for imputation")

        missing_columns = [
            column for column in self.schema.feature_names if column not in X.columns
        ]
        if missing_columns:
            raise KeyError(
                "Input is missing required columns: " + ", ".join(missing_columns)
            )

        original_index = X.index
        aligned = X.loc[:, self.schema.feature_names].copy()
        aligned_reset = aligned.reset_index(drop=True)

        base_mask = data_utils.build_missing_mask(aligned_reset)
        normalised = self._apply_training_normalization(aligned_reset)
        combined_mask = (base_mask | normalised.isna()).reset_index(drop=True)

        _, data_tensors, mask_tensors = self._prepare_training_tensors(
            normalised, combined_mask, update_layout=False
        )

        device = self._select_device()
        encoder_inputs = self._prepare_inference_inputs(aligned_reset).to(device)
        for feature_type in data_tensors:
            for column in data_tensors[feature_type]:
                data_tensors[feature_type][column] = data_tensors[feature_type][
                    column
                ].to(device)
                mask_tensors[feature_type][column] = mask_tensors[feature_type][
                    column
                ].to(device)

        if assignment_strategy is None:
            assignment_strategy = "hard" if self.behaviour == "supervised" else "soft"
        strategy_normalised = assignment_strategy.lower()
        valid_strategies = {"soft", "hard", "sample"}
        if strategy_normalised not in valid_strategies:
            raise ValueError(
                "assignment_strategy must be one of 'soft', 'hard' or 'sample'"
            )

        with torch.no_grad():
            encoder_state = self._encoder.training
            decoder_state = self._decoder.training
            self._encoder.eval()
            self._decoder.eval()
            logits_enc, mu_enc, logvar_enc = self._encoder(encoder_inputs)
            posterior_mean, _, posterior_probs = self._mixture_posterior_statistics(
                logits_enc,
                mu_enc,
                logvar_enc,
                temperature=(
                    self._inference_tau if self.behaviour == "unsupervised" else None
                ),
            )
            if strategy_normalised == "hard":
                component_indices = posterior_probs.argmax(dim=-1)
                assignments = F.one_hot(
                    component_indices, num_classes=self.n_components
                ).float()
                selected_mu = self._gather_component_parameters(
                    mu_enc, component_indices
                )
            elif strategy_normalised == "soft":
                assignments = posterior_probs
                selected_mu = posterior_mean
            else:  # sample
                if self.behaviour == "unsupervised":
                    tau = max(float(self._inference_tau), 1e-6)
                else:
                    tau = max(float(self.gumbel_temperature), 1e-6)
                assignments = F.gumbel_softmax(logits_enc, tau=tau, hard=False, dim=-1)
                selected_mu = (assignments.unsqueeze(-1) * mu_enc).sum(dim=1)
            decoder_out = self._decoder(
                selected_mu,
                assignments,
                data_tensors,
                self._norm_stats_per_col,
                mask_tensors,
            )
            if encoder_state:
                self._encoder.train()
            if decoder_state:
                self._decoder.train()

        reconstruction = sampling_utils.decoder_outputs_to_frame(
            decoder_out["per_feature"], self.schema, self._norm_stats_per_col
        )

        if only_missing:
            original_aligned = aligned_reset.copy()
            imputed = reconstruction.copy()
            for column in reconstruction.columns:
                observed_mask = ~combined_mask[column]
                if not observed_mask.any():
                    continue
                column_data = reconstruction[column]
                if isinstance(column_data.dtype, CategoricalDtype):
                    aligned_values = original_aligned.loc[observed_mask, column]
                    aligned_values = aligned_values.astype(column_data.dtype)
                    imputed.loc[observed_mask, column] = aligned_values
                else:
                    aligned_values = original_aligned.loc[observed_mask, column]
                    target_dtype = imputed[column].dtype
                    try:
                        aligned_values = aligned_values.astype(target_dtype, copy=False)
                    except (TypeError, ValueError):
                        aligned_values = aligned_values.to_numpy()
                    imputed.loc[observed_mask, column] = aligned_values
        else:
            imputed = reconstruction

        imputed.index = original_index
        extra_columns = [
            column for column in X.columns if column not in self.schema.feature_names
        ]
        if extra_columns:
            extras = X.loc[:, extra_columns].copy()
            extras.index = original_index
            imputed = pd.concat([imputed, extras], axis=1)
        return imputed.loc[:, X.columns]

    def _infer_latent_statistics(self, X: pd.DataFrame) -> tuple[Tensor, Tensor]:
        """Return posterior parameters for ``X`` using the trained encoder."""

        if not self._is_fitted or self._encoder is None:
            raise RuntimeError("Model must be fitted before encoding data")
        device = self._select_device()
        encoder_inputs = self._prepare_inference_inputs(X).to(device)
        was_training = self._encoder.training
        self._encoder.eval()
        with torch.no_grad():
            logits_enc, mu_enc, logvar_enc = self._encoder(encoder_inputs)
        if was_training:
            self._encoder.train()
        posterior_mean, posterior_logvar, _ = self._mixture_posterior_statistics(
            logits_enc,
            mu_enc,
            logvar_enc,
            temperature=(
                self._inference_tau if self.behaviour == "unsupervised" else None
            ),
        )
        return posterior_mean, posterior_logvar

    # ------------------------------------------------------------------
    # Prediction utilities
    # ------------------------------------------------------------------
    def predict_proba(
        self,
        X: pd.DataFrame,
        attr: str | int | None = None,
        *,
        mask: pd.DataFrame | np.ndarray | Tensor | None = None,
        L: int = 100,
    ) -> np.ndarray | Tensor:
        r"""Return posterior predictive probabilities.

        When ``attr`` is ``None`` the method preserves the previous behaviour
        and returns the calibrated classifier probabilities for ``X``.  When
        ``attr`` identifies a categorical or ordinal schema attribute the
        method estimates the Monte Carlo posterior predictive distribution
        :math:`p(x_{attr} \mid x^o)` by masking the target column and decoding
        samples from :math:`q(s, z \mid x^o)`.  In
        ``behaviour="unsupervised"`` mode the classifier head is disabled, so
        ``attr`` must be provided and the method only operates in the
        generative regime described above.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix whose columns match the training schema.
        attr : str or int, optional
            Name or positional index of a categorical or ordinal attribute to
            evaluate.  When omitted, classifier probabilities for the target
            label are returned.
        mask : pandas.DataFrame or numpy.ndarray or torch.Tensor, optional
            Boolean mask that marks observed values as ``False`` and missing
            entries as ``True``.  When ``attr`` is provided the corresponding
            column is always treated as missing while decoding.  Provide
            ``mask`` when ``X`` has been imputed or otherwise lacks explicit
            ``NaN`` markers so that the original missingness pattern (for
            example the mask returned by :func:`suave.data.build_missing_mask`)
            can be respected during decoding.  If ``mask`` is omitted the
            method infers missingness directly from ``X``.
        L : int, default 100
            Number of Monte Carlo samples used to approximate the posterior
            predictive distribution when ``attr`` is specified.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Calibrated classifier probabilities with shape ``(n_samples,
            n_classes)`` when ``attr`` is ``None``.  Otherwise a
            ``torch.FloatTensor`` with shape ``(n_samples, n_attr_classes)``
            containing posterior predictive probabilities for the requested
            attribute.

        Raises
        ------
        RuntimeError
            If the estimator or classifier head has not been fitted when
            ``attr`` is ``None``.
        ValueError
            If ``attr`` does not reference a categorical or ordinal feature, or
            when ``L`` is not a positive integer.

        See Also
        --------
        SUAVE.predict : Convert probabilities into deterministic predictions.


        Examples
        --------
        >>> proba = model.predict_proba(X)
        >>> proba.sum(axis=1)
        array([1., 1.])
        >>> gender_probs = model.predict_proba(X, attr="gender")
        >>> torch.allclose(gender_probs.sum(dim=1), torch.ones(len(X)))
        True
        >>> mask = suave.data.build_missing_mask(X_raw)
        >>> model.predict_proba(X_imputed, attr="gender", mask=mask)
        """

        if attr is None:
            self._ensure_classifier_available("predict_proba")
            if not self._is_fitted or self._classes is None:
                raise RuntimeError("Model must be fitted before calling predict_proba")
            cache_key = self._fingerprint_inputs(X)
            if (
                self._cached_probabilities is not None
                and self._probability_cache_key == cache_key
            ):
                return self._cached_probabilities
            logits = self._compute_logits(X, cache_key=cache_key)
            if self._cached_logits is None:
                raise RuntimeError("Logits cache was not populated")
            if self._is_calibrated:
                logits = self._temperature_scaler.transform(logits)
            probabilities = self._logits_to_probabilities(logits)
            self._cached_probabilities = probabilities
            self._probability_cache_key = cache_key
            return probabilities

        if not self._is_fitted or self._encoder is None or self._decoder is None:
            raise RuntimeError(
                "Model must be fitted before requesting attribute probabilities"
            )
        attr_name, spec, _ = self._resolve_attribute(attr)
        if spec.type not in {"cat", "ordinal"}:
            raise ValueError(
                "predict_proba is only defined for categorical or ordinal attributes"
            )
        if L <= 0:
            raise ValueError("L must be a positive integer")

        encoder_inputs, attr_data, attr_mask = self._normalize_inputs(
            X, attr_name, mask
        )
        batch = encoder_inputs.size(0)
        n_classes = int(spec.n_classes or attr_data.size(-1))
        if batch == 0:
            return torch.empty((0, n_classes), dtype=torch.float32)

        device = self.device
        encoder_inputs = encoder_inputs.to(device)
        attr_data = attr_data.to(device)
        attr_mask = attr_mask.to(device)

        encoder_state = self._encoder.training
        decoder_state = self._decoder.training
        self._encoder.eval()
        self._decoder.eval()
        with torch.no_grad():
            info = self._posterior_predictive_params(
                encoder_inputs, attr_name, spec.type, attr_data, attr_mask, L
            )
        if encoder_state:
            self._encoder.train()
        if decoder_state:
            self._decoder.train()

        probs = info.get("probs")
        if probs is None:
            params = info.get("params", {})
            logits = params.get("logits")
            if logits is None:
                raise RuntimeError(
                    "Decoder head did not expose logits for attribute probabilities"
                )
            probs = torch.softmax(logits, dim=-1)
        probs = probs.view(L, batch, -1).mean(dim=0)
        probs = torch.clamp(probs, min=1e-8)
        normaliser = probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        probs = probs / normaliser
        return probs.detach().cpu()

    def predict_confidence_interval(
        self,
        X: pd.DataFrame,
        attr: str | int,
        *,
        mask: pd.DataFrame | np.ndarray | Tensor | None = None,
        L: int = 1000,
        ci: float = 0.95,
        statistic: Literal["mean", "median", None] | None = None,
        return_samples: bool = False,
    ) -> dict[str, Tensor]:
        r"""Return Monte Carlo posterior predictive statistics for ``attr``.

        The attribute is treated as missing and the method samples ``L`` draws
        from :math:`q(s, z \mid x^o)` followed by the decoder likelihood.  The
        returned dictionary includes the requested point estimate, percentile
        confidence interval bounds and the sample standard deviation.  When
        ``return_samples`` is ``True`` the raw predictive samples are also
        returned with shape ``(n_samples, L)``.  ``mask`` can be supplied to
        preserve externally tracked missingness patterns that are not encoded
        as ``NaN`` values in ``X``.

        Parameters
        ----------
        X:
            Input features with shape ``(n_samples, n_features)``.
        attr:
            Attribute name or positional index referring to a real, positive or
            count feature.
        mask:
            Optional boolean mask marking missing entries (``True`` for
            missing).  The specified attribute is always treated as missing
            regardless of ``mask``.  Pass the mask used during training (for
            example from :func:`suave.data.build_missing_mask`) when ``X`` has
            been imputed so that the decoder can preserve the original
            missingness pattern while conditioning on ``x^o``.
        L:
            Number of Monte Carlo samples used to approximate the predictive
            distribution.  Must be positive.
        ci:
            Central confidence interval mass.  Must lie strictly between ``0``
            and ``1``.
        statistic:
            Optional override for the point estimate.  When ``None`` the mean
            is used for ``real``/``count`` attributes and the median for
            ``pos`` (log-normal) attributes.
        return_samples:
            When ``True`` the returned dictionary includes the raw predictive
            samples in ``result["samples"]``.

        Returns
        -------
        dict
            Dictionary with keys ``point``, ``lower`` and ``upper`` (and
            optionally ``std`` and ``samples``) containing :class:`torch.Tensor`
            objects of shape ``(n_samples,)`` (``samples`` has shape
            ``(n_samples, L)``).

        Examples
        --------
        >>> stats = model.predict_confidence_interval(X, "age", L=256)
        >>> stats["lower"].shape
        torch.Size([len(X)])
        >>> mask = suave.data.build_missing_mask(X_raw)
        >>> model.predict_confidence_interval(X_imputed, "age", mask=mask)
        """

        if not self._is_fitted or self._encoder is None or self._decoder is None:
            raise RuntimeError(
                "Model must be fitted before computing confidence intervals"
            )
        attr_name, spec, _ = self._resolve_attribute(attr)
        if spec.type in {"cat", "ordinal"}:
            raise ValueError(
                "predict_confidence_interval is only available for real, pos or count attributes"
            )
        if L <= 0:
            raise ValueError("L must be a positive integer")
        if not (0.0 < ci < 1.0):
            raise ValueError("ci must lie strictly between 0 and 1")
        if statistic not in {None, "mean", "median"}:
            raise ValueError("statistic must be one of None, 'mean' or 'median'")

        encoder_inputs, attr_data, attr_mask = self._normalize_inputs(
            X, attr_name, mask
        )
        batch = encoder_inputs.size(0)
        device = self.device
        if batch == 0:
            empty = torch.empty(0, dtype=torch.float32)
            result: dict[str, Tensor] = {
                "point": empty,
                "lower": empty,
                "upper": empty,
            }
            if return_samples:
                result["samples"] = empty.view(0, 0)
            return result

        encoder_inputs = encoder_inputs.to(device)
        attr_data = attr_data.to(device)
        attr_mask = attr_mask.to(device)

        encoder_state = self._encoder.training
        decoder_state = self._decoder.training
        self._encoder.eval()
        self._decoder.eval()
        with torch.no_grad():
            info = self._posterior_predictive_params(
                encoder_inputs, attr_name, spec.type, attr_data, attr_mask, L
            )
            params = info.get("params", {})
        if encoder_state:
            self._encoder.train()
        if decoder_state:
            self._decoder.train()

        if not params:
            raise RuntimeError(
                "Decoder did not return parameters for confidence interval"
            )

        samples = self._draw_predictive_samples(spec.type, params, L, batch, device)
        samples = samples.view(L, batch)
        lower_q = (1.0 - ci) / 2.0
        upper_q = 1.0 - lower_q
        lower = torch.quantile(samples, lower_q, dim=0)
        upper = torch.quantile(samples, upper_q, dim=0)
        if statistic == "mean" or (
            statistic is None and spec.type in {"real", "count"}
        ):
            point = samples.mean(dim=0)
        else:
            point = samples.median(dim=0).values
        std = samples.std(dim=0, unbiased=False)

        result = {
            "point": point.detach().cpu(),
            "lower": lower.detach().cpu(),
            "upper": upper.detach().cpu(),
            "std": std.detach().cpu(),
        }
        if return_samples:
            result["samples"] = samples.transpose(0, 1).detach().cpu()
        return result

    def predict(
        self,
        X: pd.DataFrame,
        attr: str | int | None = None,
        *,
        mask: pd.DataFrame | np.ndarray | Tensor | None = None,
        L: int = 50,
        mode: Literal["point", "sample"] = "point",
    ) -> np.ndarray | Tensor:
        """Return class labels or attribute predictions for ``X``.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix with the same columns used during training.
        attr : str or int, optional
            Name or positional index of the attribute to infer.  When omitted,
            class labels from the supervised head are returned.  In
            ``behaviour="unsupervised"`` mode the classifier head is not
            available, so ``attr`` must be supplied and only attribute-level
            predictions are produced.
        mask : pandas.DataFrame or numpy.ndarray or torch.Tensor, optional
            Boolean mask indicating observed (``False``) and missing (``True``)
            entries.  Used when requesting attribute predictions so the
            original missingness pattern can be respected.  Supply the same
            mask used during training (for example from
            :func:`suave.data.build_missing_mask`) when ``X`` no longer contains
            ``NaN`` placeholders for missing inputs; otherwise the method infers
            missingness directly from ``X``.
        L : int, default 50
            Number of Monte Carlo samples drawn when ``mode='sample'`` or when
            summarising posterior predictive statistics for non-class targets.
        mode : {"point", "sample"}, default "point"
            Controls whether the deterministic summary (mean/median) or raw
            posterior predictive samples are returned for attribute-level
            predictions.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            One-dimensional array of class labels when ``attr`` is ``None``.
            Otherwise a tensor containing either posterior predictive samples
            or point estimates for the requested attribute.

        Raises
        ------
        RuntimeError
            If the model is not fitted or the classifier head is unavailable
            when requesting label predictions.
        ValueError
            If ``mode`` is invalid or the requested attribute is unsupported by
            the decoder.

        Examples
        --------
        >>> labels = model.predict(X)
        >>> labels.shape
        (len(X),)
        >>> glucose = model.predict(X, attr="glucose")
        >>> glucose_samples = model.predict(X, attr="glucose", mode="sample")
        >>> mask = suave.data.build_missing_mask(X_raw)
        >>> model.predict(X_imputed, attr="glucose", mask=mask)

        """

        if attr is None:
            self._ensure_classifier_available("predict")
            probabilities = self.predict_proba(X)
            indices = probabilities.argmax(axis=1)
            return self._classes[indices]

        attr_name, spec, _ = self._resolve_attribute(attr)
        if mode not in {"point", "sample"}:
            raise ValueError("mode must be either 'point' or 'sample'")

        if spec.type in {"cat", "ordinal"}:
            probs = self.predict_proba(X, attr=attr_name, mask=mask, L=L)
            if probs.numel() == 0:
                return probs.new_empty((0,), dtype=torch.long)
            if mode == "point":
                return probs.argmax(dim=1)
            categorical = Categorical(probs=probs)
            return categorical.sample()

        stats = self.predict_confidence_interval(
            X,
            attr=attr_name,
            mask=mask,
            L=max(1, L),
            statistic=None,
            return_samples=(mode == "sample"),
        )
        point = stats["point"]
        if mode == "point":
            return point
        samples = stats.get("samples")
        if samples is None or samples.size(1) == 0:
            return point
        indices = torch.randint(
            0, samples.size(1), (samples.size(0),), device=samples.device
        )
        gathered = samples.gather(1, indices.view(-1, 1)).squeeze(1)
        return gathered

    # ------------------------------------------------------------------
    # Calibration utilities
    # ------------------------------------------------------------------
    def calibrate(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "SUAVE":
        """Fit the temperature scaler using logits from ``X``.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix used to compute logits for calibration.
        y : pandas.Series or numpy.ndarray
            True target labels aligned with ``X``. The array must use the same
            label encoding observed during :meth:`fit`.

        Returns
        -------
        SUAVE
            The calibrated estimator (``self``).

        Raises
        ------
        RuntimeError
            If the model has not been fitted or the classifier head is missing.
        ValueError
            If ``X`` and ``y`` do not contain the same number of samples.

        Examples
        --------
        >>> _ = model.fit(X_train, y_train)
        >>> model.calibrate(X_val, y_val)
        SUAVE(...)
        """

        self._ensure_classifier_available("calibrate")
        if not self._is_fitted or self._classes is None:
            raise RuntimeError("Fit must be called before calibrate")
        if len(X) != len(y):
            raise ValueError("X and y must have matching first dimensions")
        cache_key = self._fingerprint_inputs(X)
        logits = self._compute_logits(X, cache_key=cache_key)
        target_indices = self._map_targets_to_indices(y)
        if logits.shape[0] != target_indices.shape[0]:
            raise ValueError("Calibration logits and targets must have matching rows")
        self._temperature_scaler.fit(logits, target_indices)
        self._temperature_scaler_state = self._temperature_scaler.state_dict()
        self._is_calibrated = True
        calibrated_logits = self._temperature_scaler.transform(logits)
        probabilities = self._logits_to_probabilities(calibrated_logits)
        self._cached_probabilities = probabilities
        self._probability_cache_key = cache_key
        return self

    # ------------------------------------------------------------------
    # Latent utilities and sampling
    # ------------------------------------------------------------------
    def encode(
        self, X: pd.DataFrame, *, return_components: bool = False
    ) -> np.ndarray | Dict[str, np.ndarray]:
        """Return posterior statistics of the latent representation.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features with shape ``(n_samples, n_features)``.
        return_components : bool, default False
            When ``True`` also return mixture assignment information and
            component-specific statistics.

        Returns
        -------
        numpy.ndarray or dict
            ``numpy.ndarray`` containing the latent posterior means when
            :paramref:`return_components` is ``False``.  Otherwise a dictionary
            with keys ``"mean"``, ``"assignments"``, ``"component_mu"`` and
            ``"component_logvar"``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

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
            empty_mean = np.empty((0, self.latent_dim), dtype=np.float32)
            if not return_components:
                return empty_mean
            empty_assignments = np.empty((0, self.n_components), dtype=np.float32)
            empty_mu = np.empty((0, self.latent_dim), dtype=np.float32)
            empty_logvar = np.empty((0, self.latent_dim), dtype=np.float32)
            return {
                "mean": empty_mean,
                "assignments": empty_assignments,
                "component_mu": empty_mu,
                "component_logvar": empty_logvar,
            }

        effective_batch = min(self.batch_size, n_samples)
        if effective_batch <= 0:
            effective_batch = n_samples

        latent_batches: list[Tensor] = []
        assignments_batches: list[Tensor] = []
        selected_mu_batches: list[Tensor] = []
        selected_logvar_batches: list[Tensor] = []
        was_training = self._encoder.training
        self._encoder.eval()
        with torch.no_grad():
            for start in range(0, n_samples, effective_batch):
                end = min(start + effective_batch, n_samples)
                batch_inputs = encoder_inputs[start:end]
                logits_enc, mu_enc, logvar_enc = self._encoder(batch_inputs)
                posterior_mean, _, _ = self._mixture_posterior_statistics(
                    logits_enc,
                    mu_enc,
                    logvar_enc,
                    temperature=(
                        self._inference_tau
                        if self.behaviour == "unsupervised"
                        else None
                    ),
                )
                latent_batches.append(posterior_mean.cpu())
                if return_components:
                    posterior_probs = torch.softmax(logits_enc, dim=-1)
                    component_indices = posterior_probs.argmax(dim=-1)
                    assignments = F.one_hot(
                        component_indices, num_classes=self.n_components
                    ).float()
                    selected_mu = self._gather_component_parameters(
                        mu_enc, component_indices
                    )
                    selected_logvar = self._gather_component_parameters(
                        logvar_enc, component_indices
                    )
                    assignments_batches.append(assignments.cpu())
                    selected_mu_batches.append(selected_mu.cpu())
                    selected_logvar_batches.append(selected_logvar.cpu())
        if was_training:
            self._encoder.train()

        posterior_mean_np = torch.cat(latent_batches, dim=0).numpy()
        if not return_components:
            return posterior_mean_np
        assignments_np = torch.cat(assignments_batches, dim=0).numpy()
        mu_np = torch.cat(selected_mu_batches, dim=0).numpy()
        logvar_np = torch.cat(selected_logvar_batches, dim=0).numpy()
        return {
            "mean": posterior_mean_np,
            "assignments": assignments_np,
            "component_mu": mu_np,
            "component_logvar": logvar_np,
        }

    def sample(
        self, n_samples: int, conditional: bool = False, y: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Generate samples by decoding latent variables through the learned model.

        Parameters
        ----------
        n_samples : int
            Number of synthetic rows to generate.
        conditional : bool, default False
            When ``True`` the latent variables are drawn from posterior
            approximations of the training data whose labels match ``y``.
        y : numpy.ndarray, optional
            Sequence of class labels used when :paramref:`conditional` is
            ``True``.  The array must have length ``n_samples`` and contain
            values observed during :meth:`fit`.

        Returns
        -------
        pandas.DataFrame
            DataFrame with shape ``(n_samples, n_features)`` whose columns match
            the training schema.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If conditional sampling is requested without valid labels.

        Examples
        --------
        >>> samples = model.sample(5)
        >>> samples.shape
        (5, len(model.schema.feature_names))
        """

        if not self._is_fitted or self._decoder is None:
            raise RuntimeError("Model must be fitted before sampling")
        if self.schema is None:
            raise RuntimeError("Schema is required to generate samples")
        if n_samples <= 0:
            return pd.DataFrame(columns=list(self.schema.feature_names))

        device = self._select_device()
        latents, assignments = self._draw_latent_samples(
            n_samples, conditional=conditional, targets=y, device=device
        )
        data_tensors, mask_tensors = sampling_utils.build_placeholder_batches(
            self._feature_layout, n_samples, device=device
        )
        decoder_training_state = self._decoder.training
        self._decoder.eval()
        with torch.no_grad():
            decoder_out = self._decoder(
                latents,
                assignments,
                data_tensors,
                self._norm_stats_per_col,
                mask_tensors,
            )
        if decoder_training_state:
            self._decoder.train()

        samples = sampling_utils.decoder_outputs_to_frame(
            decoder_out["per_feature"], self.schema, self._norm_stats_per_col
        )
        return samples

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _state_dict_to_cpu(module: Module | None) -> OrderedDict[str, Tensor] | None:
        """Return a CPU copy of ``module``'s state dict."""

        if module is None:
            return None
        state_dict = module.state_dict()
        return OrderedDict(
            (key, value.detach().cpu() if isinstance(value, Tensor) else value)
            for key, value in state_dict.items()
        )

    @staticmethod
    def _tensor_to_cpu(tensor: Tensor | None) -> Tensor | None:
        """Detach ``tensor`` to CPU if present."""

        if tensor is None:
            return None
        return tensor.detach().cpu()

    def save(self, path: str | Path) -> Path:
        """Serialise minimal model state to ``path``.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file where the torch archive will be written.

        Returns
        -------
        pathlib.Path
            Path to the written archive.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        Examples
        --------
        >>> output = model.save("suave_model.pt")
        >>> output.exists()
        True
        """

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        path = Path(path)
        metadata: dict[str, Any] = {
            "schema": self.schema.to_dict() if self.schema else None,
            "behaviour": self.behaviour,
            "latent_dim": self.latent_dim,
            "n_components": self.n_components,
            "hidden_dims": list(self.hidden_dims),
            "head_hidden_dims": list(self.head_hidden_dims),
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "kl_warmup_epochs": self.kl_warmup_epochs,
            "val_split": self.val_split,
            "stratify": self.stratify,
            "random_state": self.random_state,
            "tau_start": self._tau_start,
            "tau_min": self._tau_min,
            "tau_decay": self._tau_decay,
            "inference_tau": self._inference_tau,
            "warmup_epochs": self.warmup_epochs,
            "head_epochs": self.head_epochs,
            "finetune_epochs": self.finetune_epochs,
            "joint_decoder_lr_scale": self.joint_decoder_lr_scale,
            "early_stop_patience": self.early_stop_patience,
            "auto_configured": {
                key: bool(value) for key, value in self._auto_configured.items()
            },
            "heuristic_overrides": {
                key: bool(value) for key, value in self._auto_configured.items()
            },
            "auto_hyperparameters": serialise_heuristic_hyperparameters(
                self.auto_hyperparameters_
            ),
        }
        classifier_state: dict[str, Any] | None = None
        if self._classifier is not None:
            class_weight = None
            use_weight = bool(getattr(self._classifier, "_use_weight", False))
            if use_weight:
                class_weight = self._classifier.class_weight.detach().cpu()
            classifier_state = {
                "state_dict": self._state_dict_to_cpu(self._classifier),
                "use_weight": use_weight,
                "class_weight": class_weight,
            }
        prior_state: dict[str, Any] = {
            "logits": self._prior_component_logits_tensor().detach().cpu(),
            "logvar": self._prior_component_logvar_tensor().detach().cpu(),
            "mu": self._prior_component_means_tensor().detach().cpu(),
        }
        if self.behaviour == "unsupervised":
            if self._prior_mean_layer is None:
                raise RuntimeError("Prior mean layer is not initialised")
            prior_state["mean_state_dict"] = self._state_dict_to_cpu(
                self._prior_mean_layer
            )
        modules = {
            "encoder": self._state_dict_to_cpu(self._encoder),
            "decoder": self._state_dict_to_cpu(self._decoder),
            "classifier": classifier_state,
            "temperature_scaler": self._temperature_scaler.state_dict(),
            "prior": prior_state,
        }
        artefacts: dict[str, Any] = {
            "classes": self._classes,
            "class_to_index": self._class_to_index,
            "normalization": self._norm_stats_per_col,
            "feature_layout": self._feature_layout,
            "temperature_scaler_state": self._temperature_scaler_state,
            "is_calibrated": self._is_calibrated,
            "train_latent_mu": self._tensor_to_cpu(self._train_latent_mu),
            "train_latent_logvar": self._tensor_to_cpu(self._train_latent_logvar),
            "train_component_logits": self._tensor_to_cpu(self._train_component_logits),
            "train_component_mu": self._tensor_to_cpu(self._train_component_mu),
            "train_component_logvar": self._tensor_to_cpu(self._train_component_logvar),
            "train_component_probs": self._tensor_to_cpu(self._train_component_probs),
            "train_target_indices": self._train_target_indices,
            "cached_logits": self._cached_logits,
            "cached_probabilities": self._cached_probabilities,
            "logits_cache_key": self._logits_cache_key,
            "probability_cache_key": self._probability_cache_key,
            "warmup_val_history": self._warmup_val_history,
            "joint_val_metrics": self._joint_val_metrics,
        }
        payload = {
            "metadata": metadata,
            "modules": modules,
            "artefacts": artefacts,
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SUAVE":
        """Load a model saved with :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            Location of the serialised model archive.

        Returns
        -------
        SUAVE
            Fully reconstructed estimator ready for inference.

        Raises
        ------
        ValueError
            If the archive format is unrecognised.

        Examples
        --------
        >>> restored = SUAVE.load("suave_model.pt")
        >>> isinstance(restored, SUAVE)
        True
        """

        path = Path(path)
        try:
            payload = torch.load(path, map_location="cpu")
        except (RuntimeError, pickle.UnpicklingError, EOFError):
            payload = None
        if isinstance(payload, dict):
            if "metadata" in payload:
                return cls._load_from_payload(payload)
            legacy_keys = {
                "schema",
                "prior",
                "behaviour",
                "classes",
                "artefacts",
                "modules",
            }
            if legacy_keys.intersection(payload.keys()):
                return cls._load_from_legacy_json(payload)
        if payload is not None:
            raise ValueError("Unexpected model archive format")
        data = json.loads(path.read_text())
        return cls._load_from_legacy_json(data)

    @classmethod
    def _load_from_payload(cls, payload: dict[str, Any]) -> "SUAVE":
        """Instantiate ``SUAVE`` from a dictionary produced by :meth:`save`."""

        metadata: dict[str, Any] = payload.get("metadata", {})
        modules: dict[str, Any] = payload.get("modules", {})
        artefacts: dict[str, Any] = payload.get("artefacts", {})
        schema_dict = metadata.get("schema") or {}
        schema = Schema(schema_dict) if schema_dict else None
        behaviour = _normalise_behaviour(str(metadata.get("behaviour", "supervised")))
        metadata["behaviour"] = behaviour
        init_kwargs: dict[str, Any] = {}
        for key in (
            "latent_dim",
            "n_components",
            "hidden_dims",
            "head_hidden_dims",
            "dropout",
            "learning_rate",
            "batch_size",
            "kl_warmup_epochs",
            "val_split",
            "stratify",
            "random_state",
            "tau_start",
            "tau_min",
            "tau_decay",
            "warmup_epochs",
            "head_epochs",
            "finetune_epochs",
            "joint_decoder_lr_scale",
            "early_stop_patience",
        ):
            if key in metadata and metadata[key] is not None:
                value = metadata[key]
                if key == "hidden_dims":
                    value = tuple(int(v) for v in value)
                elif key == "head_hidden_dims":
                    value = tuple(int(v) for v in value)
                elif key in {
                    "tau_start",
                    "tau_min",
                    "tau_decay",
                    "joint_decoder_lr_scale",
                }:
                    value = float(value)
                elif key in {
                    "latent_dim",
                    "n_components",
                    "batch_size",
                    "kl_warmup_epochs",
                    "warmup_epochs",
                    "head_epochs",
                    "finetune_epochs",
                    "early_stop_patience",
                }:
                    value = int(value)
                init_kwargs[key] = value
        model = cls(schema=schema, behaviour=behaviour, **init_kwargs)

        auto_configured = metadata.get("heuristic_overrides")
        if not isinstance(auto_configured, dict):
            auto_configured = metadata.get("auto_configured")
        if isinstance(auto_configured, dict):
            model._auto_configured = {
                config_key: bool(config_value)
                for config_key, config_value in auto_configured.items()
            }
        auto_hparams = metadata.get("auto_hyperparameters")
        if isinstance(auto_hparams, dict):
            parsed = parse_heuristic_hyperparameters(auto_hparams)
            model.auto_hyperparameters_ = parsed or None
        else:
            model.auto_hyperparameters_ = None

        inference_tau = metadata.get("inference_tau")
        if inference_tau is not None:
            model._inference_tau = float(inference_tau)

        model._norm_stats_per_col = artefacts.get("normalization", {})
        feature_layout = artefacts.get("feature_layout") or model._feature_layout
        model._feature_layout = feature_layout
        model._class_to_index = artefacts.get("class_to_index")
        classes = artefacts.get("classes")
        if classes is not None:
            model._classes = np.asarray(classes)
        cached_logits = artefacts.get("cached_logits")
        model._cached_logits = (
            None if cached_logits is None else np.asarray(cached_logits)
        )
        cached_prob = artefacts.get("cached_probabilities")
        model._cached_probabilities = (
            None if cached_prob is None else np.asarray(cached_prob)
        )
        model._logits_cache_key = artefacts.get("logits_cache_key")
        model._probability_cache_key = artefacts.get("probability_cache_key")
        model._temperature_scaler_state = artefacts.get("temperature_scaler_state")
        model._is_calibrated = bool(artefacts.get("is_calibrated", False))
        model._warmup_val_history = artefacts.get("warmup_val_history", [])
        model._joint_val_metrics = artefacts.get("joint_val_metrics")

        def _clone_optional_tensor(value: Any) -> Tensor | None:
            if value is None:
                return None
            tensor = torch.as_tensor(value)
            return tensor.clone().detach()

        model._train_latent_mu = _clone_optional_tensor(
            artefacts.get("train_latent_mu")
        )
        model._train_latent_logvar = _clone_optional_tensor(
            artefacts.get("train_latent_logvar")
        )
        model._train_component_logits = _clone_optional_tensor(
            artefacts.get("train_component_logits")
        )
        model._train_component_mu = _clone_optional_tensor(
            artefacts.get("train_component_mu")
        )
        model._train_component_logvar = _clone_optional_tensor(
            artefacts.get("train_component_logvar")
        )
        model._train_component_probs = _clone_optional_tensor(
            artefacts.get("train_component_probs")
        )
        train_targets = artefacts.get("train_target_indices")
        model._train_target_indices = (
            None if train_targets is None else np.asarray(train_targets)
        )

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model._device = device
        encoder_state = modules.get("encoder")
        decoder_state = modules.get("decoder")
        classifier_payload = modules.get("classifier")
        temperature_state = modules.get("temperature_scaler") or {}
        prior_state = modules.get("prior") or {}

        encoder_input_dim = cls._encoder_input_dim_from_layout(model._feature_layout)
        model._encoder = EncoderMLP(
            encoder_input_dim,
            model.latent_dim,
            hidden=model.hidden_dims,
            dropout=model.dropout,
            n_components=model.n_components,
        ).to(device)
        if isinstance(encoder_state, OrderedDict):
            model._encoder.load_state_dict(encoder_state)
        elif encoder_state is not None:
            model._encoder.load_state_dict(OrderedDict(encoder_state))

        if model.schema is None:
            raise ValueError("Schema is required to restore decoder state")
        model._decoder = Decoder(
            model.latent_dim,
            model.schema,
            hidden=model.hidden_dims,
            dropout=model.dropout,
            n_components=model.n_components,
        ).to(device)
        if isinstance(decoder_state, OrderedDict):
            model._decoder.load_state_dict(decoder_state)
        elif decoder_state is not None:
            model._decoder.load_state_dict(OrderedDict(decoder_state))

        if behaviour == "supervised" and classifier_payload:
            if model._classes is None:
                raise ValueError("Saved model is missing class labels")
            class_weight = None
            if classifier_payload.get("use_weight"):
                weight_value = classifier_payload.get("class_weight")
                class_weight = torch.as_tensor(weight_value, dtype=torch.float32)
            model._classifier = ClassificationHead(
                model.latent_dim,
                int(model._classes.size),
                class_weight=class_weight,
                dropout=model.dropout,
                hidden_dims=model.head_hidden_dims,
            ).to(device)
            state = classifier_payload.get("state_dict")
            if isinstance(state, OrderedDict):
                model._classifier.load_state_dict(state)
            elif state is not None:
                model._classifier.load_state_dict(OrderedDict(state))
        else:
            model._classifier = None

        model._temperature_scaler.load_state_dict(temperature_state)
        model._temperature_scaler_state = temperature_state
        model._is_calibrated = bool(temperature_state.get("fitted", False))

        model._move_prior_parameters_to_device(device)
        logits_value = prior_state.get("logits")
        if logits_value is not None:
            logits_tensor = torch.as_tensor(
                logits_value, dtype=torch.float32, device=device
            )
            param = model._prior_component_logits_tensor()
            if logits_tensor.shape != param.shape:
                raise ValueError(
                    "Saved prior logits do not match the model configuration"
                )
            with torch.no_grad():
                param.copy_(logits_tensor)
        logvar_value = prior_state.get("logvar")
        if logvar_value is not None:
            logvar_tensor = torch.as_tensor(
                logvar_value, dtype=torch.float32, device=device
            )
            param_logvar = model._prior_component_logvar_tensor()
            if logvar_tensor.shape != param_logvar.shape:
                raise ValueError(
                    "Saved prior log-variance does not match the model configuration"
                )
            with torch.no_grad():
                param_logvar.copy_(logvar_tensor)
        if model.behaviour == "unsupervised":
            if model._prior_mean_layer is None:
                raise RuntimeError("Prior mean layer is not initialised")
            mean_state = prior_state.get("mean_state_dict")
            if mean_state is not None:
                state = (
                    mean_state
                    if isinstance(mean_state, OrderedDict)
                    else OrderedDict(mean_state)
                )
                model._prior_mean_layer.load_state_dict(state)
            else:
                mu_value = prior_state.get("mu")
                if mu_value is not None:
                    means_tensor = torch.as_tensor(
                        mu_value, dtype=torch.float32, device=device
                    )
                    model._prior_mean_layer.load_component_means(means_tensor)
        else:
            mu_value = prior_state.get("mu")
            if mu_value is not None and model._prior_component_mu is not None:
                mu_tensor = torch.as_tensor(
                    mu_value, dtype=torch.float32, device=device
                )
                if mu_tensor.shape != model._prior_component_mu.shape:
                    raise ValueError(
                        "Saved prior means do not match the model configuration"
                    )
                with torch.no_grad():
                    model._prior_component_mu.copy_(mu_tensor)

        model._is_fitted = True
        return model

    @classmethod
    def _load_from_legacy_json(cls, data: dict[str, Any]) -> "SUAVE":
        """Fallback loader for the legacy JSON serialisation format."""

        if not isinstance(data, dict):
            raise TypeError("Legacy payload must be a mapping")

        metadata = dict(data.get("metadata") or {})
        top_level_metadata_keys = {
            "schema",
            "behaviour",
            "latent_dim",
            "n_components",
            "hidden_dims",
            "dropout",
            "learning_rate",
            "batch_size",
            "kl_warmup_epochs",
            "val_split",
            "stratify",
            "random_state",
            "tau_start",
            "tau_min",
            "tau_decay",
            "inference_tau",
        }
        for key in top_level_metadata_keys:
            if key not in metadata and key in data:
                metadata[key] = data[key]
        metadata.setdefault("schema", data.get("schema"))
        behaviour_value = metadata.get("behaviour", data.get("behaviour", "supervised"))
        behaviour_value = _normalise_behaviour(str(behaviour_value))
        metadata["behaviour"] = behaviour_value

        prior_payload = data.get("prior") or data.get("modules", {}).get("prior")
        inferred_latent_dim: int | None = None
        inferred_components: int | None = None
        if isinstance(prior_payload, dict):
            mu_value = prior_payload.get("mu")
            if mu_value is not None:
                mu_tensor: Tensor | None
                if isinstance(mu_value, Tensor):
                    mu_tensor = mu_value
                else:
                    try:
                        mu_tensor = torch.as_tensor(mu_value)
                    except (TypeError, ValueError):
                        mu_tensor = None
                if mu_tensor is not None and mu_tensor.ndim > 0:
                    if mu_tensor.ndim == 1:
                        inferred_latent_dim = int(mu_tensor.shape[0])
                        inferred_components = 1
                    else:
                        inferred_components = int(mu_tensor.shape[0])
                        inferred_latent_dim = int(mu_tensor.shape[-1])
        if inferred_latent_dim is not None:
            if "latent_dim" not in metadata or metadata["latent_dim"] is None:
                metadata["latent_dim"] = inferred_latent_dim
        if inferred_components is not None:
            if "n_components" not in metadata or metadata["n_components"] is None:
                metadata["n_components"] = inferred_components

        modules = dict(data.get("modules") or {})
        for key in ("encoder", "decoder", "classifier", "temperature_scaler", "prior"):
            if key not in modules and key in data:
                modules[key] = data[key]

        artefact_keys = [
            "classes",
            "class_to_index",
            "normalization",
            "feature_layout",
            "temperature_scaler_state",
            "is_calibrated",
            "train_latent_mu",
            "train_latent_logvar",
            "train_component_logits",
            "train_component_mu",
            "train_component_logvar",
            "train_component_probs",
            "train_target_indices",
            "cached_logits",
            "cached_probabilities",
        ]
        artefacts = dict(data.get("artefacts") or {})
        for key in artefact_keys:
            if key not in artefacts and key in data:
                artefacts[key] = data[key]

        payload: dict[str, Any] = {
            "metadata": metadata,
            "modules": modules,
            "artefacts": artefacts,
        }

        def _coerce_tensor(value: Any) -> Tensor:
            if value is None:
                raise TypeError("Cannot convert None to tensor")
            if isinstance(value, Tensor):
                tensor = value.detach().clone()
            else:
                try:
                    tensor = torch.as_tensor(value)
                except (TypeError, ValueError) as exc:
                    if isinstance(value, dict):
                        try:
                            ordered_values = [
                                value[key] for key in sorted(value.keys())
                            ]
                        except Exception as inner_exc:  # pragma: no cover - defensive
                            raise TypeError(
                                "Cannot interpret legacy tensor payload"
                            ) from inner_exc
                        tensor = torch.as_tensor(ordered_values)
                    else:
                        raise TypeError(
                            "Cannot interpret legacy tensor payload"
                        ) from exc
            tensor = tensor.clone().detach()
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=torch.float32)
            return tensor.cpu()

        def _convert_state_dict(state: Any) -> OrderedDict[str, Tensor] | None:
            if state is None:
                return None
            if isinstance(state, (OrderedDict, dict)):
                items = list(state.items())
            elif isinstance(state, list):
                items = []
                for entry in state:
                    if isinstance(entry, dict) and {"key", "value"} <= entry.keys():
                        items.append((entry["key"], entry["value"]))
                    elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                        items.append((entry[0], entry[1]))
                    else:  # pragma: no cover - defensive
                        raise ValueError("Unsupported legacy state dict entry format")
            else:  # pragma: no cover - defensive
                raise ValueError("Unsupported legacy state dict format")
            converted = OrderedDict()
            for key, value in items:
                if value is None:
                    continue
                converted[key] = _coerce_tensor(value)
            return converted

        for module_key in ("encoder", "decoder"):
            state = modules.get(module_key)
            converted_state = _convert_state_dict(state)
            if converted_state is not None:
                modules[module_key] = converted_state

        classifier_payload = modules.get("classifier")
        if isinstance(classifier_payload, dict):
            classifier_state = classifier_payload.get("state_dict")
            converted_state = _convert_state_dict(classifier_state)
            if converted_state is not None:
                classifier_payload["state_dict"] = converted_state
            class_weight = classifier_payload.get("class_weight")
            if class_weight is not None:
                try:
                    classifier_payload["class_weight"] = _coerce_tensor(class_weight)
                except TypeError:
                    pass

        prior_state = modules.get("prior")
        if isinstance(prior_state, dict):
            for key in ("logits", "logvar", "mu"):
                if key in prior_state and prior_state[key] is not None:
                    try:
                        prior_state[key] = _coerce_tensor(prior_state[key])
                    except TypeError:
                        pass
            mean_state = prior_state.get("mean_state_dict")
            converted_mean = _convert_state_dict(mean_state)
            if converted_mean is not None:
                prior_state["mean_state_dict"] = converted_mean

        return cls._load_from_payload(payload)

    @staticmethod
    def _encoder_input_dim_from_layout(
        feature_layout: dict[str, dict[str, int]],
    ) -> int:
        """Return encoder input dimensionality given ``feature_layout``."""

        value_dims = 0
        mask_dims = 0
        for columns in feature_layout.values():
            value_dims += sum(int(width) for width in columns.values())
            mask_dims += len(columns)
        return value_dims + mask_dims

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
