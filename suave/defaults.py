"""Heuristic hyperparameter recommendations for SUAVE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = [
    "HeuristicHyperparameters",
    "recommend_hyperparameters",
    "serialise_heuristic_hyperparameters",
    "parse_heuristic_hyperparameters",
]


@dataclass(frozen=True)
class HeuristicHyperparameters:
    """Container for the heuristically derived hyperparameters."""

    latent_dim: int
    hidden_dims: tuple[int, ...]
    dropout: float
    learning_rate: float
    batch_size: int
    kl_warmup_epochs: int
    warmup_epochs: int
    head_epochs: int
    finetune_epochs: int
    early_stop_patience: int

    def to_dict(self) -> dict[str, object]:
        """Return a plain ``dict`` representation."""

        return {
            "latent_dim": int(self.latent_dim),
            "hidden_dims": tuple(int(v) for v in self.hidden_dims),
            "dropout": float(self.dropout),
            "learning_rate": float(self.learning_rate),
            "batch_size": int(self.batch_size),
            "kl_warmup_epochs": int(self.kl_warmup_epochs),
            "warmup_epochs": int(self.warmup_epochs),
            "head_epochs": int(self.head_epochs),
            "finetune_epochs": int(self.finetune_epochs),
            "early_stop_patience": int(self.early_stop_patience),
        }


def _dataset_size_bucket(n_samples: int) -> str:
    """Categorise ``n_samples`` into coarse dataset regimes."""

    if n_samples <= 0 or n_samples <= 512:
        return "tiny"
    if n_samples <= 5000:
        return "small"
    if n_samples <= 20000:
        return "medium"
    return "large"


def _round_up(value: int, multiple: int) -> int:
    """Return ``value`` rounded up to the nearest ``multiple``."""

    if multiple <= 0:
        return int(value)
    value = int(value)
    if value % multiple == 0:
        return value
    return int(((value // multiple) + 1) * multiple)


def _round_to_power_of_two(value: int) -> int:
    """Return ``value`` rounded to the closest power of two."""

    value = max(int(value), 1)
    exponent = int(round(math.log2(value))) if value > 0 else 0
    return int(2 ** max(exponent, 0))


def _recommend_hidden_dims(
    input_dim: int, latent_dim: int, n_samples: int
) -> tuple[int, int]:
    """Return a two-layer MLP width suited to ``input_dim``."""

    bucket = _dataset_size_bucket(n_samples)
    width = max(input_dim, latent_dim * 4)
    width = _round_up(width, 16)
    if bucket == "tiny":
        width = max(96, min(width, 192))
    elif bucket == "small":
        width = max(128, min(width, 256))
    elif bucket == "medium":
        width = max(192, min(width, 320))
    else:
        width = max(256, min(width, 384))
    second = max(latent_dim * 2, width // 2)
    second = _round_up(second, 16)
    second = min(second, width)
    second = max(16, second)
    return int(width), int(second)


def _recommend_dropout(n_samples: int) -> float:
    """Return a dropout level inversely proportional to dataset size."""

    bucket = _dataset_size_bucket(n_samples)
    if bucket == "tiny":
        return 0.2
    if bucket == "small":
        return 0.15
    if bucket == "medium":
        return 0.1
    return 0.05


def _recommend_batch_size(n_samples: int) -> int:
    """Return a mini-batch size bounded by ``n_samples``."""

    n_samples = max(int(n_samples), 1)
    bucket = _dataset_size_bucket(n_samples)
    if bucket == "tiny":
        candidate = _round_to_power_of_two(max(n_samples // 4, 1))
        candidate = max(4, min(candidate, 64))
    elif bucket == "small":
        candidate = 128
    elif bucket == "medium":
        candidate = 192
    else:
        candidate = 256
    return int(max(1, min(candidate, n_samples)))


def _recommend_learning_rate(n_samples: int) -> float:
    """Return an Adam learning rate tuned for ``n_samples``."""

    bucket = _dataset_size_bucket(n_samples)
    if bucket == "tiny":
        return 2e-3
    if bucket == "small":
        return 1e-3
    if bucket == "medium":
        return 8e-4
    return 5e-4


def _recommend_kl_warmup_epochs(n_samples: int) -> int:
    """Return KL warm-up epochs scaled to dataset size."""

    bucket = _dataset_size_bucket(n_samples)
    if bucket == "tiny":
        return 5
    if bucket == "small":
        return 10
    if bucket == "medium":
        return 15
    return 20


def _recommend_warmup_epochs(n_samples: int) -> int:
    """Return warm-up epochs before classifier training."""

    bucket = _dataset_size_bucket(n_samples)
    if bucket == "tiny":
        return 15
    if bucket == "small":
        return 20
    if bucket == "medium":
        return 25
    return 30


def _recommend_head_epochs(n_samples: int, class_counts: np.ndarray | None) -> int:
    """Return classifier head epochs factoring class balance."""

    bucket = _dataset_size_bucket(n_samples)
    base = {"tiny": 5, "small": 6, "medium": 8, "large": 10}[bucket]
    if class_counts is not None and class_counts.size > 0:
        total = float(class_counts.sum())
        if total > 0:
            frequencies = class_counts / total
            min_freq = float(frequencies.min())
            if min_freq < 0.1:
                base += 2
            elif min_freq < 0.2:
                base += 1
    return int(base)


def _recommend_finetune_epochs(n_samples: int) -> int:
    """Return fine-tuning epochs for the joint stage."""

    bucket = _dataset_size_bucket(n_samples)
    if bucket == "tiny":
        return 8
    if bucket == "small":
        return 10
    if bucket == "medium":
        return 15
    return 20


def _recommend_early_stop_patience(n_samples: int) -> int:
    """Return early-stopping patience suited to ``n_samples``."""

    bucket = _dataset_size_bucket(n_samples)
    if bucket == "tiny":
        return 3
    if bucket == "small":
        return 5
    if bucket == "medium":
        return 6
    return 8


def recommend_hyperparameters(
    *, input_dim: int, n_train_samples: int, class_counts: np.ndarray | None
) -> HeuristicHyperparameters:
    """Return heuristic hyperparameters tailored to the dataset."""

    latent_dim = max(8, min(128, int(round(math.sqrt(max(input_dim, 1)) * 4))))
    bucket = _dataset_size_bucket(n_train_samples)
    if bucket == "tiny":
        latent_dim = int(max(8, min(latent_dim, 24)))
    elif bucket == "small":
        latent_dim = int(max(16, min(latent_dim, 40)))
    elif bucket == "medium":
        latent_dim = int(max(24, min(latent_dim, 64)))
    else:
        latent_dim = int(max(32, min(latent_dim, 96)))

    hidden_dims = _recommend_hidden_dims(input_dim, latent_dim, n_train_samples)
    dropout = _recommend_dropout(n_train_samples)
    batch_size = _recommend_batch_size(n_train_samples)
    learning_rate = _recommend_learning_rate(n_train_samples)
    kl_epochs = _recommend_kl_warmup_epochs(n_train_samples)
    warmup_epochs = _recommend_warmup_epochs(n_train_samples)
    head_epochs = _recommend_head_epochs(n_train_samples, class_counts)
    finetune_epochs = _recommend_finetune_epochs(n_train_samples)
    patience = _recommend_early_stop_patience(n_train_samples)

    return HeuristicHyperparameters(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        kl_warmup_epochs=kl_epochs,
        warmup_epochs=warmup_epochs,
        head_epochs=head_epochs,
        finetune_epochs=finetune_epochs,
        early_stop_patience=patience,
    )


def serialise_heuristic_hyperparameters(
    config: HeuristicHyperparameters | dict[str, object] | None,
) -> dict[str, object] | None:
    """Convert heuristic hyperparameters to a JSON-friendly mapping."""

    if config is None:
        return None
    if isinstance(config, HeuristicHyperparameters):
        data = config.to_dict()
    else:
        data = dict(config)
    serialised: dict[str, object] = {}
    for key, value in data.items():
        if value is None:
            continue
        if key == "hidden_dims":
            serialised[key] = [int(v) for v in value]
        elif isinstance(value, (tuple, list)):
            serialised[key] = [int(v) for v in value]
        elif isinstance(value, np.ndarray):
            serialised[key] = value.tolist()
        elif isinstance(value, (np.generic,)):
            serialised[key] = value.item()
        elif isinstance(value, float):
            serialised[key] = float(value)
        elif isinstance(value, int):
            serialised[key] = int(value)
        else:
            serialised[key] = value
    return serialised or None


def parse_heuristic_hyperparameters(
    data: dict[str, object] | None,
) -> dict[str, object]:
    """Reconstruct heuristic hyperparameters from serialised form."""

    if not data:
        return {}
    parsed: dict[str, object] = {}
    for key, value in data.items():
        if value is None:
            continue
        if key == "hidden_dims":
            parsed[key] = tuple(int(v) for v in value)
        elif key in {
            "latent_dim",
            "batch_size",
            "kl_warmup_epochs",
            "warmup_epochs",
            "head_epochs",
            "finetune_epochs",
            "early_stop_patience",
        }:
            parsed[key] = int(value)
        elif key in {"dropout", "learning_rate"}:
            parsed[key] = float(value)
        else:
            parsed[key] = value
    return parsed
