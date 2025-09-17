"""Placeholder encoder module for the minimal SUAVE package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class Encoder:
    """Lightweight placeholder for the hierarchical VAE encoder.

    Parameters
    ----------
    hidden_dims:
        Sequence describing the hidden layer sizes.  The class does not
        perform any computation yet; it merely stores configuration so the
        training pipeline can be wired in subsequent iterations.
    """

    hidden_dims: Sequence[int]

    def __call__(self, inputs):  # pragma: no cover - placeholder behaviour
        return inputs
