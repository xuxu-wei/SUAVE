"""Placeholder decoder module for the minimal SUAVE package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class Decoder:
    """Minimal configuration holder for the generative decoder."""

    output_dims: Sequence[int]

    def __call__(self, latents):  # pragma: no cover - placeholder behaviour
        return latents
