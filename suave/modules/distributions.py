"""Distribution helpers used by the future VAE implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NormalDistribution:
    """Trivial normal distribution stub storing mean and scale."""

    mean: float
    scale: float

    def sample(self):  # pragma: no cover - placeholder behaviour
        return self.mean
