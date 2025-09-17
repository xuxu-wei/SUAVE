"""Classification heads for the minimal SUAVE placeholder implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassificationHead:
    """Placeholder classification head returning zero logits."""

    n_classes: int

    def __call__(self, latents):  # pragma: no cover - placeholder behaviour
        return [0.0] * self.n_classes
