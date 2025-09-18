"""Loss helpers for the placeholder implementation."""

from __future__ import annotations


def elbo_placeholder(*_, **__):  # pragma: no cover - placeholder behaviour
    """Return a zero ELBO value.

    Parameters are ignored for the minimal package skeleton.
    """

    return 0.0
