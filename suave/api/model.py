"""Compatibility layer exposing :class:`TabVAEClassifier` as ``SUAVE``.

The project now uses the TabVAE-based classifier as the primary model.  The
previous architecture is retained as :func:`suave_old_version` for legacy
purposes and will be dropped in a future release.
"""

from __future__ import annotations

from ..models.tabvae import TabVAEClassifier

__all__ = ["TabVAEClassifier"]
