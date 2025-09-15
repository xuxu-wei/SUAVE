"""Compatibility layer exposing the :class:`TabVAEClassifier`.

The intent of this module is to keep the external API stable while
internally delegating to the new implementation located in
:mod:`suave.models.tabvae`.
"""

from __future__ import annotations

from ..models.tabvae import TabVAEClassifier

__all__ = ["TabVAEClassifier"]
