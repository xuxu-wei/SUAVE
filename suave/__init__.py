"""Top-level package for the lightweight SUAVE API.

This module exposes the user-facing classes that are required for the
minimal runnable skeleton requested in the roadmap.  The real
implementation will iterate on top of these public entry points.
"""

from .model import SUAVE
from .types import Schema
from .schema_inference import SchemaInferenceMode, SchemaInferenceResult, SchemaInferencer

__all__ = [
    "SUAVE",
    "Schema",
    "SchemaInferencer",
    "SchemaInferenceMode",
    "SchemaInferenceResult",
]
