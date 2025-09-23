"""
This module exposes the user-facing classes that are required for the
minimal runnable skeleton requested in the roadmap.  The real
implementation will iterate on top of these public entry points.

In addition to the primary estimator and schema helpers, we re-export
common utility submodules (``data``, ``evaluate``, ``sampling`` and
``interactive``) so that ``import suave`` mirrors the public namespace
documented throughout the examples.  Users can therefore rely on
``suave.data`` or ``suave.evaluate`` without performing an additional
import of the underlying modules.
"""

from importlib import import_module

from .model import SUAVE
from .types import Schema
from .schema_inference import (
    SchemaInferenceMode,
    SchemaInferenceResult,
    SchemaInferencer,
)

# Re-export commonly used helper modules so that importing ``suave``
# exposes the documented namespace without requiring users to jump
# between subpackages explicitly.
data = import_module(".data", __name__)
evaluate = import_module(".evaluate", __name__)
interactive = import_module(".interactive", __name__)
sampling = import_module(".sampling", __name__)
types = import_module(".types", __name__)
schema_inference = import_module(".schema_inference", __name__)

__all__ = [
    "SUAVE",
    "Schema",
    "SchemaInferencer",
    "SchemaInferenceMode",
    "SchemaInferenceResult",
    "data",
    "evaluate",
    "interactive",
    "sampling",
    "types",
    "schema_inference",
]
