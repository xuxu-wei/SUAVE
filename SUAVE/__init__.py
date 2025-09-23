"""Compatibility shim exposing the lowercase :mod:`suave` package as ``SUAVE``."""

from importlib import import_module
import sys

import suave as _suave
from suave import *  # noqa: F401,F403

__path__ = list(_suave.__path__)

for _name in (
    "interactive",
    "types",
    "schema_inference",
    "data",
    "evaluate",
    "sampling",
):
    sys.modules[f"SUAVE.{_name}"] = import_module(f"suave.{_name}")

sys.modules["SUAVE.interactive.schema_builder"] = import_module(
    "suave.interactive.schema_builder"
)

del import_module, sys, _suave, _name
