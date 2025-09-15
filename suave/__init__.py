"""Top-level package exporting the SUAVE model."""

from .models.suave import SUAVE
from .sklearn import SuaveClassifier

__all__ = ["SUAVE", "SuaveClassifier"]
__version__ = "0.1.2a1"
