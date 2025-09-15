"""Top-level package exporting the SUAVE model."""

from .models.suave import AnnealSchedule, InfoVAEConfig, SUAVE
from .sklearn import SuaveClassifier

__all__ = ["SUAVE", "SuaveClassifier", "AnnealSchedule", "InfoVAEConfig"]
__version__ = "0.1.2a1"
