"""Top-level package exporting the TabVAE-based SUAVE model.

``SUAVE`` now refers to :class:`~suave.models.tabvae.TabVAEClassifier`.  The
previous implementation remains accessible as :func:`suave_old_version` for
backwards compatibility and will be dropped in a future release.
"""

from .models.tabvae import TabVAEClassifier as SUAVE
from .suave import suave_old_version
from .sklearn import SuaveClassifier
from .api import TabVAEClassifier

__all__ = ["SUAVE", "suave_old_version", "SuaveClassifier", "TabVAEClassifier"]
__version__ = "0.1.2a1"
