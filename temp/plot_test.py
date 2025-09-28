# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: ai_exp
#     language: python
#     name: python3
# ---

# %%
import sys
from pathlib import Path
def _resolve_examples_dir() -> Path:
    """Resolve the examples directory for both scripts and notebooks."""

    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        # ``__file__`` is not defined inside notebooks executed via Jupyter.
        pass

    cwd = Path.cwd()
    candidates.extend([cwd])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "Run this notebook from the repository root so 'temp' is available."
    )
    
ROOT_DIR = _resolve_examples_dir().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from suave import SUAVE  # noqa: E402
from suave.plots import (  # noqa: E402
    compute_feature_latent_correlation,
    plot_feature_latent_correlation_bubble,
    plot_feature_latent_correlation_heatmap,
    plot_feature_latent_outcome_path_graph,
)


