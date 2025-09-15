"""Latent variable interpretation helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from statsmodels.stats.multitest import multipletests


def latent_feature_correlation(
    Z: np.ndarray, features: np.ndarray, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spearman correlation between latents ``Z`` and input ``features``.

    Applies Benjamini–Hochberg FDR correction and returns the correlation
    matrix, p-values and a boolean significance mask.
    """

    d_z, d_x = Z.shape[1], features.shape[1]
    corr = np.zeros((d_z, d_x))
    pval = np.zeros((d_z, d_x))
    for i in range(d_z):
        for j in range(d_x):
            r, p = spearmanr(Z[:, i], features[:, j])
            corr[i, j] = r
            pval[i, j] = p
    _, qvals, _, _ = multipletests(pval.ravel(), alpha=alpha, method="fdr_bh")
    sig = qvals.reshape(pval.shape) < alpha
    return corr, pval, sig


def embed_latent(
    Z: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
    random_state: Optional[int] = 0,
):
    """Embed latent codes using UMAP or t-SNE."""

    if method == "umap":  # pragma: no cover - optional dependency
        import umap

        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(Z)
    reducer = TSNE(n_components=n_components, random_state=random_state)
    return reducer.fit_transform(Z)
