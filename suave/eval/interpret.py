"""Latent variable interpretation helpers."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr


def latent_feature_correlation(
    Z: np.ndarray,
    features: np.ndarray,
    feature_idx: Sequence[int] | None = None,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spearman correlation with Benjamini–Hochberg FDR correction.

    Parameters
    ----------
    Z:
        Latent codes of shape ``(n_samples, latent_dim)``.
    features:
        Input features of shape ``(n_samples, n_features)``.
    feature_idx:
        Optional subset of feature indices to include.
    alpha:
        FDR control level for BH-FDR correction.

    Returns
    -------
    corr : np.ndarray
        Correlation coefficients of shape ``(latent_dim, n_features_sel)``.
    signif : np.ndarray
        Boolean matrix indicating statistically significant correlations after
        BH-FDR correction.
    """

    if feature_idx is not None:
        features = features[:, list(feature_idx)]

    combined = np.column_stack([Z, features])
    corr_full, pval_full = spearmanr(combined, axis=0)
    d_z = Z.shape[1]
    corr = corr_full[:d_z, d_z:]
    pval = pval_full[:d_z, d_z:]

    # Benjamini–Hochberg FDR
    flat_p = pval.ravel()
    order = np.argsort(flat_p)
    ranked = np.arange(1, len(flat_p) + 1)
    thresh = alpha * ranked / len(flat_p)
    sig_flat = flat_p[order] <= thresh
    signif = np.zeros_like(flat_p, dtype=bool)
    signif[order] = sig_flat
    signif = signif.reshape(pval.shape)

    return corr, signif


def latent_projection(
    Z: np.ndarray, method: str = "tsne", **kwargs
) -> np.ndarray:
    """Project latent codes to 2-D using UMAP or t-SNE."""

    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, **kwargs).fit_transform(Z)
    if method == "umap":
        try:  # pragma: no cover - optional dependency
            import umap
        except Exception as e:  # pragma: no cover - optional
            raise ImportError("umap-learn is required for method='umap'") from e

        return umap.UMAP(n_components=2, **kwargs).fit_transform(Z)
    raise ValueError("Unknown projection method")


__all__ = ["latent_feature_correlation", "latent_projection"]
