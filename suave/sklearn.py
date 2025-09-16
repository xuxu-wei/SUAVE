from __future__ import annotations

"""Lightweight scikit-learn style wrapper around :class:`SUAVE`."""

from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score

from .models.suave import SUAVE


class SuaveClassifier:
    """Train a separate :class:`SUAVE` model for each task.

    Parameters
    ----------
    input_dim:
        Number of input features.
    task_classes:
        List specifying the number of classes for each prediction task.
    latent_dim:
        Dimensionality of the latent space for each SUAVE model (default: 8).
    **model_kwargs:
        Additional keyword arguments forwarded to :class:`SUAVE`. This can be
        used to enable :math:`\beta`-VAE (via ``beta``) or InfoVAE
        (``info_config``) behaviour.
    """

    def __init__(self, input_dim: int, task_classes: List[int], latent_dim: int = 8, **model_kwargs: object):
        self.input_dim = input_dim
        self.task_classes = task_classes
        self.latent_dim = latent_dim
        self.model_kwargs = model_kwargs
        self.models = [
            SUAVE(
                input_dim=input_dim,
                latent_dim=latent_dim,
                num_classes=c,
                **model_kwargs,
            )
            for c in task_classes
        ]

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 20,
        *,
        batch_size: int | None = None,
        **_: object,
    ) -> "SuaveClassifier":
        for model, y in zip(self.models, Y.T):
            model.fit(X, y, epochs=epochs, batch_size=batch_size)
        return self

    def predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        return [m.predict_proba(X) for m in self.models]

    def predict(self, X: np.ndarray) -> List[np.ndarray]:
        return [m.predict(X) for m in self.models]

    def score(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        auc_scores = []
        for idx, (c, proba) in enumerate(zip(self.task_classes, probas)):
            y = Y[:, idx]
            if c > 2:
                auc = roc_auc_score(y, proba, multi_class="ovr", average="macro")
            else:
                classes = np.unique(y)
                if proba.ndim == 1 or proba.shape[1] == 1:
                    pos_scores = proba.squeeze()
                else:
                    pos_label = classes[-1]
                    class_order = classes if classes.size == proba.shape[1] else np.arange(proba.shape[1])
                    matches = np.flatnonzero(class_order == pos_label)
                    pos_idx = int(matches[0]) if matches.size else proba.shape[1] - 1
                    pos_scores = proba[:, pos_idx]
                auc = roc_auc_score(y, pos_scores)
            auc_scores.append(auc)
        return np.asarray(auc_scores)
