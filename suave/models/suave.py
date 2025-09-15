"""SUAVE model.

This module implements a variational autoencoder with an attached classifier
where the latent variable ``z`` acts as a single information hub.  By routing
prediction, missing-free sample synthesis and latent-factor interpretation
through ``z`` we provide a unified interface for downstream clinical workflows.
The design follows a *probability first* philosophy: mixed data types,
missingness and class imbalance are handled with explicit likelihood terms
rather than ad-hoc preprocessing.  The latent-variable view offers compact and
interpretable representations that are especially useful in clinical research.

Only continuous variables are implemented here; categorical/count support can be
added in a backwards compatible fashion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.losses import (
    gaussian_nll,
    kl_divergence,
    linear_anneal,
    maximum_mean_discrepancy,
)


@dataclass
class AnnealSchedule:
    """Linear weight schedule used for KL and classification terms."""

    start: float
    end: float
    epochs: int


@dataclass
class InfoVAEConfig:
    """Configuration for the InfoVAE objective.

    Parameters
    ----------
    alpha:
        Balances the mutual information term in InfoVAE. ``alpha=0`` recovers
        the vanilla (or :math:`\beta`-) VAE objective.
    lambda_:
        Coefficient for the MMD penalty enforcing ``q(z)`` close to the prior.
    kernel:
        Kernel used for the MMD computation. Only ``"rbf"`` is currently
        implemented which matches the original InfoVAE paper.
    kernel_bandwidth:
        Bandwidth of the RBF kernel; larger values encourage smoother matching
        between the aggregated posterior and the prior.
    """

    alpha: float = 0.0
    lambda_: float = 1.0
    kernel: str = "rbf"
    kernel_bandwidth: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in the interval [0, 1]")
        if self.lambda_ <= 0:
            raise ValueError("lambda_ must be positive")
        if self.kernel != "rbf":
            raise NotImplementedError("Only the RBF kernel is implemented")
        if self.kernel_bandwidth <= 0:
            raise ValueError("kernel_bandwidth must be positive")

    @property
    def kl_weight(self) -> float:
        return 1.0 - self.alpha

    @property
    def mmd_weight(self) -> float:
        return self.alpha + self.lambda_ - 1.0


class Encoder(nn.Module):
    """MLP encoder producing ``mu`` and ``log_var``.

    The encoder consumes the concatenation of the observed values ``x`` and a
    binary mask ``m`` that indicates which entries are present (1=observed).
    Architecture: ``(2*D) -> 256 -> 256 -> 128`` with SiLU activations,
    batch normalisation and dropout.
    """

    def __init__(self, input_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        in_dim = input_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([x, m], dim=-1))
        return self.mu(h), self.log_var(h), h


class Decoder(nn.Module):
    """Shared decoder with a Gaussian head for continuous variables."""

    def __init__(self, latent_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        )
        self.mu = nn.Linear(128, output_dim)
        self.log_sigma = nn.Linear(128, output_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(z)
        return self.mu(h), self.log_sigma(h)


class Classifier(nn.Module):
    """Predictor operating on the latent representation and encoder features."""

    def __init__(self, in_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SUAVE(nn.Module):
    """Unified model for prediction, generation and interpretation.

    The model exposes a single latent representation ``z`` which serves as the
    meeting point for all three tasks.  A classifier operates on ``z`` (and the
    last encoder layer) to provide predictions, the decoder reconstructs inputs
    to enable missing-free synthetic data generation, and ``z`` can be probed for
    correlations with clinical covariates.

    This class is exported from :mod:`suave` as ``SUAVE``.

    Parameters
    ----------
    input_dim:
        Number of input features.
    latent_dim:
        Dimensionality of the latent space (default: 8). Clinical use cases
        typically operate with latent dimensions in the 4–8 range.
    num_classes:
        Number of target classes (default: 2).
    dropout:
        Dropout rate for all MLPs (default: 0.3).
    beta_schedule, lambda_schedule:
        Linear schedules controlling the weight of the KL and classification
        terms during optimisation.
    beta:
        Global multiplier for the KL term. Setting ``beta > 1`` recovers a
        :math:`\beta`-VAE style objective while ``beta = 1`` corresponds to the
        default VAE. ``beta`` is combined with ``beta_schedule`` to allow for
        KL annealing.
    info_config:
        Optional configuration enabling the InfoVAE objective. When provided,
        the KL term is scaled by ``(1 - alpha)`` and an additional MMD penalty
        ``(alpha + lambda - 1) * MMD(q(z), p(z))`` is added. Passing
        ``InfoVAEConfig(alpha=0, lambda_=1)`` therefore falls back to the
        standard VAE formulation.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        num_classes: int = 2,
        dropout: float = 0.3,
        beta_schedule: AnnealSchedule | None = None,
        lambda_schedule: AnnealSchedule | None = None,
        beta: float = 1.0,
        info_config: InfoVAEConfig | None = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device or torch.device("cpu")

        self.encoder = Encoder(input_dim, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, input_dim, dropout)
        # Classifier consumes latent code concatenated with last encoder layer
        self.classifier = Classifier(latent_dim + 128, num_classes, dropout)

        self.beta_schedule = beta_schedule or AnnealSchedule(0.0, 0.7, 15)
        self.lambda_schedule = lambda_schedule or AnnealSchedule(0.5, 1.0, 15)
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = beta
        self.info_config = info_config

        self.to(self.device)

        # Book-keeping for evaluation: updated during training.
        self._last_epoch = 0
        initial_beta = self.beta * linear_anneal(0, self.beta_schedule)
        if self.info_config is not None:
            initial_beta *= self.info_config.kl_weight
        self._current_beta = float(initial_beta)
        self._current_lambda = float(linear_anneal(0, self.lambda_schedule))
        self._current_mmd = (
            float(self.info_config.mmd_weight) if self.info_config else 0.0
        )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ----------------------------------------------------------------- forward
    def forward(
        self, x: torch.Tensor, m: Optional[torch.Tensor] = None
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        if m is None:
            m = torch.ones_like(x)
        mu, log_var, h = self.encoder(x, m)
        z = self._reparameterize(mu, log_var)
        recon_mu, recon_log_sigma = self.decoder(z)
        logits = self.classifier(torch.cat([z, h], dim=-1))
        return z, recon_mu, recon_log_sigma, mu, log_var, logits

    # ---------------------------------------------------------------- training
    def fit(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 20, lr: float = 3e-4
    ) -> "SUAVE":
        """Train the model using full-batch optimisation.

        The routine is intentionally simple for testability.  Missing values in
        ``X`` are indicated by ``NaN`` and are ignored in the reconstruction
        loss while still influencing the encoder through the mask.
        """

        self.train()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        mask = torch.isfinite(X_t).float()
        X_t = torch.nan_to_num(X_t, nan=0.0)

        # retain training data for conditional generation
        self.X_train = X
        self.y_train = y

        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(epochs):
            opt.zero_grad()
            z, recon_mu, recon_log_sigma, mu, log_var, logits = self.forward(
                X_t, mask
            )
            nll = gaussian_nll(recon_mu, recon_log_sigma, X_t, mask)
            kl = kl_divergence(mu, log_var)
            ce = F.cross_entropy(logits, y_t)
            beta = self.beta * linear_anneal(epoch, self.beta_schedule)
            lam = linear_anneal(epoch, self.lambda_schedule)
            if self.info_config is not None:
                kl_weight = beta * self.info_config.kl_weight
                info_weight = self.info_config.mmd_weight
                prior_samples = torch.randn_like(z)
                mmd = maximum_mean_discrepancy(
                    z,
                    prior_samples,
                    kernel=self.info_config.kernel,
                    bandwidth=self.info_config.kernel_bandwidth,
                )
                info_penalty = info_weight * mmd
            else:
                kl_weight = beta
                info_weight = 0.0
                info_penalty = kl.new_tensor(0.0)
            loss = (nll + kl_weight * kl) + lam * ce + info_penalty
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            self._last_epoch = epoch
            self._current_beta = float(kl_weight)
            self._current_lambda = float(lam)
            self._current_mmd = float(info_weight)
        return self

    # --------------------------------------------------------------- inference
    def predict_logits(self, X: np.ndarray) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            mask = torch.isfinite(X_t).float()
            X_t = torch.nan_to_num(X_t, nan=0.0)
            z, _, _, _, _, logits = self.forward(X_t, mask)
        return logits.cpu()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(X)
        return torch.softmax(logits, dim=-1).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # --------------------------------------------------------------- generation
    def generate(
        self,
        n_samples: int,
        conditional: Optional[dict[str, np.ndarray | float | int]] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate ``n_samples`` synthetic rows (mode 0: no missing values).

        Parameters
        ----------
        n_samples:
            Number of samples to generate.
        conditional:
            Optional dictionary specifying conditioning information. Currently
            supports ``{"y": label}`` for class-conditioned generation where the
            label can be a scalar or an array of length ``n_samples``.
        seed:
            Optional random seed for reproducibility.
        """

        rng = np.random.default_rng(seed)

        if conditional and "y" in conditional:
            if not hasattr(self, "X_train"):
                raise RuntimeError("Model must be fitted before conditional generation")
            y_cond = conditional["y"]
            if np.ndim(y_cond) == 0:
                y_vec = np.full(n_samples, int(y_cond))
            else:
                y_vec = np.asarray(y_cond, dtype=int)
                if len(y_vec) != n_samples:
                    raise ValueError("Length of conditional labels must match n_samples")
            idxs = []
            for lbl in y_vec:
                candidates = np.where(self.y_train == lbl)[0]
                if len(candidates) == 0:
                    raise ValueError(f"No training samples with label {lbl}")
                idxs.append(rng.choice(candidates))
            X_sel = self.X_train[idxs]
            X_t = torch.as_tensor(X_sel, dtype=torch.float32, device=self.device)
            mask = torch.isfinite(X_t).float()
            X_t = torch.nan_to_num(X_t, nan=0.0)
            with torch.no_grad():
                mu, log_var, _ = self.encoder(X_t, mask)
                z = self._reparameterize(mu, log_var)
                mu_x, log_sigma = self.decoder(z)
                sigma = torch.exp(log_sigma)
                x = mu_x + sigma * torch.randn_like(mu_x)
        else:
            z = torch.from_numpy(rng.standard_normal((n_samples, self.latent_dim))).float()
            z = z.to(self.device)
            with torch.no_grad():
                mu, log_sigma = self.decoder(z)
                sigma = torch.exp(log_sigma)
                x = mu + sigma * torch.randn_like(mu)

        cols = [f"x{i}" for i in range(self.input_dim)]
        df = pd.DataFrame(x.cpu().numpy(), columns=cols)
        assert not df.isna().any().any()
        return df

    def latent(self, X: np.ndarray) -> np.ndarray:
        """Return latent codes ``z`` for the given ``X``."""

        self.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            mask = torch.isfinite(X_t).float()
            X_t = torch.nan_to_num(X_t, nan=0.0)
            mu, log_var, _ = self.encoder(X_t, mask)
            z = self._reparameterize(mu, log_var)
        return z.cpu().numpy()

    # ------------------------------------------------------------------ metrics
    def eval_loss(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float, float, float, float, float]:
        """Return loss components on the provided data."""

        self.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        mask = torch.isfinite(X_t).float()
        X_t = torch.nan_to_num(X_t, nan=0.0)
        with torch.no_grad():
            z, recon_mu, recon_log_sigma, mu, log_var, logits = self.forward(
                X_t, mask
            )
            nll = gaussian_nll(recon_mu, recon_log_sigma, X_t, mask)
            kl = kl_divergence(mu, log_var)
            ce = F.cross_entropy(logits, y_t)
            kl_weight = kl.new_tensor(self._current_beta)
            lam = ce.new_tensor(self._current_lambda)
            if self.info_config is not None:
                info_weight = self._current_mmd
                prior_samples = torch.randn_like(z)
                mmd = maximum_mean_discrepancy(
                    z,
                    prior_samples,
                    kernel=self.info_config.kernel,
                    bandwidth=self.info_config.kernel_bandwidth,
                )
                info_penalty = ce.new_tensor(info_weight) * mmd
            else:
                info_weight = 0.0
                info_penalty = ce.new_tensor(0.0)
            loss = (nll + kl_weight * kl) + lam * ce + info_penalty
        return (
            float(loss.item()),
            float(nll.item()),
            float(kl.item()),
            float(ce.item()),
            float(info_penalty.item()),
            float(info_weight),
        )


__all__ = ["SUAVE", "AnnealSchedule", "InfoVAEConfig"]

