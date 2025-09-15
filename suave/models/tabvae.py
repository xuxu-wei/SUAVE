"""Tabular VAE with a classification head (Mode 0 generation).

The implementation follows the high level blueprint described in ``AGENTS.md``.
It currently supports **continuous** inputs with optional missing values and
exposes a compact API used throughout the tests.  The focus is on clarity rather
than raw performance and only a subset of the full project features (e.g.
categorical embeddings) are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.losses import gaussian_nll, kl_divergence, linear_anneal


@dataclass
class AnnealSchedule:
    """Linear weight schedule used for KL and classification terms."""

    start: float
    end: float
    epochs: int


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


class TabVAEClassifier(nn.Module):
    """Tabular VAE with an attached classifier.

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
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        num_classes: int = 2,
        dropout: float = 0.3,
        beta_schedule: AnnealSchedule | None = None,
        lambda_schedule: AnnealSchedule | None = None,
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

        self.to(self.device)

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
    ) -> "TabVAEClassifier":
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
            beta = linear_anneal(epoch, self.beta_schedule)
            lam = linear_anneal(epoch, self.lambda_schedule)
            loss = (nll + beta * kl) + lam * ce
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
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
            nll = gaussian_nll(recon_mu, recon_log_sigma, X_t, mask).item()
            kl = kl_divergence(mu, log_var).item()
            ce = F.cross_entropy(logits, y_t).item()
            loss = nll + kl + ce
        return float(loss), float(nll), float(kl), float(ce), 0.0, 0.0


__all__ = ["TabVAEClassifier"]

