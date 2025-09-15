from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Simple MLP encoder producing mean and log-variance."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, latent_dim)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.fc(x))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    """Simple MLP decoder reconstructing the input."""

    def __init__(self, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, latent_dim)
        self.out = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc(z))
        return self.out(h)


class Classifier(nn.Module):
    """Linear classifier on top of latent variables."""

    def __init__(self, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


class TabVAEClassifier(nn.Module):
    """A lightweight VAE with an attached classification head.

    The implementation is intentionally small to serve as a stand-in for the
    full model described in the project blueprint. It supports basic fitting,
    prediction and data generation functionality required by the tests.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        num_classes: int = 2,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device or torch.device("cpu")

        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.classifier = Classifier(latent_dim, num_classes)
        self.to(self.device)

    # ------------------------------ core utils -----------------------------
    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self._reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        logits = self.classifier(z)
        return x_hat, mu, logvar, logits

    # ------------------------------ public API -----------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, lr: float = 1e-3) -> "TabVAEClassifier":
        """Train the model on ``X`` and ``y``.

        The training routine is deliberately simple and operates on the full
        dataset without mini-batching to keep the runtime short for tests.
        """

        self.train()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            x_hat, mu, logvar, logits = self.forward(X_t)
            recon = F.mse_loss(x_hat, X_t)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            ce = F.cross_entropy(logits, y_t)
            loss = recon + kl + ce
            loss.backward()
            opt.step()
        return self

    def predict_logits(self, X: np.ndarray) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            mu, _ = self.encoder(X_t)
            logits = self.classifier(mu)
        return logits.cpu()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(X)
        return torch.softmax(logits, dim=-1).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)

    def generate(self, n_samples: int, conditional: Optional[dict[str, float]] = None, seed: Optional[int] = None) -> pd.DataFrame:
        """Generate ``n_samples`` synthetic rows.

        Generation ignores ``conditional`` as this lightweight implementation
        does not support conditional sampling.
        """
        rng = np.random.default_rng(seed)
        z = torch.from_numpy(rng.standard_normal((n_samples, self.latent_dim))).float()
        with torch.no_grad():
            x_hat = self.decoder(z.to(self.device)).cpu().numpy()
        cols = [f"x{i}" for i in range(self.input_dim)]
        return pd.DataFrame(x_hat, columns=cols)

    def eval_loss(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float, float]:
        """Evaluate reconstruction and classification losses on the data."""
        self.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        with torch.no_grad():
            x_hat, mu, logvar, logits = self.forward(X_t)
            recon = F.mse_loss(x_hat, X_t).item()
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()).item()
            ce = F.cross_entropy(logits, y_t).item()
            loss = recon + kl + ce
        return float(loss), float(recon), float(kl), float(ce), 0.0, 0.0
