"""Encoder network translating inputs into latent Gaussian parameters."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


class EncoderMLP(nn.Module):
    r"""Multi-layer perceptron producing the mean and log-variance of ``z``.

    The architecture mirrors the TensorFlow HI-VAE baseline: a stack of dense
    layers followed by two linear heads generating the parameters of the
    approximate posterior :math:`q(z \mid x)`.  The log-variance output is
    clamped for numerical stability, matching the behaviour of the reference
    implementation.

    Parameters
    ----------
    input_dim:
        Dimensionality of the flattened input vector fed to the encoder.
    latent_dim:
        Size of the latent space to model.
    hidden:
        Iterable containing the hidden layer sizes.
    dropout:
        Dropout probability applied after each activation layer.
    """

    LOGVAR_RANGE = (-15.0, 15.0)

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        hidden: Iterable[int] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.mu_layer = nn.Linear(previous_dim, latent_dim)
        self.logvar_layer = nn.Linear(previous_dim, latent_dim)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return the posterior mean and log-variance for ``inputs``."""

        hidden = self.backbone(inputs)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        min_val, max_val = self.LOGVAR_RANGE
        logvar = torch.clamp(logvar, min=min_val, max=max_val)
        return mu, logvar
