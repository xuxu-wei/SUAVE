"""Encoder network translating inputs into latent Gaussian parameters."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


class EncoderMLP(nn.Module):
    r"""Multi-layer perceptron producing the mean and log-variance of ``z``.

    The architecture mirrors the legacy TensorFlow unsupervised baseline: a stack of dense
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
        n_components: int = 1,
    ) -> None:
        super().__init__()
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = int(n_components)
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        projection_dim = latent_dim * self.n_components
        self.component_logits = nn.Linear(previous_dim, self.n_components)
        self.mu_layer = nn.Linear(previous_dim, projection_dim)
        self.logvar_layer = nn.Linear(previous_dim, projection_dim)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return mixture logits alongside per-component Gaussian parameters."""

        hidden = self.backbone(inputs)
        logits = self.component_logits(hidden)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        min_val, max_val = self.LOGVAR_RANGE
        logvar = torch.clamp(logvar, min=min_val, max=max_val)
        batch_size = inputs.size(0)
        mu = mu.view(batch_size, self.n_components, -1)
        logvar = logvar.view(batch_size, self.n_components, -1)
        return logits, mu, logvar
