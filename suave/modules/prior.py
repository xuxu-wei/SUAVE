"""Prior modules defining HI-VAE specific parameterisations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PriorMean(nn.Module):
    """Linear layer producing component-wise Gaussian means from assignments."""

    def __init__(self, n_components: int, latent_dim: int) -> None:
        super().__init__()
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        self.n_components = int(n_components)
        self.latent_dim = int(latent_dim)
        self.linear = nn.Linear(self.n_components, self.latent_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, assignments: Tensor) -> Tensor:
        """Return the Gaussian means associated with ``assignments``."""

        if assignments.size(-1) != self.n_components:
            raise ValueError("assignments must have dimension matching n_components")
        return self.linear(assignments)

    def component_means(self) -> Tensor:
        r"""Return the mean of :math:`p(z \mid s)` for each mixture component."""

        eye = torch.eye(
            self.n_components,
            dtype=self.linear.weight.dtype,
            device=self.linear.weight.device,
        )
        return self.forward(eye)

    def load_component_means(self, means: Tensor) -> None:
        """Initialise the linear layer so that components map to ``means``."""

        if means.shape != (self.n_components, self.latent_dim):
            raise ValueError("means must have shape (n_components, latent_dim)")
        means = means.to(
            dtype=self.linear.weight.dtype, device=self.linear.weight.device
        )
        with torch.no_grad():
            self.linear.bias.zero_()
            self.linear.weight.copy_(means.t())
