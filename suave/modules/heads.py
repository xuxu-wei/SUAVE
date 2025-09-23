"""Classification heads used by the SUAVE classifier stack."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ClassificationHead(nn.Module):
    """Linear classifier operating on latent representations.

    Parameters
    ----------
    in_features:
        Dimensionality of the latent vectors produced by the encoder.
    n_classes:
        Number of target classes.
    class_weight:
        Optional weighting applied to the cross-entropy objective.  The
        semantics follow :func:`torch.nn.functional.cross_entropy`.
    dropout:
        Dropout probability applied after each hidden layer (and directly on
        the inputs when no hidden layers are requested). ``0.0`` disables
        dropout.
    hidden_dims:
        Optional hidden layer widths inserted before the ``n_classes``
        projection. Each hidden layer follows a ``Linear → ReLU → Dropout``
        pattern.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        *,
        class_weight: Iterable[float] | Sequence[float] | Tensor | None = None,
        dropout: float = 0.0,
        hidden_dims: Iterable[int] = (),
    ) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError("Classification head requires at least two classes")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in the interval [0, 1)")
        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")

        layers: list[nn.Module] = []
        previous_features = int(in_features)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_features, hidden_dim))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(float(dropout)))
            previous_features = hidden_dim

        self.hidden_layers = nn.Sequential(*layers) if layers else None
        self._final_dropout = (
            nn.Dropout(float(dropout)) if dropout and not hidden_dims else None
        )
        self.linear = nn.Linear(previous_features, n_classes)
        if class_weight is not None:
            weight_tensor = torch.as_tensor(class_weight, dtype=torch.float32)
            if weight_tensor.numel() != n_classes:
                raise ValueError(
                    "class_weight must provide a value for every target class"
                )
            self.register_buffer("class_weight", weight_tensor)
            self._use_weight = True
        else:
            self.register_buffer("class_weight", torch.ones(n_classes))
            self._use_weight = False

    def forward(self, latents: Tensor) -> Tensor:
        """Return unnormalised logits for ``latents``."""

        if self.hidden_layers is not None:
            latents = self.hidden_layers(latents)
        elif self._final_dropout is not None:
            latents = self._final_dropout(latents)
        return self.linear(latents)

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Cross-entropy objective respecting optional class weights."""

        weight = self.class_weight if self._use_weight else None
        return F.cross_entropy(logits, targets, weight=weight)
