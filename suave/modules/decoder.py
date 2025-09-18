"""Decoder heads translating latent codes into column-wise likelihood terms."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..types import Schema
from . import distributions


class LikelihoodHead(nn.Module):
    """Base class used by the concrete distribution-specific heads."""

    param_dim: int

    def sample_from_params(
        self,
        params: Tensor,
        norm_stats: Mapping[str, float | Iterable[object]] | None,
    ) -> Dict[str, Tensor]:
        """Return samples drawn from the likelihood defined by ``params``."""

        raise NotImplementedError

    def forward(
        self,
        x: Tensor,
        params: Tensor,
        norm_stats: Mapping[str, float | Iterable[object]],
        mask: Tensor | None,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError


class RealHead(LikelihoodHead):
    """Gaussian reconstruction head mirroring HI-VAE."""

    param_dim = 2

    def forward(
        self,
        x: Tensor,
        params: Tensor,
        norm_stats: Mapping[str, float | Iterable[object]] | None,
        mask: Tensor | None,
    ) -> Dict[str, Tensor]:
        mean_raw, var_raw = params.split(1, dim=-1)
        var = F.softplus(var_raw) + distributions.EPS
        log_px = -distributions.nll_gaussian(x, mean_raw, var, mask)
        mask_missing = None if mask is None else 1.0 - mask
        log_px_missing = -distributions.nll_gaussian(x, mean_raw, var, mask_missing)

        mean_scale = float(norm_stats.get("mean", 0.0)) if norm_stats else 0.0
        std_scale = float(norm_stats.get("std", 1.0)) if norm_stats else 1.0
        params_out = {
            "mean": mean_raw * std_scale + mean_scale,
            "var": var * (std_scale**2),
        }
        sample = distributions.sample_gaussian(mean_raw, var)
        sample = sample * std_scale + mean_scale
        return {
            "log_px": log_px,
            "log_px_missing": log_px_missing,
            "params": params_out,
            "sample": sample,
        }

    def sample_from_params(
        self,
        params: Tensor,
        norm_stats: Mapping[str, float | Iterable[object]] | None,
    ) -> Dict[str, Tensor]:
        mean_raw, var_raw = params.split(1, dim=-1)
        var = F.softplus(var_raw) + distributions.EPS
        mean_scale = float(norm_stats.get("mean", 0.0)) if norm_stats else 0.0
        std_scale = float(norm_stats.get("std", 1.0)) if norm_stats else 1.0
        params_out = {
            "mean": mean_raw * std_scale + mean_scale,
            "var": var * (std_scale**2),
        }
        sample = distributions.sample_gaussian(mean_raw, var)
        sample = sample * std_scale + mean_scale
        return {"params": params_out, "sample": sample}


class CatHead(LikelihoodHead):
    """Categorical head producing logits for the discrete features."""

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        if n_classes <= 1:
            raise ValueError("Categorical features require at least two classes")
        self.param_dim = n_classes

    def forward(
        self,
        x: Tensor,
        params: Tensor,
        norm_stats: Mapping[str, float | Iterable[object]] | None,
        mask: Tensor | None,
    ) -> Dict[str, Tensor]:
        logits = params
        log_px = -distributions.nll_categorical(x, logits, mask)
        mask_missing = None if mask is None else 1.0 - mask
        log_px_missing = -distributions.nll_categorical(x, logits, mask_missing)
        sample_onehot = distributions.sample_categorical(logits)
        params_out = {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
        }
        sample_codes = sample_onehot.argmax(dim=-1)
        return {
            "log_px": log_px,
            "log_px_missing": log_px_missing,
            "params": params_out,
            "sample": sample_codes,
        }

    def sample_from_params(
        self,
        params: Tensor,
        norm_stats: Mapping[str, float | Iterable[object]] | None,
    ) -> Dict[str, Tensor]:
        logits = params
        sample_onehot = distributions.sample_categorical(logits)
        params_out = {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
        }
        sample_codes = sample_onehot.argmax(dim=-1)
        return {"params": params_out, "sample": sample_codes}


class PosHead(LikelihoodHead):
    """Placeholder for the positive (log-normal) likelihood head."""

    param_dim = 2

    def forward(self, *args, **kwargs):  # pragma: no cover - future work
        raise NotImplementedError(
            "Positive-valued head is not implemented in the warm-start release"
        )

    def sample_from_params(self, *args, **kwargs):  # pragma: no cover - future work
        raise NotImplementedError(
            "Positive-valued head is not implemented in the warm-start release"
        )


class CountHead(LikelihoodHead):
    """Placeholder for the count (Poisson) likelihood head."""

    param_dim = 1

    def forward(self, *args, **kwargs):  # pragma: no cover - future work
        raise NotImplementedError(
            "Count head is not implemented in the warm-start release"
        )

    def sample_from_params(self, *args, **kwargs):  # pragma: no cover - future work
        raise NotImplementedError(
            "Count head is not implemented in the warm-start release"
        )


class OrdinalHead(LikelihoodHead):
    """Placeholder for the ordinal cumulative link head."""

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.param_dim = n_classes

    def forward(self, *args, **kwargs):  # pragma: no cover - future work
        raise NotImplementedError(
            "Ordinal head is not implemented in the warm-start release"
        )

    def sample_from_params(self, *args, **kwargs):  # pragma: no cover - future work
        raise NotImplementedError(
            "Ordinal head is not implemented in the warm-start release"
        )


class Decoder(nn.Module):
    """Shared backbone projecting latent variables into per-column heads."""

    def __init__(
        self,
        latent_dim: int,
        schema: Schema,
        *,
        hidden: Iterable[int] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.schema = schema
        layers: list[nn.Module] = []
        previous_dim = latent_dim
        for hidden_dim in hidden:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        self._feature_types: dict[str, str] = {}
        self.param_projections = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        for column in schema.feature_names:
            spec = schema[column]
            if spec.type == "real":
                head = RealHead()
            elif spec.type == "cat":
                assert spec.n_classes is not None
                head = CatHead(spec.n_classes)
            elif spec.type == "pos":
                head = PosHead()
            elif spec.type == "count":
                head = CountHead()
            elif spec.type == "ordinal":
                assert spec.n_classes is not None
                head = OrdinalHead(spec.n_classes)
            else:
                raise ValueError(
                    f"Unsupported column type '{spec.type}' for '{column}'"
                )
            self._feature_types[column] = spec.type
            self.heads[column] = head
            self.param_projections[column] = nn.Linear(previous_dim, head.param_dim)

    def forward(
        self,
        latents: Tensor,
        data: Dict[str, Dict[str, Tensor]],
        norm_stats: Mapping[str, Mapping[str, float | Iterable[object]]],
        masks: Dict[str, Dict[str, Tensor]],
    ) -> Dict[str, object]:
        """Return reconstruction statistics for every column in the schema."""

        hidden = self.backbone(latents)
        per_feature: dict[str, Dict[str, Tensor]] = {}
        log_px_terms: list[Tensor] = []
        log_px_missing_terms: list[Tensor] = []
        for column, head in self.heads.items():
            feature_type = self._feature_types[column]
            if feature_type == "real":
                x = data["real"][column]
                mask = masks["real"][column]
            elif feature_type == "cat":
                x = data["cat"][column]
                mask = masks["cat"][column]
            else:
                # Heads for the unsupported types are placeholders; the forward
                # pass should never reach this branch in the current release.
                raise NotImplementedError(
                    f"Column type '{feature_type}' is not enabled in warm-start training"
                )
            params = self.param_projections[column](hidden)
            head_output = head(
                x=x,
                params=params,
                norm_stats=norm_stats.get(column, {}),
                mask=mask,
            )
            per_feature[column] = head_output
            log_px_terms.append(head_output["log_px"])
            log_px_missing_terms.append(head_output["log_px_missing"])
        return {
            "per_feature": per_feature,
            "log_px": log_px_terms,
            "log_px_missing": log_px_missing_terms,
        }

    def sample(
        self,
        latents: Tensor,
        norm_stats: Mapping[str, Mapping[str, float | Iterable[object]]],
    ) -> Dict[str, Dict[str, Tensor]]:
        """Generate samples for every feature given ``latents``."""

        hidden = self.backbone(latents)
        samples: dict[str, Tensor] = {}
        params: dict[str, Dict[str, Tensor]] = {}
        for column, head in self.heads.items():
            params_tensor = self.param_projections[column](hidden)
            stats = norm_stats.get(column, {})
            generated = head.sample_from_params(params_tensor, stats)
            samples[column] = generated["sample"]
            params[column] = generated["params"]
        return {"samples": samples, "params": params}
