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

    def __init__(self, y_dim: int, n_components: int) -> None:
        super().__init__()
        self.y_dim = int(y_dim)
        self.n_components = int(n_components)

    def forward(
        self,
        y: Tensor,
        s: Tensor,
        x: Tensor,
        norm_stats: Mapping[str, float | Iterable[object]],
        mask: Tensor | None,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError


def _mask_to_bool(mask: Tensor) -> Tensor:
    mask_bool = mask
    if mask_bool.dim() > 1:
        mask_bool = mask_bool.squeeze(-1)
    return mask_bool > 0.5


def _apply_observed_linear(
    layer: nn.Linear, features: Tensor, mask: Tensor | None
) -> Tensor:
    if mask is None:
        return layer(features)
    mask_bool = _mask_to_bool(mask)
    output = features.new_zeros(features.size(0), layer.out_features)
    if mask_bool.any():
        output[mask_bool] = layer(features[mask_bool])
    if (~mask_bool).any():
        with torch.no_grad():
            output[~mask_bool] = layer(features[~mask_bool])
    return output


class RealHead(LikelihoodHead):
    """Gaussian reconstruction head mirroring the HI-VAE design."""

    def __init__(self, y_dim: int, n_components: int) -> None:
        super().__init__(y_dim, n_components)
        self.mean_layer = nn.Linear(self.y_dim + self.n_components, 1, bias=False)
        self.var_layer = nn.Linear(self.n_components, 1, bias=False)

    def forward(
        self,
        y: Tensor,
        s: Tensor,
        x: Tensor,
        norm_stats: Mapping[str, float | Iterable[object]] | None,
        mask: Tensor | None,
    ) -> Dict[str, Tensor]:
        features = torch.cat([y, s], dim=-1)
        mean_raw = _apply_observed_linear(self.mean_layer, features, mask)
        var_raw = _apply_observed_linear(self.var_layer, s, mask)
        var = F.softplus(var_raw) + distributions.EPS

        mean_scale = float(norm_stats.get("mean", 0.0)) if norm_stats else 0.0
        std_scale = float(norm_stats.get("std", 1.0)) if norm_stats else 1.0
        mean = mean_raw * std_scale + mean_scale
        var_scaled = var * (std_scale**2)

        x_raw = x * std_scale + mean_scale
        log_px = -distributions.nll_gaussian(x_raw, mean, var_scaled, mask)
        mask_missing = None if mask is None else 1.0 - mask
        log_px_missing = -distributions.nll_gaussian(
            x_raw, mean, var_scaled, mask_missing
        )

        params_out = {"mean": mean, "var": var_scaled}
        sample = distributions.sample_gaussian(mean_raw, var)
        sample = sample * std_scale + mean_scale
        return {
            "log_px": log_px,
            "log_px_missing": log_px_missing,
            "params": params_out,
            "sample": sample,
        }


class PosHead(LikelihoodHead):
    """Log-normal reconstruction head for strictly positive features."""

    def __init__(self, y_dim: int, n_components: int) -> None:
        super().__init__(y_dim, n_components)
        self.mean_layer = nn.Linear(self.y_dim + self.n_components, 1, bias=False)
        self.var_layer = nn.Linear(self.n_components, 1, bias=False)

    def forward(
        self,
        y: Tensor,
        s: Tensor,
        x: Tensor,
        norm_stats: Mapping[str, float | Iterable[object]] | None,
        mask: Tensor | None,
    ) -> Dict[str, Tensor]:
        features = torch.cat([y, s], dim=-1)
        mean_raw = _apply_observed_linear(self.mean_layer, features, mask)
        var_raw = _apply_observed_linear(self.var_layer, s, mask)
        var = F.softplus(var_raw) + distributions.EPS

        mean_log = float(norm_stats.get("mean_log", 0.0)) if norm_stats else 0.0
        std_log = float(norm_stats.get("std_log", 1.0)) if norm_stats else 1.0

        log_x = x * std_log + mean_log
        mean = mean_raw * std_log + mean_log
        var_scaled = var * (std_log**2)

        log_px = -distributions.nll_lognormal(log_x, mean, var_scaled, mask)
        mask_missing = None if mask is None else 1.0 - mask
        log_px_missing = -distributions.nll_lognormal(
            log_x, mean, var_scaled, mask_missing
        )

        params_out = {
            "mean_log": mean,
            "var_log": var_scaled,
            "mean": torch.exp(mean + 0.5 * torch.clamp(var_scaled, min=0.0)) - 1.0,
        }
        sample = distributions.sample_lognormal(mean, var_scaled)
        return {
            "log_px": log_px,
            "log_px_missing": log_px_missing,
            "params": params_out,
            "sample": sample,
        }


class CountHead(LikelihoodHead):
    """Poisson reconstruction head for count-valued features."""

    def __init__(self, y_dim: int, n_components: int) -> None:
        super().__init__(y_dim, n_components)
        self.rate_layer = nn.Linear(self.y_dim + self.n_components, 1, bias=False)

    def forward(
        self,
        y: Tensor,
        s: Tensor,
        x: Tensor,
        norm_stats: Mapping[str, float | Iterable[object]] | None,
        mask: Tensor | None,
    ) -> Dict[str, Tensor]:
        features = torch.cat([y, s], dim=-1)
        rate_raw = _apply_observed_linear(self.rate_layer, features, mask)
        rate = F.softplus(rate_raw) + distributions.EPS
        offset = float(norm_stats.get("offset", 0.0)) if norm_stats else 0.0
        counts = torch.exp(x) - offset
        counts = torch.clamp(counts, min=0.0)
        log_px = -distributions.nll_poisson(counts, rate, mask)
        mask_missing = None if mask is None else 1.0 - mask
        log_px_missing = -distributions.nll_poisson(counts, rate, mask_missing)
        sample = distributions.sample_poisson(rate)
        params_out = {"rate": rate}
        return {
            "log_px": log_px,
            "log_px_missing": log_px_missing,
            "params": params_out,
            "sample": sample,
        }


class CatHead(LikelihoodHead):
    """Categorical head producing logits for discrete features."""

    def __init__(self, y_dim: int, n_components: int, n_classes: int) -> None:
        super().__init__(y_dim, n_components)
        if n_classes <= 1:
            raise ValueError("Categorical features require at least two classes")
        self.n_classes = int(n_classes)
        self.logits_layer = nn.Linear(
            self.y_dim + self.n_components, self.n_classes - 1, bias=False
        )

    def forward(
        self,
        y: Tensor,
        s: Tensor,
        x: Tensor,
        norm_stats: Mapping[str, float | Iterable[object]] | None,
        mask: Tensor | None,
    ) -> Dict[str, Tensor]:
        features = torch.cat([y, s], dim=-1)
        partial_logits = _apply_observed_linear(self.logits_layer, features, mask)
        zeros = torch.zeros(
            partial_logits.size(0),
            1,
            device=partial_logits.device,
            dtype=partial_logits.dtype,
        )
        logits = torch.cat([zeros, partial_logits], dim=-1)
        log_px = -distributions.nll_categorical(x, logits, mask)
        mask_missing = None if mask is None else 1.0 - mask
        log_px_missing = -distributions.nll_categorical(x, logits, mask_missing)
        sample_onehot = distributions.sample_categorical(logits)
        params_out = {"logits": logits, "probs": torch.softmax(logits, dim=-1)}
        sample_codes = sample_onehot.argmax(dim=-1)
        return {
            "log_px": log_px,
            "log_px_missing": log_px_missing,
            "params": params_out,
            "sample": sample_codes,
        }


class OrdinalHead(LikelihoodHead):
    """Cumulative link reconstruction head for ordinal features."""

    def __init__(self, y_dim: int, n_components: int, n_classes: int) -> None:
        super().__init__(y_dim, n_components)
        if n_classes <= 1:
            raise ValueError("Ordinal features require at least two classes")
        self.n_classes = int(n_classes)
        self.partition_layer = nn.Linear(
            self.n_components, self.n_classes - 1, bias=False
        )
        self.mean_layer = nn.Linear(self.y_dim + self.n_components, 1, bias=False)

    def forward(
        self,
        y: Tensor,
        s: Tensor,
        x: Tensor,
        norm_stats: Mapping[str, float | Iterable[object]] | None,
        mask: Tensor | None,
    ) -> Dict[str, Tensor]:
        partition = _apply_observed_linear(self.partition_layer, s, mask)
        mean_param = _apply_observed_linear(
            self.mean_layer, torch.cat([y, s], dim=-1), mask
        )
        probs, thresholds = distributions.ordinal_probabilities(partition, mean_param)
        thermometer = x
        levels = torch.round(thermometer.sum(dim=-1)).long()
        targets = torch.clamp(levels - 1, min=0)
        mask_vector = None if mask is None else mask.squeeze(-1)
        log_px = -distributions.nll_ordinal(probs, targets, mask_vector)
        mask_missing = None if mask_vector is None else 1.0 - mask_vector
        log_px_missing = -distributions.nll_ordinal(probs, targets, mask_missing)
        samples = distributions.sample_ordinal(probs)
        params_out = {"probs": probs, "thresholds": thresholds, "mean": mean_param}
        return {
            "log_px": log_px,
            "log_px_missing": log_px_missing,
            "params": params_out,
            "sample": samples,
        }


class Decoder(nn.Module):
    """Shared backbone projecting latent variables into per-column heads."""

    def __init__(
        self,
        latent_dim: int,
        schema: Schema,
        *,
        hidden: Iterable[int] = (256, 128),
        dropout: float = 0.1,
        n_components: int = 1,
    ) -> None:
        super().__init__()
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.schema = schema
        self.n_components = int(n_components)

        layers: list[nn.Module] = []
        previous_dim = latent_dim
        for hidden_dim in hidden:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        y_dimensions = schema.y_dimensions()
        total_y_dim = int(sum(y_dimensions.values()))
        if total_y_dim <= 0:
            raise ValueError("The latent partition must allocate positive dimensions")
        self.y_projection = nn.Linear(previous_dim, total_y_dim)

        self._feature_types: dict[str, str] = {}
        self._y_slices: dict[str, slice] = {}
        self.heads = nn.ModuleDict()

        start = 0
        for column in schema.feature_names:
            spec = schema[column]
            y_dim = int(y_dimensions[column])
            end = start + y_dim
            self._y_slices[column] = slice(start, end)
            self._feature_types[column] = spec.type
            self.heads[column] = self._build_head(spec, y_dim)
            start = end

    def _build_head(self, spec, y_dim: int) -> LikelihoodHead:
        if spec.type == "real":
            return RealHead(y_dim, self.n_components)
        if spec.type == "pos":
            return PosHead(y_dim, self.n_components)
        if spec.type == "count":
            return CountHead(y_dim, self.n_components)
        if spec.type == "cat":
            if spec.n_classes is None:
                raise ValueError("Categorical columns must define 'n_classes'")
            return CatHead(y_dim, self.n_components, spec.n_classes)
        if spec.type == "ordinal":
            if spec.n_classes is None:
                raise ValueError("Ordinal columns must define 'n_classes'")
            return OrdinalHead(y_dim, self.n_components, spec.n_classes)
        raise ValueError(f"Unsupported column type '{spec.type}' for '{spec}'")

    def forward(
        self,
        latents: Tensor,
        assignments: Tensor,
        data: Dict[str, Dict[str, Tensor]],
        norm_stats: Mapping[str, Mapping[str, float | Iterable[object]]],
        masks: Dict[str, Dict[str, Tensor]],
    ) -> Dict[str, object]:
        """Return reconstruction statistics for every column in the schema."""

        hidden = self.backbone(latents)
        y_full = self.y_projection(hidden)
        per_feature: dict[str, Dict[str, Tensor]] = {}
        log_px_terms: list[Tensor] = []
        log_px_missing_terms: list[Tensor] = []
        for column, head in self.heads.items():
            feature_type = self._feature_types[column]
            if feature_type == "real":
                x = data["real"][column]
                mask = masks["real"][column]
            elif feature_type == "pos":
                x = data["pos"][column]
                mask = masks["pos"][column]
            elif feature_type == "count":
                x = data["count"][column]
                mask = masks["count"][column]
            elif feature_type == "cat":
                x = data["cat"][column]
                mask = masks["cat"][column]
            elif feature_type == "ordinal":
                x = data["ordinal"][column]
                mask = masks["ordinal"][column]
            else:
                raise NotImplementedError(
                    f"Column type '{feature_type}' is not enabled in warm-start training"
                )
            y_slice = y_full[:, self._y_slices[column]]
            head_output = head(
                y=y_slice,
                s=assignments,
                x=x,
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
