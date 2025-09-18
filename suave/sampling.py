"""Sampling utilities translating decoder outputs into tabular data."""

from __future__ import annotations

import warnings
from typing import Dict, Mapping

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .types import Schema


def _initialise_feature_dict() -> Dict[str, Dict[str, Tensor]]:
    """Return an empty nested dictionary keyed by schema feature types."""

    return {"real": {}, "pos": {}, "count": {}, "cat": {}, "ordinal": {}}


def build_placeholder_batches(
    feature_layout: Mapping[str, Mapping[str, int]],
    n_samples: int,
    *,
    device: torch.device,
) -> tuple[Dict[str, Dict[str, Tensor]], Dict[str, Dict[str, Tensor]]]:
    """Return zero-filled tensors matching the decoder's feature layout."""

    data_tensors = _initialise_feature_dict()
    mask_tensors = _initialise_feature_dict()

    for feature_type, columns in feature_layout.items():
        for column, width in columns.items():
            if feature_type in {"real", "pos", "count"}:
                shape = (n_samples, 1)
            else:
                shape = (n_samples, int(width))
            data_tensors[feature_type][column] = torch.zeros(shape, device=device)
            mask_tensors[feature_type][column] = torch.ones(
                (n_samples, 1), device=device
            )
    return data_tensors, mask_tensors


def _categorical_from_codes(
    codes: np.ndarray,
    categories: list[object] | None,
    *,
    ordered: bool = False,
    column: str | None = None,
    expected_n_classes: int | None = None,
) -> pd.Categorical:
    """Return a categorical series from integer ``codes`` and ``categories``."""

    max_code = int(codes.max(initial=-1))
    inferred_classes = max_code + 1 if max_code >= 0 else 0
    target_size = max(inferred_classes, expected_n_classes or 0)

    if categories is None:
        categories_list = list(range(target_size)) if target_size > 0 else []
    else:
        categories_list = list(categories)
        if len(categories_list) < target_size:
            missing = target_size - len(categories_list)
            warnings.warn(
                (
                    "Column '%s' produced %d unseen categorical levels during sampling; "
                    "extending the category list with numeric placeholders."
                )
                % (column or "<unknown>", missing),
                RuntimeWarning,
                stacklevel=2,
            )
            categories_list.extend(range(len(categories_list), target_size))
    if not categories_list:
        return pd.Categorical([], categories=categories_list, ordered=ordered)
    return pd.Categorical.from_codes(codes, categories=categories_list, ordered=ordered)


def decoder_outputs_to_frame(
    per_feature: Mapping[str, Mapping[str, Tensor]],
    schema: Schema,
    norm_stats: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Convert decoder samples into a :class:`pandas.DataFrame`."""

    data: Dict[str, object] = {}
    for column in schema.feature_names:
        spec = schema[column]
        feature_output = per_feature[column]
        sample_tensor = feature_output["sample"].detach().cpu()
        if spec.type in {"real", "pos"}:
            values = sample_tensor.squeeze(-1).numpy().astype(np.float32, copy=False)
            data[column] = values
        elif spec.type == "count":
            values = sample_tensor.squeeze(-1).round().clamp(min=0)
            data[column] = values.to(torch.int64).cpu().numpy()
        elif spec.type == "cat":
            codes = sample_tensor.to(torch.int64).cpu().numpy()
            categories = norm_stats.get(column, {}).get("categories")
            data[column] = _categorical_from_codes(
                codes,
                categories,
                column=column,
                expected_n_classes=spec.n_classes,
            )
        elif spec.type == "ordinal":
            thermometer = sample_tensor
            levels = (thermometer > 0.5).sum(dim=-1)
            codes = torch.clamp(levels - 1, min=0).to(torch.int64).cpu().numpy()
            categories = norm_stats.get(column, {}).get("categories")
            data[column] = _categorical_from_codes(
                codes,
                categories,
                ordered=True,
                column=column,
                expected_n_classes=spec.n_classes,
            )
        else:
            raise ValueError(f"Unsupported feature type '{spec.type}' for '{column}'")
    return pd.DataFrame(data)
