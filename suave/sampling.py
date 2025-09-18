"""Sampling utilities decoding latent variables into tabular samples."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from torch import Tensor

from .modules.decoder import Decoder
from .types import Schema


def _empty_frame(
    schema: Schema,
    norm_stats: Mapping[str, Mapping[str, float | Sequence[object]]],
) -> pd.DataFrame:
    """Return an empty dataframe that preserves schema column types."""

    data: dict[str, pd.Series] = {}
    for column in schema.feature_names:
        spec = schema[column]
        if spec.type == "real":
            data[column] = pd.Series(dtype=np.float32)
        elif spec.type == "cat":
            categories = norm_stats.get(column, {}).get("categories", [])
            categorical = pd.Categorical([], categories=categories)
            data[column] = pd.Series(categorical)
        else:  # pragma: no cover - defensive: unsupported heads disabled upstream
            data[column] = pd.Series(dtype=np.float32)
    return pd.DataFrame(data)


def decode_from_latents(
    *,
    decoder: Decoder,
    latents: Tensor,
    schema: Schema,
    norm_stats: Mapping[str, Mapping[str, float | Sequence[object]]],
) -> pd.DataFrame:
    """Decode ``latents`` with ``decoder`` into a pandas dataframe."""

    if latents.ndim != 2:
        raise ValueError("latents must be a 2D tensor")
    n_samples = latents.size(0)
    if n_samples == 0:
        return _empty_frame(schema, norm_stats)

    decoded = decoder.sample(latents, norm_stats)
    samples = decoded["samples"]

    columns: dict[str, pd.Series] = {}
    for column in schema.feature_names:
        spec = schema[column]
        column_tensor = samples[column].detach().cpu()
        if spec.type == "real":
            values = (
                column_tensor.reshape(n_samples, -1)
                .squeeze(-1)
                .numpy()
                .astype(np.float32)
            )
            columns[column] = pd.Series(values, dtype=np.float32)
        elif spec.type == "cat":
            codes = column_tensor.numpy().astype(np.int64)
            categories = norm_stats.get(column, {}).get("categories", [])
            if categories:
                mapped: list[object | None] = []
                for code in codes.tolist():
                    if 0 <= code < len(categories):
                        mapped.append(categories[code])
                    else:
                        mapped.append(None)
                categorical = pd.Categorical(mapped, categories=categories)
            else:
                categorical = pd.Categorical(codes.tolist())
            columns[column] = pd.Series(categorical)
        else:  # pragma: no cover - unsupported types not enabled yet
            raise NotImplementedError(
                f"Sampling for column type '{spec.type}' is not implemented"
            )

    frame = pd.DataFrame(columns)
    return frame
