"""Tests for the optional schema builder module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from suave.interactive import schema_builder
from suave.interactive.schema_builder import (
    SchemaBuilderError,
    _coerce_optional_positive_int,
    _distribution_payload,
    _summarise_series,
)
from suave.types import Schema


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, np.nan, 40],
            "sex": ["F", "M", "F", "M"],
        }
    )


def test_coerce_optional_positive_int_accepts_positive_numbers() -> None:
    assert _coerce_optional_positive_int(3) == 3
    assert _coerce_optional_positive_int("4") == 4
    assert _coerce_optional_positive_int(None) is None
    assert _coerce_optional_positive_int("") is None


@pytest.mark.parametrize("value", [0, -1, "-3", False])
def test_coerce_optional_positive_int_rejects_invalid(value: object) -> None:
    with pytest.raises(ValueError):
        _coerce_optional_positive_int(value)


def test_summarise_series_returns_expected_stats(sample_frame: pd.DataFrame) -> None:
    summary = _summarise_series(sample_frame["age"])
    assert summary["dtype"] == "float64"
    assert summary["nunique"] == 3
    assert summary["missing"] == 1
    assert len(summary["sample"]) == 4


def test_distribution_payload_numeric(sample_frame: pd.DataFrame) -> None:
    payload = _distribution_payload(sample_frame["age"], "age")
    assert payload["type"] == "hist"
    assert payload["column"] == "age"
    assert len(payload["counts"]) > 0


def test_distribution_payload_categorical(sample_frame: pd.DataFrame) -> None:
    payload = _distribution_payload(sample_frame["sex"], "sex")
    assert payload["type"] == "bar"
    assert payload["labels"]
    assert all(isinstance(label, str) for label in payload["labels"])


def test_distribution_payload_empty_series() -> None:
    series = pd.Series([np.nan, np.nan])
    payload = _distribution_payload(series, "empty")
    assert payload["type"] == "empty"
    assert "message" in payload


def test_launch_schema_builder_requires_flask(monkeypatch, sample_frame: pd.DataFrame) -> None:
    monkeypatch.setattr(schema_builder, "Flask", None)
    monkeypatch.setattr(schema_builder, "jsonify", None)
    monkeypatch.setattr(schema_builder, "request", None)
    monkeypatch.setattr(schema_builder, "make_server", None)

    with pytest.raises(SchemaBuilderError):
        schema_builder.launch_schema_builder(sample_frame, inferencer=None, open_browser=False)


def test_schema_builder_preserves_initial_schema(monkeypatch, sample_frame: pd.DataFrame) -> None:
    if schema_builder.Flask is None:
        pytest.skip("Flask is not available; integration test skipped.")

    class DummyBuilder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self) -> Schema:
            return Schema({"age": {"type": "real"}, "sex": {"type": "cat", "n_classes": 2}})

    monkeypatch.setattr(schema_builder, "_SchemaBuilder", DummyBuilder)
    schema = schema_builder.launch_schema_builder(sample_frame, open_browser=False)
    assert isinstance(schema, Schema)
    assert schema.to_dict()["sex"]["n_classes"] == 2
