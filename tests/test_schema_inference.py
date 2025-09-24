"""Tests for the schema inference helper."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from suave.schema_inference import (
    InferenceConfidence,
    SchemaInferencer,
)
from suave.types import Schema


def test_infer_silent_mode_returns_schema_only() -> None:
    df = pd.DataFrame(
        {
            "age": np.linspace(10, 80, num=50),
            "sex": ["M", "F"] * 25,
            "count_feature": [0, 1, 2, 3, 4] * 10,
        }
    )
    inferencer = SchemaInferencer(categorical_overrides={"sex"})
    result = inferencer.infer(df, mode="silent")
    schema_dict = result.schema.to_dict()

    assert schema_dict["age"]["type"] == "real"
    assert schema_dict["sex"]["type"] == "cat"
    assert schema_dict["count_feature"]["type"] == "count"
    assert result.mode == "silent"
    assert result.messages == []


def test_info_mode_highlights_columns_near_threshold() -> None:
    repeated_pattern = [0, 1, 2] * 20  # unique ratio close to categorical threshold
    df = pd.DataFrame(
        {
            "possibly_categorical": repeated_pattern,
            "continuous": np.linspace(0.0, 1.0, num=len(repeated_pattern)),
        }
    )
    inferencer = SchemaInferencer()
    result = inferencer.infer(df, mode="info")

    assert "possibly_categorical" in result.review_columns
    assert any("possibly_categorical" in message for message in result.messages)


def test_interactive_mode_gracefully_falls_back(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "flagged": [0, 1] + [0] * 38,
        }
    )
    inferencer = SchemaInferencer()
    monkeypatch.setattr(
        SchemaInferencer,
        "_can_launch_gui",
        staticmethod(lambda: False),
    )
    fallback_message = (
        "Browser-based schema builder unavailable; falling back to Matplotlib review."
    )
    monkeypatch.setattr(
        SchemaInferencer,
        "_try_browser_schema_builder",
        lambda self, df, columns: (None, fallback_message),
        raising=False,
    )

    result = inferencer.infer(df, mode="interactive")

    assert "flagged" in result.review_columns
    assert any(fallback_message in message for message in result.messages)
    assert any("Interactive review not available" in message for message in result.messages)



def test_interactive_mode_prefers_browser_builder(monkeypatch) -> None:
    df = pd.DataFrame({"flagged": [0, 1, 0, 1]})
    inferencer = SchemaInferencer()
    custom_schema = Schema({"flagged": {"type": "pos"}})

    monkeypatch.setattr(
        SchemaInferencer,
        "_try_browser_schema_builder",
        lambda self, df, columns: (custom_schema, None),
        raising=False,
    )

    result = inferencer.infer(df, mode="interactive")

    assert result.schema.to_dict()["flagged"]["type"] == "pos"
    assert result.review_columns == []
    assert any(
        "Browser-based schema builder applied" in message for message in result.messages
    )


def test_high_cardinality_float_skips_unique(monkeypatch) -> None:
    inferencer = SchemaInferencer()
    high_cardinality = pd.Series(np.linspace(-500.5, 499.5, num=2048) + 0.123456)

    module = importlib.import_module(SchemaInferencer.__module__)
    call_counter = {"count": 0}
    original_unique = module.np.unique

    def tracking_unique(*args, **kwargs):
        call_counter["count"] += 1
        return original_unique(*args, **kwargs)

    monkeypatch.setattr(module.np, "unique", tracking_unique)

    spec, notes, confidence = inferencer._infer_numeric_schema(high_cardinality)

    assert spec == {"type": "real"}
    assert notes == ""
    assert confidence is InferenceConfidence.HIGH
    assert call_counter["count"] == 0


def test_infer_reports_column_confidence_levels() -> None:
    df = pd.DataFrame(
        {
            "binary": [0, 1] * 10,
            "converted": [str(i) for i in range(19)] + ["oops"],
            "mixed": list("abcdefghij") * 2,
        }
    )

    inferencer = SchemaInferencer()
    result = inferencer.infer(df, mode="info")

    confidences = result.column_confidence
    assert confidences["binary"] is InferenceConfidence.HIGH
    assert confidences["converted"] is InferenceConfidence.MEDIUM
    assert confidences["mixed"] is InferenceConfidence.LOW


def test_integer_inference_handles_wide_range() -> None:
    wide_range = [0, 10_000_000, 20_000_000]
    df = pd.DataFrame({"wide_range_ints": wide_range})

    inferencer = SchemaInferencer()
    result = inferencer.infer(df, mode="silent")
    schema_dict = result.schema.to_dict()

    assert schema_dict["wide_range_ints"]["type"] == "count"


def test_long_integer_ladder_prefers_count_over_categorical() -> None:
    repeated_cycle = list(range(16)) * 50  # unique ratio below categorical threshold
    df = pd.DataFrame({"cycled_counts": repeated_cycle})

    inferencer = SchemaInferencer()
    result = inferencer.infer(df, mode="silent")

    assert result.schema.to_dict()["cycled_counts"] == {"type": "count"}


def test_low_cardinality_integer_with_wide_dispersion_prefers_real() -> None:
    repeated = np.tile(np.array([0.0, 100.0, 200.0, 300.0]), 25)
    series = pd.Series(repeated)

    inferencer = SchemaInferencer()
    spec, note, confidence = inferencer._infer_numeric_schema(series)

    assert spec == {"type": "real"}
    assert "Dispersion suggests continuous behaviour despite low cardinality." in note
    assert confidence is InferenceConfidence.MEDIUM


def test_low_dispersion_float_prefers_categorical() -> None:
    tiny_variation = np.repeat(np.array([0.1, 0.1000005]), 50)
    series = pd.Series(tiny_variation)

    inferencer = SchemaInferencer()
    spec, note, confidence = inferencer._infer_numeric_schema(series)

    assert spec == {"type": "cat", "n_classes": 2}
    assert note == "Dispersion too small; defaulting to categorical."
    assert confidence is InferenceConfidence.LOW


def test_invalid_mode_raises_value_error() -> None:
    df = pd.DataFrame({"feature": [1, 2, 3]})
    inferencer = SchemaInferencer()

    with pytest.raises(ValueError, match="mode must be one of"):
        inferencer.infer(df, mode="invalid")
