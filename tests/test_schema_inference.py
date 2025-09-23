"""Tests for the schema inference helper."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from suave.schema_inference import (
    InferenceConfidence,
    SchemaInferenceMode,
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
    result = inferencer.infer(df, mode=SchemaInferenceMode.SILENT)
    schema_dict = result.schema.to_dict()

    assert schema_dict["age"]["type"] == "real"
    assert schema_dict["sex"]["type"] == "cat"
    assert schema_dict["count_feature"]["type"] == "count"
    assert result.mode is SchemaInferenceMode.SILENT
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
    result = inferencer.infer(df, mode=SchemaInferenceMode.INFO)

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

    result = inferencer.infer(df, mode=SchemaInferenceMode.INTERACTIVE)

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

    result = inferencer.infer(df, mode=SchemaInferenceMode.INTERACTIVE)

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
    result = inferencer.infer(df, mode=SchemaInferenceMode.INFO)

    confidences = result.column_confidence
    assert confidences["binary"] is InferenceConfidence.HIGH
    assert confidences["converted"] is InferenceConfidence.MEDIUM
    assert confidences["mixed"] is InferenceConfidence.LOW


def test_integer_inference_handles_wide_range() -> None:
    wide_range = [0, 10_000_000, 20_000_000]
    df = pd.DataFrame({"wide_range_ints": wide_range})

    inferencer = SchemaInferencer()
    result = inferencer.infer(df, mode=SchemaInferenceMode.SILENT)
    schema_dict = result.schema.to_dict()

    assert schema_dict["wide_range_ints"]["type"] == "count"


def test_long_integer_ladder_prefers_count_over_categorical() -> None:
    repeated_cycle = list(range(16)) * 50  # unique ratio below categorical threshold
    df = pd.DataFrame({"cycled_counts": repeated_cycle})

    inferencer = SchemaInferencer()
    result = inferencer.infer(df, mode=SchemaInferenceMode.SILENT)

    assert result.schema.to_dict()["cycled_counts"] == {"type": "count"}
