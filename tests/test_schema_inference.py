"""Tests for the schema inference helper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from suave.schema_inference import SchemaInferenceMode, SchemaInferencer


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
    result = inferencer.infer(df, mode=SchemaInferenceMode.INTERACTIVE)

    assert "flagged" in result.review_columns
    assert any("Interactive review not available" in message for message in result.messages)
