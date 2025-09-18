"""Minimal end-to-end example for the placeholder SUAVE package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from suave import SUAVE, Schema
from suave.plots import plot_reliability_curve


def main() -> None:
    """Run the minimal training, prediction, calibration, and plotting loop."""

    data_path = Path(__file__).parent / "data" / "sepsis_toy.csv"
    df = pd.read_csv(data_path)

    feature_columns = ["age", "sofa_score", "gender"]
    schema = Schema(
        {
            "age": {"type": "real"},
            "sofa_score": {"type": "real"},
            "gender": {"type": "cat", "n_classes": 2},
        }
    )

    X = df[feature_columns]
    y = df["outcome"]

    model = SUAVE(schema=schema)
    model.fit(X, y, epochs=1)
    probabilities = model.predict_proba(X)
    model.calibrate(X, y)

    output_path = data_path.with_name("reliability_placeholder.png")
    plot_reliability_curve(probabilities[:, 1], y, output_path=output_path)
    print(f"Reliability curve saved to {output_path}")


if __name__ == "__main__":
    main()
