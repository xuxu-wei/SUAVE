"""Tests for dataset-grouped bootstrap report export utilities."""

from pathlib import Path
import sys

import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "research_template"))

from cls_eval import export_dataset_grouped_tables


def test_export_dataset_grouped_tables(tmp_path):
    """Ensure dataset-grouped export merges dataset cells and formats CIs."""

    df = pd.DataFrame(
        {
            "Target": ["demo"] * 3,
            "Model": ["SUAVE", "LogReg", "Clinical"],
            "Dataset": ["Test", "Test", "External"],
            "accuracy": [0.9, 0.85, 0.8],
            "accuracy_ci_low": [0.88, 0.83, 0.79],
            "accuracy_ci_high": [0.92, 0.87, 0.81],
            "roc_auc": [0.95, 0.9, 0.85],
            "roc_auc_ci_low": [0.93, 0.88, 0.83],
            "roc_auc_ci_high": [0.97, 0.92, 0.87],
        }
    )

    output_path = tmp_path / "report.xlsx"
    result = export_dataset_grouped_tables(
        {"Summary": df},
        output_path,
        index_columns={"Summary": ["Model"]},
        dataset_column="Dataset",
        dataset_order=["Test", "External"],
        drop_columns=("Target",),
        ci_label_text="95%",
        ci_only=True,
    )

    assert result == output_path
    assert output_path.exists()

    workbook = load_workbook(output_path)
    sheet = workbook["Summary"]

    assert sheet["A1"].value == "Dataset"
    assert sheet["B1"].value == "Model"
    assert sheet["C1"].value == "accuracy (95% CI)"
    assert sheet["D1"].value == "roc_auc (95% CI)"

    assert sheet["A2"].value == "Test"
    assert sheet["B2"].value == "LogReg"
    assert sheet["C2"].value == "0.850 (0.830–0.870)"
    assert sheet["D2"].value == "0.900 (0.880–0.920)"

    assert sheet["B3"].value == "SUAVE"
    assert sheet["C3"].value == "0.900 (0.880–0.920)"
    assert sheet["D3"].value == "0.950 (0.930–0.970)"

    # Second row shares merged dataset cell with first row.
    merged_ranges = {str(rng) for rng in sheet.merged_cells.ranges}
    assert "A2:A3" in merged_ranges

    # External dataset appears last with its own CI string.
    assert sheet["A4"].value == "External"
    assert sheet["C4"].value == "0.800 (0.790–0.810)"
