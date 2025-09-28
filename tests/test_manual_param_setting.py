"""Tests for manual parameter tuning scaffolding."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import importlib

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


if "IPython" not in sys.modules:
    ipython_stub = types.ModuleType("IPython")
    display_stub = types.ModuleType("IPython.display")

    def _noop_display(*_: object, **__: object) -> None:  # pragma: no cover - stub
        return None

    display_stub.display = _noop_display
    ipython_stub.display = display_stub
    ipython_stub.get_ipython = lambda: None  # pragma: no cover - stub attribute
    ipython_stub.version_info = (0, 0)
    sys.modules["IPython"] = ipython_stub
    sys.modules["IPython.display"] = display_stub


@pytest.mark.parametrize(
    "module_path",
    [
        "examples.mimic_mortality_utils",
        "research_template.analysis_utils",
    ],
)
def test_prepare_analysis_output_directories_initialises_manual_script(
    tmp_path: Path, module_path: str
) -> None:
    """Ensure manual_param_setting.py is created with a default dictionary."""

    module = importlib.import_module(module_path)
    output_root = tmp_path / module_path.replace(".", "_")
    directories = module.prepare_analysis_output_directories(
        output_root, ["suave_model"]
    )

    manual_script = directories["suave_model"] / "manual_param_setting.py"
    assert manual_script.exists()
    content = manual_script.read_text(encoding="utf-8")
    assert "manual_param_setting: dict = {}" in content

    manual_script.write_text(
        "manual_param_setting: dict = {\n    'learning_rate': 0.01\n}\n",
        encoding="utf-8",
    )

    # Running the directory preparation again should keep custom content intact.
    module.prepare_analysis_output_directories(output_root, ["suave_model"])
    assert manual_script.read_text(encoding="utf-8").strip().startswith(
        "manual_param_setting: dict = {"
    )


@pytest.mark.parametrize(
    "module_path",
    [
        "examples.mimic_mortality_utils",
        "research_template.analysis_utils",
    ],
)
def test_manual_manifest_round_trip(tmp_path: Path, module_path: str) -> None:
    """Manual manifest helpers should normalise paths and persist params."""

    module = importlib.import_module(module_path)
    model_dir = tmp_path / f"{module_path.replace('.', '_')}_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "manual.pt"
    calibrator_path = model_dir / "manual_calibrator.joblib"
    model_path.write_text("model", encoding="utf-8")
    calibrator_path.write_text("calibrator", encoding="utf-8")

    manifest_path = module.record_manual_model_manifest(
        model_dir,
        "mortality",
        model_path=model_path,
        calibrator_path=calibrator_path,
        params={"learning_rate": 0.01},
    )
    assert manifest_path.exists()

    manifest = module.load_manual_model_manifest(model_dir, "mortality")
    assert manifest.get("model_path") == "manual.pt"
    assert manifest.get("params", {}).get("learning_rate") == 0.01

    resolved = module.manifest_artifact_paths(manifest, model_dir)
    assert resolved["model"] == model_path
    assert resolved["calibrator"] == calibrator_path


@pytest.mark.parametrize(
    "module_path",
    [
        "examples.mimic_mortality_utils",
        "research_template.analysis_utils",
    ],
)
def test_parse_script_arguments_accepts_manual(module_path: str) -> None:
    """The CLI parser should accept the manual override keyword."""

    module = importlib.import_module(module_path)
    assert module.parse_script_arguments(["manual"]) == "manual"
    assert module.parse_script_arguments(["--trial-id", "manual"]) == "manual"
    assert module.parse_script_arguments([]) is None
