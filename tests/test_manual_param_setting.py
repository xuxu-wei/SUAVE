"""Tests for manual parameter tuning scaffolding."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import importlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pytest
import numpy as np
import pandas as pd


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
    assert "manual_param_setting: dict = {" in content
    assert "'behaviour': 'supervised'" in content
    assert "'latent_dim': None" in content

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
def test_load_manual_tuning_overrides_missing_module(
    tmp_path: Path, module_path: str
) -> None:
    """Missing manual override modules should raise ``FileNotFoundError``."""

    module = importlib.import_module(module_path)
    manual_config = {
        "module": "nonexistent_manual_module",
        "attribute": "manual_param_setting",
    }

    with pytest.raises(FileNotFoundError):
        module.load_manual_tuning_overrides(manual_config, tmp_path)


@pytest.mark.parametrize(
    "module_path",
    [
        "examples.mimic_mortality_utils",
        "research_template.analysis_utils",
    ],
)
def test_load_manual_tuning_overrides_missing_attribute(
    tmp_path: Path, module_path: str
) -> None:
    """Missing manual override attributes should raise ``RuntimeError``."""

    module = importlib.import_module(module_path)
    module_name = "manual_override_missing_attribute"
    script_path = tmp_path / f"{module_name}.py"
    script_path.write_text("""# empty manual overrides module\n""", encoding="utf-8")

    manual_config = {"module": module_name, "attribute": "manual_param_setting"}

    try:
        with pytest.raises(RuntimeError):
            module.load_manual_tuning_overrides(manual_config, tmp_path)
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize(
    "module_path",
    [
        "examples.mimic_mortality_utils",
        "research_template.analysis_utils",
    ],
)
def test_load_manual_tuning_overrides_requires_mapping(
    tmp_path: Path, module_path: str
) -> None:
    """Non-mapping manual override attributes should raise ``RuntimeError``."""

    module = importlib.import_module(module_path)
    module_name = "manual_override_not_mapping"
    script_path = tmp_path / f"{module_name}.py"
    script_path.write_text(
        "manual_param_setting = ['not', 'a', 'mapping']\n",
        encoding="utf-8",
    )

    manual_config = {"module": module_name, "attribute": "manual_param_setting"}

    try:
        with pytest.raises(RuntimeError):
            module.load_manual_tuning_overrides(manual_config, tmp_path)
    finally:
        sys.modules.pop(module_name, None)


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
    assert manifest.get("trial_number") == "manual"
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
def test_collect_manual_and_optuna_overview(
    tmp_path: Path, module_path: str
) -> None:
    """Manual/Optuna overview should prioritise manual entries and Pareto rows."""

    module = importlib.import_module(module_path)
    model_dir = tmp_path / f"{module_path.replace('.', '_')}_model"
    optuna_dir = tmp_path / f"{module_path.replace('.', '_')}_optuna"
    model_dir.mkdir(parents=True, exist_ok=True)
    optuna_dir.mkdir(parents=True, exist_ok=True)

    manual_model = model_dir / "manual.pt"
    manual_calibrator = model_dir / "manual_calibrator.joblib"
    manual_model.write_text("model", encoding="utf-8")
    manual_calibrator.write_text("calibrator", encoding="utf-8")

    module.record_manual_model_manifest(
        model_dir,
        "mortality",
        model_path=manual_model,
        calibrator_path=manual_calibrator,
        params={"learning_rate": 0.01},
        values=(0.9, 0.02),
        validation_metrics={"ROAUC": 0.9},
        tstr_metrics={"auroc": 0.88},
        trtr_metrics={"auroc": 0.91},
    )

    best_info_path = optuna_dir / "optuna_best_info_mortality.json"
    best_info_payload = {
        "preferred_trial_number": 1,
        "pareto_front": [
            {
                "trial_number": 1,
                "values": [0.92, 0.02],
                "params": {"latent_dim": 16},
            },
            {
                "trial_number": 2,
                "values": [0.91, 0.015],
                "params": {"latent_dim": 12},
            },
        ],
    }
    best_info_path.write_text(json.dumps(best_info_payload), encoding="utf-8")

    best_params_path = optuna_dir / "optuna_best_params_mortality.json"
    best_params_payload = {"preferred_params": {"latent_dim": 16}}
    best_params_path.write_text(json.dumps(best_params_payload), encoding="utf-8")

    trials_path = optuna_dir / "optuna_trials_mortality.csv"
    trials_path.write_text(
        (
            "trial_number,validation_roauc,tstr_trtr_delta_auc,dropout\n"
            "1,0.92,0.02,0.1\n"
            "3,0.89,0.01,0.2\n"
        ),
        encoding="utf-8",
    )

    summary_df, ranked_df = module.collect_manual_and_optuna_overview(
        target_label="mortality",
        model_dir=model_dir,
        optuna_dir=optuna_dir,
        study_prefix=None,
        storage=None,
    )

    assert not summary_df.empty
    assert summary_df.iloc[0]["Source"] == "Manual override"
    assert "manual.pt" in summary_df.iloc[0]["Model path"]
    assert "Validation ROAUC" in summary_df.columns

    assert not ranked_df.empty
    assert ranked_df.iloc[0]["Source"] == "Manual override"
    assert "Optuna study" in ranked_df["Source"].tolist()
    assert {"learning_rate", "latent_dim", "dropout"}.issubset(
        set(ranked_df.columns)
    )
    assert pytest.approx(ranked_df.iloc[0]["learning_rate"]) == 0.01
    assert "Validation ROAUC" in ranked_df.columns
    assert "TSTR/TRTR ΔAUC" in ranked_df.columns
    assert pytest.approx(ranked_df.iloc[0]["Validation ROAUC"]) == 0.9


@pytest.mark.parametrize(
    "module_path",
    [
        "examples.mimic_mortality_utils",
        "research_template.analysis_utils",
    ],
)
def test_run_manual_override_training_history_flag(
    tmp_path: Path, module_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manual training should optionally start from Optuna history."""

    module = importlib.import_module(module_path)

    X_train = pd.DataFrame({"feature": [0.1, 0.2, 0.3, 0.4]})
    y_train = pd.Series([0, 1, 0, 1])
    X_validation = pd.DataFrame({"feature": [0.15, 0.25]})
    y_validation = pd.Series([0, 1])

    captured_params: List[Dict[str, Any]] = []
    manifest_params: List[Dict[str, Any]] = []

    class _DummyModel:
        def __init__(self, params: Mapping[str, Any]):
            self.params = dict(params)

        def fit(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - stub
            return None

        def save(self, path: Path) -> None:
            Path(path).write_text("model", encoding="utf-8")

    class _DummyCalibrator:
        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:  # pragma: no cover - stub
            return np.tile(np.array([[0.4, 0.6]]), (len(features), 1))

    def _build(params: Mapping[str, Any], *_args: object, **_kwargs: object) -> _DummyModel:
        captured_params.append(dict(params))
        return _DummyModel(params)

    def _record(
        model_dir: Path,
        target_label: str,
        *,
        params: Mapping[str, Any],
        **_: object,
    ) -> Path:
        manifest_params.append(dict(params))
        manifest_path = model_dir / f"manifest_{target_label}.json"
        manifest_path.write_text(json.dumps({"params": dict(params)}), encoding="utf-8")
        return manifest_path

    def _evaluate(model: _DummyModel, **_: object) -> Dict[str, Any]:
        return {
            "validation_metrics": {"ROAUC": 0.8},
            "tstr_metrics": {"auroc": 0.79},
            "trtr_metrics": {"auroc": 0.78},
            "values": (0.8, 0.01),
        }

    monkeypatch.setattr(module, "build_suave_model", _build)
    monkeypatch.setattr(module, "resolve_suave_fit_kwargs", lambda params: {})
    monkeypatch.setattr(module, "is_interactive_session", lambda: False)
    monkeypatch.setattr(module, "fit_isotonic_calibrator", lambda *args, **kwargs: _DummyCalibrator())
    monkeypatch.setattr(module.joblib, "dump", lambda obj, path: Path(path).write_text("cal", encoding="utf-8"))
    monkeypatch.setattr(module, "evaluate_candidate_model_performance", _evaluate)
    monkeypatch.setattr(module, "record_manual_model_manifest", _record)

    model_dir = tmp_path / "model_dir"
    calibration_dir = tmp_path / "calibration_dir"

    base_params = {"learning_rate": 0.01}
    manual_overrides = {"dropout": 0.2}

    history_result = module.run_manual_override_training(
        target_label="mortality",
        manual_overrides=manual_overrides,
        base_params=base_params,
        override_on_history=True,
        schema={},
        feature_columns=["feature"],
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        model_dir=model_dir,
        calibration_dir=calibration_dir,
        random_state=0,
    )

    assert captured_params[0] == {"learning_rate": 0.01, "dropout": 0.2}
    assert manifest_params[0] == captured_params[0]
    assert history_result["params"] == captured_params[0]

    fresh_result = module.run_manual_override_training(
        target_label="mortality",
        manual_overrides=manual_overrides,
        base_params=base_params,
        override_on_history=False,
        schema={},
        feature_columns=["feature"],
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        model_dir=model_dir,
        calibration_dir=calibration_dir,
        random_state=0,
    )

    assert captured_params[1] == {"dropout": 0.2}
    assert manifest_params[1] == captured_params[1]
    assert fresh_result["params"] == captured_params[1]


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


@pytest.mark.parametrize(
    "module_path",
    [
        "examples.mimic_mortality_utils",
        "research_template.analysis_utils",
    ],
)
def test_prompt_manual_override_action_accepts_manual(module_path: str) -> None:
    """Manual selection helper should accept the 'manual' keyword."""

    module = importlib.import_module(module_path)
    responses = iter(["manual"])

    action = module.prompt_manual_override_action(input_fn=lambda prompt: next(responses))
    assert action == "reuse"


@pytest.mark.parametrize(
    "module_path",
    [
        "examples.mimic_mortality_utils",
        "research_template.analysis_utils",
    ],
)
def test_prompt_manual_override_action_handles_interrupt(module_path: str) -> None:
    """Keyboard interrupts should default to continuing with Optuna once confirmed."""

    module = importlib.import_module(module_path)

    def _inputs() -> Iterable[object]:
        yield KeyboardInterrupt()
        yield "y"

    responses = _inputs()

    def _input(prompt: str) -> str:
        result = next(responses)
        if isinstance(result, BaseException):
            raise result
        return str(result)

    action = module.prompt_manual_override_action(input_fn=_input)
    assert action == "optuna"


@pytest.mark.parametrize(
    "module_path",
    [
        "examples.mimic_mortality_utils",
        "research_template.analysis_utils",
    ],
)
def test_evaluate_candidate_model_performance_shared(module_path: str) -> None:
    """Both modules should expose the shared candidate evaluation helper."""

    module = importlib.import_module(module_path)

    X_train = pd.DataFrame({"age": [30, 40, 50, 60], "score": [0.1, 0.2, 0.3, 0.4]})
    y_train = pd.Series([0, 1, 0, 1])
    X_validation = pd.DataFrame({"age": [35, 45], "score": [0.15, 0.25]})
    y_validation = pd.Series([0, 1])

    class _DummyModel:
        def __init__(self, sample_frame: pd.DataFrame) -> None:
            self._sample_frame = sample_frame.reset_index(drop=True)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover - unused
            return np.tile(np.array([0.4, 0.6]), (len(X), 1))

        def sample(
            self,
            n_samples: int,
            conditional: bool = True,
            y: Optional[pd.Series] = None,
        ) -> pd.DataFrame:
            reps = (n_samples + len(self._sample_frame) - 1) // len(self._sample_frame)
            tiled = pd.concat([self._sample_frame] * reps, ignore_index=True)
            return tiled.iloc[:n_samples].reset_index(drop=True)

    dummy_model = _DummyModel(X_train)
    validation_probs = np.column_stack(
        [1 - y_validation.to_numpy(), y_validation.to_numpy()]
    )

    results = module.evaluate_candidate_model_performance(
        dummy_model,  # type: ignore[arg-type]
        feature_columns=["age", "score"],
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        random_state=0,
        probability_fn=lambda _model, frame: validation_probs,
    )

    assert set(results.keys()) == {
        "validation_metrics",
        "tstr_metrics",
        "trtr_metrics",
        "delta_auc",
        "values",
    }
    assert isinstance(results["values"], tuple)
    assert len(results["values"]) == 2
