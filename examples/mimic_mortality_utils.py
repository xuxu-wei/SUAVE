"""Shared utilities for the MIMIC mortality modelling examples."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from tabulate import tabulate

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    roc_curve,
)
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from suave import Schema, SchemaInferencer, SUAVE  # noqa: E402
from suave.evaluate import (  # noqa: E402
    compute_auroc,
    evaluate_classification,
    kolmogorov_smirnov_statistic,
    mutual_information_feature,
    rbf_mmd,
)
from cls_eval import evaluate_predictions  # noqa: E402


RANDOM_STATE: int = 20201021
TARGET_COLUMNS: Tuple[str, str] = ("in_hospital_mortality", "28d_mortality")
BENCHMARK_COLUMNS = ('APS_III', 'APACHE_IV', 'SAPS_II', 'OASIS') # do not include in training. Only use for benchamrk validation.

CALIBRATION_SIZE: float = 0.2
VALIDATION_SIZE: float = 0.2

HIDDEN_DIMENSION_OPTIONS: Dict[str, Tuple[int, ...]] = {
    "lean": (64, 32),
    "compact": (96, 48),
    "small": (128, 64),
    "medium": (256, 128),
    "wide": (384, 192),
    "extra_wide": (512, 256),
    "ultra_wide": (640, 320),
}

HEAD_HIDDEN_DIMENSION_OPTIONS: Dict[str, Tuple[int, ...]] = {
    "minimal": (16,),
    "compact": (32,),
    "small": (48,),
    "medium": (48, 32),
    "wide": (96, 48, 16),
    "extra_wide": (64, 128, 64, 16),
    "deep": (128, 64, 32),
}

__all__ = [
    "RANDOM_STATE",
    "TARGET_COLUMNS",
    "BENCHMARK_COLUMNS",
    "CALIBRATION_SIZE",
    "VALIDATION_SIZE",
    "Schema",
    "SchemaInferencer",
    "HIDDEN_DIMENSION_OPTIONS",
    "HEAD_HIDDEN_DIMENSION_OPTIONS",
    "build_prediction_dataframe",
    "compute_auc",
    "compute_binary_metrics",
    "dataframe_to_markdown",
    "define_schema",
    "extract_positive_probabilities",
    "fit_isotonic_calibrator",
    "format_float",
    "is_interactive_session",
    "kolmogorov_smirnov_statistic",
    "load_dataset",
    "load_or_create_iteratively_imputed_features",
    "make_logistic_pipeline",
    "make_random_forest_pipeline",
    "make_xgboost_pipeline",
    "make_baseline_model_factories",
    "SuaveTransferEstimator",
    "mutual_information_feature",
    "plot_benchmark_curves",
    "plot_calibration_curves",
    "plot_latent_space",
    "plot_transfer_metric_bars",
    "prepare_features",
    "render_dataframe",
    "rbf_mmd",
    "schema_markdown_table",
    "schema_to_dataframe",
    "slugify_identifier",
    "split_train_validation_calibration",
    "to_numeric_frame",
    "build_tstr_training_sets",
    "evaluate_transfer_baselines",
    "build_suave_model",
    "resolve_classification_loss_weight",
]


def is_interactive_session() -> bool:
    """Return ``True`` when executed inside an interactive IPython session."""

    try:
        from IPython import get_ipython
    except ImportError:  # pragma: no cover - optional dependency
        return False
    return get_ipython() is not None


def render_dataframe(
    df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    floatfmt: Optional[str] = ".3f",
) -> None:
    """Render a dataframe either via ``display`` or ``tabulate``."""

    if title:
        print(title)
    if df.empty:
        print("(empty table)")
        return
    if is_interactive_session():
        display(df)
        return
    tabulate_kwargs = {"headers": "keys", "tablefmt": "github", "showindex": False}
    if floatfmt is not None:
        tabulate_kwargs["floatfmt"] = floatfmt
    print(tabulate(df, **tabulate_kwargs))


def dataframe_to_markdown(df: pd.DataFrame, *, floatfmt: Optional[str] = ".3f") -> str:
    """Return a GitHub-flavoured Markdown representation of ``df``."""

    if df.empty:
        return "_No data available._"
    tabulate_kwargs = {"headers": "keys", "tablefmt": "github", "showindex": False}
    if floatfmt is not None:
        tabulate_kwargs["floatfmt"] = floatfmt
    return tabulate(df, **tabulate_kwargs)


def schema_to_dataframe(schema: Schema) -> pd.DataFrame:
    """Convert a :class:`Schema` into a tidy dataframe."""

    records: List[Dict[str, object]] = []
    for column, spec in schema.to_dict().items():
        records.append(
            {
                "Column": column,
                "Type": spec.get("type", ""),
                "n_classes": spec.get("n_classes", ""),
                "y_dim": spec.get("y_dim", ""),
            }
        )
    return pd.DataFrame(records)


def slugify_identifier(value: str) -> str:
    """Return a filesystem-friendly identifier derived from ``value``."""

    cleaned = [char.lower() if char.isalnum() else "_" for char in value.strip()]
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def load_or_create_iteratively_imputed_features(
    feature_sets: Mapping[str, pd.DataFrame],
    *,
    output_dir: Path,
    target_label: str,
    reference_key: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Path], bool]:
    """Load cached iterative imputations or fit a new imputer."""

    if reference_key not in feature_sets:
        raise KeyError(
            f"Reference key '{reference_key}' missing from feature sets: {list(feature_sets)}"
        )

    dataset_paths: Dict[str, Path] = {
        name: output_dir
        / f"iterative_imputed_{slugify_identifier(name)}_{slugify_identifier(target_label)}.csv"
        for name in feature_sets
    }

    loaded_features: Dict[str, pd.DataFrame] = {}
    load_successful = True
    for name, path in dataset_paths.items():
        features = feature_sets[name]
        if not path.exists():
            load_successful = False
            break
        cached = pd.read_csv(path, index_col=0)
        column_match = list(cached.columns) == list(features.columns)
        length_match = len(cached) == len(features)
        if not (column_match and length_match):
            load_successful = False
            break
        try:
            cached = cached.loc[features.index]
        except KeyError:
            load_successful = False
            break
        loaded_features[name] = cached

    if load_successful:
        return loaded_features, dataset_paths, True

    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    imputer = IterativeImputer()
    imputer.fit(feature_sets[reference_key])

    imputed_features: Dict[str, pd.DataFrame] = {}
    for name, features in feature_sets.items():
        transformed = imputer.transform(features)
        imputed_df = pd.DataFrame(
            transformed,
            columns=features.columns,
            index=features.index,
        )
        path = dataset_paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        imputed_df.to_csv(path)
        imputed_features[name] = imputed_df

    return imputed_features, dataset_paths, False


def make_logistic_pipeline(random_state: Optional[int] = None) -> Pipeline:
    """Factory for the baseline classifier used in TSTR/TRTR evaluations."""

    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("imputer", IterativeImputer()),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=200, random_state=random_state),
            ),
        ]
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a TSV file into a :class:`pandas.DataFrame`."""

    return pd.read_csv(path, sep="\t")


def make_random_forest_pipeline(random_state: Optional[int] = None) -> Pipeline:
    """Return a random forest pipeline with iterative imputation."""

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    return Pipeline(
        [
            ("imputer", IterativeImputer()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_xgboost_pipeline(random_state: Optional[int] = None) -> Pipeline:
    """Return an XGBoost pipeline with iterative imputation."""

    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    try:
        from xgboost import XGBClassifier
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "xgboost is required for the mortality TSTR evaluation."
        ) from error

    return Pipeline(
        [
            ("imputer", IterativeImputer()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="auc",
                    reg_lambda=1.0,
                    random_state=random_state,
                    tree_method="hist",
                    n_jobs=-1,
                ),
            ),
        ]
    )


class SuaveTransferEstimator:
    """Lightweight wrapper that trains SUAVE for TSTR/TRTR evaluation."""

    def __init__(
        self,
        params: Mapping[str, Any],
        schema: Optional[Schema],
        *,
        random_state: int,
    ) -> None:
        self._params = dict(params)
        self._schema = schema
        self._random_state = random_state
        self.model: Optional[SUAVE] = None
        self.classes_: Optional[np.ndarray] = None

    def _fit_kwargs(self) -> Dict[str, Any]:
        """Return training keyword arguments derived from Optuna params."""

        return {
            "warmup_epochs": int(self._params.get("warmup_epochs", 3)),
            "kl_warmup_epochs": int(self._params.get("kl_warmup_epochs", 0)),
            "head_epochs": int(self._params.get("head_epochs", 2)),
            "finetune_epochs": int(self._params.get("finetune_epochs", 2)),
            "joint_decoder_lr_scale": float(
                self._params.get("joint_decoder_lr_scale", 0.1)
            ),
            "early_stop_patience": int(
                self._params.get("early_stop_patience", 10)
            ),
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SuaveTransferEstimator":
        """Instantiate and train SUAVE with cached Optuna parameters."""

        self.model = build_suave_model(
            self._params,
            self._schema,
            random_state=self._random_state,
        )
        self.model.fit(X, y, **self._fit_kwargs())
        self.classes_ = getattr(self.model, "classes_", None)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Delegate probability prediction to the trained SUAVE model."""

        if self.model is None:
            raise RuntimeError("SUAVE model must be fitted before prediction")
        probabilities = self.model.predict_proba(X)
        return np.asarray(probabilities)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return deterministic class predictions from the SUAVE model."""

        if self.model is None:
            raise RuntimeError("SUAVE model must be fitted before prediction")
        return np.asarray(self.model.predict(X))


def make_baseline_model_factories(
    random_state: int,
    *,
    suave_params: Optional[Mapping[str, Any]] = None,
    schema: Optional[Schema] = None,
) -> Dict[str, Callable[[], object]]:
    """Return model factories for the supervised transfer comparison."""

    factories: Dict[str, Callable[[], object]] = {
        "Logistic regression": lambda: make_logistic_pipeline(random_state),
        "Random forest": lambda: make_random_forest_pipeline(random_state),
        "XGBoost": lambda: make_xgboost_pipeline(random_state),
    }
    if suave_params is not None:
        factories["SUAVE"] = lambda: SuaveTransferEstimator(
            suave_params,
            schema,
            random_state=random_state,
        )
    return factories


def define_schema(
    df: pd.DataFrame, feature_columns: Iterable[str], mode: str = "info"
) -> Schema:
    """Create a :class:`Schema` describing ``df``'s feature columns."""

    inferencer = SchemaInferencer()
    result = inferencer.infer(
        df,
        feature_columns,
        mode=mode,
    )
    for message in result.messages:
        print(f"[schema] {message}")
    return result.schema


def prepare_features(df: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    """Return features aligned to ``feature_columns``."""

    return df.loc[:, list(feature_columns)].copy()


def split_train_validation_calibration(
    features: pd.DataFrame,
    targets: pd.Series,
    *,
    calibration_size: float,
    validation_size: float,
    random_state: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Split data into train, validation, and calibration subsets."""

    from sklearn.model_selection import train_test_split

    X_model, X_calibration, y_model, y_calibration = train_test_split(
        features,
        targets,
        test_size=calibration_size,
        stratify=targets,
        random_state=random_state,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_model,
        y_model,
        test_size=validation_size,
        stratify=y_model,
        random_state=random_state,
    )
    return (
        X_train.reset_index(drop=True),
        X_validation.reset_index(drop=True),
        X_calibration.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_validation.reset_index(drop=True),
        y_calibration.reset_index(drop=True),
    )


def schema_markdown_table(schema: Schema) -> str:
    """Return a Markdown table summarising ``schema``."""

    header = "| Column | Type | n_classes | y_dim |\n| --- | --- | --- | --- |"
    rows = [header]
    schema_dict: MutableMapping[str, MutableMapping[str, object]] = schema.to_dict()
    for name, spec in schema_dict.items():
        n_classes = spec.get("n_classes", "")
        y_dim = spec.get("y_dim", "")
        rows.append(f"| {name} | {spec['type']} | {n_classes} | {y_dim} |")
    return "\n".join(rows)


def format_float(value: Optional[float]) -> str:
    """Format floating point numbers for Markdown tables."""

    if value is None:
        return "nan"
    if isinstance(value, float) and not np.isfinite(value):
        return "nan"
    return f"{float(value):.3f}"


def _normalize_zero_indexed_labels(targets: pd.Series | np.ndarray) -> np.ndarray:
    """Return ``targets`` as an integer array with labels mapped to start at zero."""

    labels = np.asarray(targets)
    if labels.size == 0:
        if labels.dtype != int and not np.issubdtype(labels.dtype, np.integer):
            labels = labels.astype(int, copy=False)
        return labels

    if labels.dtype != int and not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(int, copy=False)

    unique = np.unique(labels)
    if np.array_equal(unique, np.arange(unique.size)):
        return labels

    return np.searchsorted(unique, labels)


def compute_auc(probabilities: np.ndarray, targets: pd.Series | np.ndarray) -> float:
    """Return the ROC AUC given predicted probabilities and targets."""

    labels = _normalize_zero_indexed_labels(targets)

    try:
        return float(compute_auroc(probabilities, labels))
    except ValueError:
        return float("nan")


def to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns in ``df`` to numeric values."""

    numeric = df.copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric


def _ensure_feature_frame(
    samples: pd.DataFrame | np.ndarray,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Return ``samples`` restricted to ``feature_columns`` as a dataframe."""

    if isinstance(samples, pd.DataFrame):
        missing = [
            column for column in feature_columns if column not in samples.columns
        ]
        if missing:
            raise KeyError(
                "Sampled dataframe is missing expected feature columns: " f"{missing}"
            )
        frame = samples.loc[:, list(feature_columns)].copy()
    else:
        frame = pd.DataFrame(samples, columns=list(feature_columns))
    return frame


def _generate_balanced_labels(
    labels: Sequence[object],
    total_samples: int,
    *,
    random_state: int,
) -> np.ndarray:
    """Generate a balanced label vector using ``labels`` as candidates."""

    unique = np.unique(np.asarray(labels))
    if unique.size == 0:
        raise ValueError("Cannot balance labels when no classes are present")

    base = total_samples // unique.size
    remainder = total_samples % unique.size
    counts = {value: base for value in unique}

    rng = np.random.default_rng(random_state)
    if remainder > 0:
        extras = rng.choice(unique, size=remainder, replace=False)
        for value in extras:
            counts[value] += 1

    balanced = np.concatenate([np.full(counts[value], value) for value in unique])
    shuffle_rng = np.random.default_rng(random_state + 1)
    shuffle_rng.shuffle(balanced)
    return balanced


def build_tstr_training_sets(
    model: SUAVE,
    feature_columns: Sequence[str],
    real_features: pd.DataFrame,
    real_labels: pd.Series,
    *,
    random_state: int,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Construct real and synthetic training sets for TSTR/TRTR evaluation."""

    feature_columns = list(feature_columns)
    real_feature_frame = to_numeric_frame(real_features.loc[:, feature_columns])
    real_label_series = pd.Series(real_labels).reset_index(drop=True)
    real_label_series.name = real_labels.name

    datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
        "TRTR (real)": (
            real_feature_frame.reset_index(drop=True),
            real_label_series.copy(),
        )
    }

    n_train = len(real_label_series)
    label_array = real_label_series.to_numpy()

    def sample_features(
        n_samples: int,
        *,
        conditional: bool,
        labels: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        sampled = model.sample(
            n_samples,
            conditional=conditional,
            y=labels if conditional else None,
        )
        frame = _ensure_feature_frame(sampled, feature_columns)
        return to_numeric_frame(frame).reset_index(drop=True)

    # Conditional sampling that mirrors the empirical class distribution.
    unconditional_labels = np.random.default_rng(random_state).choice(
        label_array,
        size=n_train,
        replace=True,
    )
    datasets["TSTR synthesis"] = (
        sample_features(
            n_train,
            conditional=True,
            labels=np.asarray(unconditional_labels),
        ),
        pd.Series(unconditional_labels, name=real_label_series.name),
    )

    balanced_labels = _generate_balanced_labels(
        label_array,
        n_train,
        random_state=random_state + 10,
    )
    datasets["TSTR synthesis-balance"] = (
        sample_features(
            len(balanced_labels),
            conditional=True,
            labels=np.asarray(balanced_labels),
        ),
        pd.Series(balanced_labels, name=real_label_series.name),
    )

    label_counts = real_label_series.value_counts().sort_index()
    target_count = int(label_counts.max()) if not label_counts.empty else 0
    augmented_features = [real_feature_frame.reset_index(drop=True)]
    augmented_labels = [real_label_series.copy()]
    for value, count in label_counts.items():
        deficit = target_count - int(count)
        if deficit <= 0:
            continue
        class_labels = np.full(deficit, value)
        synthetic_block = sample_features(
            deficit,
            conditional=True,
            labels=class_labels,
        )
        augmented_features.append(synthetic_block)
        augmented_labels.append(pd.Series(class_labels, name=real_label_series.name))

    datasets["TSTR synthesis-augment"] = (
        to_numeric_frame(pd.concat(augmented_features, ignore_index=True)),
        pd.concat(augmented_labels, ignore_index=True).reset_index(drop=True),
    )

    five_x = n_train * 5
    five_x_labels = np.random.default_rng(random_state + 20).choice(
        label_array,
        size=five_x,
        replace=True,
    )
    datasets["TSTR synthesis-5x"] = (
        sample_features(
            five_x,
            conditional=True,
            labels=np.asarray(five_x_labels),
        ),
        pd.Series(five_x_labels, name=real_label_series.name),
    )

    five_x_balanced = _generate_balanced_labels(
        label_array,
        five_x,
        random_state=random_state + 30,
    )
    datasets["TSTR synthesis-5x balance"] = (
        sample_features(
            len(five_x_balanced),
            conditional=True,
            labels=np.asarray(five_x_balanced),
        ),
        pd.Series(five_x_balanced, name=real_label_series.name),
    )

    return datasets


def evaluate_transfer_baselines(
    training_sets: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
    evaluation_sets: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
    *,
    model_factories: Mapping[str, Callable[[], object]],
    bootstrap_n: int,
    random_state: int,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]
]:
    """Train classical models on each training set and evaluate with bootstraps."""

    summary_rows: List[Dict[str, object]] = []
    long_rows: List[Dict[str, object]] = []
    nested_results: Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]] = {}

    for training_name, (train_X, train_y) in training_sets.items():
        nested_results.setdefault(training_name, {})
        train_columns = list(train_X.columns)
        for model_name, factory in model_factories.items():
            estimator = factory()
            estimator.fit(train_X, train_y)
            nested_results[training_name].setdefault(model_name, {})
            classes = getattr(estimator, "classes_", None)
            if classes is None:
                classes = np.unique(np.asarray(train_y))
            class_names = [str(value) for value in classes]
            positive_label = class_names[-1] if len(class_names) == 2 else None

            for evaluation_name, (eval_X, eval_y) in evaluation_sets.items():
                if eval_X.empty or len(eval_y) == 0:
                    continue
                aligned_eval = eval_X.loc[:, train_columns]
                probabilities = estimator.predict_proba(aligned_eval)
                predictions = estimator.predict(aligned_eval)
                prediction_df = build_prediction_dataframe(
                    probabilities,
                    eval_y,
                    predictions,
                    class_names,
                )

                results = evaluate_predictions(
                    prediction_df,
                    label_col="label",
                    pred_col="y_pred",
                    positive_label=positive_label,
                    bootstrap_n=bootstrap_n,
                    random_state=random_state,
                )
                nested_results[training_name][model_name][evaluation_name] = results

                overall_df = results.get("overall", pd.DataFrame())
                if overall_df.empty:
                    continue
                row: Dict[str, object] = {
                    "training_dataset": training_name,
                    "evaluation_dataset": evaluation_name,
                    "model": model_name,
                }
                for metric in ("accuracy", "roc_auc"):
                    if metric in overall_df.columns:
                        value = overall_df.at[0, metric]
                        row[metric] = float(value)
                        low_col = f"{metric}_ci_low"
                        high_col = f"{metric}_ci_high"
                        row[low_col] = (
                            float(overall_df.at[0, low_col])
                            if low_col in overall_df.columns
                            else float("nan")
                        )
                        row[high_col] = (
                            float(overall_df.at[0, high_col])
                            if high_col in overall_df.columns
                            else float("nan")
                        )
                        long_rows.append(
                            {
                                "training_dataset": training_name,
                                "evaluation_dataset": evaluation_name,
                                "model": model_name,
                                "metric": metric,
                                "estimate": float(value),
                                "ci_low": (
                                    float(overall_df.at[0, low_col])
                                    if low_col in overall_df.columns
                                    else float("nan")
                                ),
                                "ci_high": (
                                    float(overall_df.at[0, high_col])
                                    if high_col in overall_df.columns
                                    else float("nan")
                                ),
                            }
                        )
                    else:
                        row[metric] = float("nan")
                        row[f"{metric}_ci_low"] = float("nan")
                        row[f"{metric}_ci_high"] = float("nan")
                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    long_df = pd.DataFrame(long_rows)
    return summary_df, long_df, nested_results


def _generate_balanced_labels(
    labels: Sequence[object],
    total_samples: int,
    *,
    random_state: int,
) -> np.ndarray:
    """Generate a balanced label vector using ``labels`` as candidates."""

    unique = np.unique(np.asarray(labels))
    if unique.size == 0:
        raise ValueError("Cannot balance labels when no classes are present")

    base = total_samples // unique.size
    remainder = total_samples % unique.size
    counts = {value: base for value in unique}

    rng = np.random.default_rng(random_state)
    if remainder > 0:
        extras = rng.choice(unique, size=remainder, replace=False)
        for value in extras:
            counts[value] += 1

    balanced = np.concatenate([np.full(counts[value], value) for value in unique])
    shuffle_rng = np.random.default_rng(random_state + 1)
    shuffle_rng.shuffle(balanced)
    return balanced


def extract_positive_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Return the positive-class probabilities as a 1-D array."""

    prob_matrix = np.asarray(probabilities)
    if prob_matrix.ndim == 1:
        return prob_matrix
    return prob_matrix[:, -1]


def compute_binary_metrics(
    probabilities: np.ndarray, targets: pd.Series | np.ndarray
) -> Dict[str, float]:
    """Compute AUROC, accuracy, specificity, sensitivity, and Brier score."""

    labels = _normalize_zero_indexed_labels(targets)

    try:
        classification = evaluate_classification(probabilities, labels)
    except ValueError:
        classification = {
            "accuracy": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        }

    positive_probs = extract_positive_probabilities(probabilities)
    predictions = (positive_probs >= 0.5).astype(int, copy=False)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")

    return {
        "ROAUC": classification.get("auroc", float("nan")),
        "AUC": classification.get("auroc", float("nan")),
        "ACC": classification.get("accuracy", float("nan")),
        "SPE": specificity,
        "SEN": sensitivity,
        "Brier": classification.get("brier", float("nan")),
    }


def fit_isotonic_calibrator(
    model: SUAVE,
    features: pd.DataFrame,
    targets: pd.Series | np.ndarray,
) -> CalibratedClassifierCV:
    """Wrap ``model`` with an isotonic :class:`CalibratedClassifierCV`."""

    calibrator = CalibratedClassifierCV(
        base_estimator=model,
        method="isotonic",
        cv="prefit",
    )
    calibrator.fit(features, np.asarray(targets))
    return calibrator


def plot_calibration_curves(
    probability_map: Mapping[str, np.ndarray],
    label_map: Mapping[str, np.ndarray],
    *,
    target_name: str,
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """Generate calibration curves with Brier scores annotated in the legend."""

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color="tab:gray", label="Perfect calibration"
    )

    for dataset_name, probs in probability_map.items():
        labels = label_map[dataset_name]
        pos_probs = extract_positive_probabilities(probs)
        try:
            frac_pos, mean_pred = calibration_curve(labels, pos_probs, n_bins=n_bins)
        except ValueError:
            continue
        brier = brier_score_loss(labels, pos_probs)
        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            label=f"{dataset_name} (Brier={brier:.3f})",
        )

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration: {target_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_latent_space(
    model: "SUAVE",
    feature_map: Mapping[str, pd.DataFrame],
    label_map: Mapping[str, Sequence[object]],
    *,
    target_name: str,
    output_path: Path,
) -> None:
    """Project latent representations with PCA and create scatter plots."""

    latent_blocks: List[np.ndarray] = []
    dataset_keys: List[str] = []
    for name, features in feature_map.items():
        if features.empty:
            continue
        latents = model.encode(features)
        if latents.size == 0:
            continue
        latent_blocks.append(latents)
        dataset_keys.append(name)

    if not latent_blocks:
        return

    concatenated = np.vstack(latent_blocks)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(concatenated)

    offsets = np.cumsum([0] + [block.shape[0] for block in latent_blocks])
    fig, axes = plt.subplots(
        1,
        len(latent_blocks),
        figsize=(6 * len(latent_blocks), 5),
        sharex=True,
        sharey=True,
    )

    if len(latent_blocks) == 1:
        axes = [axes]

    for idx, (ax, name) in enumerate(zip(axes, dataset_keys)):
        start, end = offsets[idx], offsets[idx + 1]
        subset = projected[start:end]
        labels = np.asarray(label_map[name])
        scatter = ax.scatter(
            subset[:, 0],
            subset[:, 1],
            c=labels,
            cmap="coolwarm",
            alpha=0.7,
            edgecolor="none",
        )
        ax.set_title(f"{name}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        legend = ax.legend(*scatter.legend_elements(), title="Label")
        ax.add_artist(legend)

    fig.suptitle(f"Latent space projection: {target_name}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_benchmark_curves(
    dataset_name: str,
    y_true: np.ndarray,
    model_probability_lookup: Mapping[str, np.ndarray],
    *,
    output_dir: Path,
    target_label: str,
    abbreviation_lookup: Optional[Mapping[str, str]] = None,
    n_bins: int = 10,
) -> Optional[Path]:
    """Plot ROC and calibration curves for the supplied dataset."""

    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        print(f"Skipping {dataset_name} curves because only one class is present.")
        return None

    fig, (roc_ax, cal_ax) = plt.subplots(1, 2, figsize=(12, 5))

    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    roc_ax.set_title(f"ROC – {dataset_name}")
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")

    cal_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    cal_ax.set_title(f"Calibration – {dataset_name}")
    cal_ax.set_xlabel("Mean predicted probability")
    cal_ax.set_ylabel("Fraction of positives")

    for model_name, probs in model_probability_lookup.items():
        abbrev = (
            model_name
            if abbreviation_lookup is None
            else abbreviation_lookup.get(model_name, model_name)
        )
        positive_probs = extract_positive_probabilities(probs)
        fpr, tpr, _ = roc_curve(y_true, positive_probs)
        roc_ax.plot(fpr, tpr, label=abbrev)

        try:
            frac_pos, mean_pred = calibration_curve(
                y_true, positive_probs, n_bins=n_bins, strategy="quantile"
            )
        except ValueError:
            print(
                f"Calibration curve for {model_name} on {dataset_name} skipped due to insufficient variation."
            )
        else:
            cal_ax.plot(mean_pred, frac_pos, marker="o", label=abbrev)

    roc_ax.legend(loc="lower right")
    cal_ax.legend(loc="upper left")
    fig.suptitle(f"Benchmark ROC & calibration – {dataset_name}")
    fig.tight_layout()

    dataset_slug = dataset_name.lower().replace(" ", "_")
    figure_path = output_dir / f"benchmark_curves_{dataset_slug}_{target_label}.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved benchmark curves for {dataset_name} to {figure_path}")
    return figure_path


def plot_transfer_metric_bars(
    metric_df: pd.DataFrame,
    *,
    metric: str,
    evaluation_dataset: str,
    training_order: Sequence[str],
    model_order: Sequence[str],
    output_dir: Path,
    target_label: str,
) -> Optional[Path]:
    """Plot grouped bar charts with error bars for TSTR/TRTR comparisons."""

    subset = metric_df[
        (metric_df["metric"] == metric)
        & (metric_df["evaluation_dataset"] == evaluation_dataset)
    ]
    if subset.empty:
        print(
            f"Skipping {metric} bars for {evaluation_dataset} because no data was provided."
        )
        return None

    training_order = list(training_order)
    model_order = list(model_order)
    x_positions = np.arange(len(training_order), dtype=float)
    width = 0.8 / max(len(model_order), 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, model_name in enumerate(model_order):
        model_subset = (
            subset[subset["model"] == model_name]
            .set_index("training_dataset")
            .reindex(training_order)
        )
        estimates = model_subset["estimate"].to_numpy()
        lower = estimates - model_subset["ci_low"].to_numpy()
        upper = model_subset["ci_high"].to_numpy() - estimates
        lower = np.nan_to_num(lower, nan=0.0, posinf=0.0, neginf=0.0)
        upper = np.nan_to_num(upper, nan=0.0, posinf=0.0, neginf=0.0)
        offsets = (idx - (len(model_order) - 1) / 2) * width
        ax.bar(
            x_positions + offsets,
            estimates,
            width=width,
            label=model_name,
            yerr=np.vstack([lower, upper]),
            capsize=4,
            alpha=0.9,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(training_order, rotation=20, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{metric.upper()} – {evaluation_dataset}")
    ax.legend()
    fig.tight_layout()

    dataset_slug = slugify_identifier(evaluation_dataset)
    figure_path = (
        output_dir
        / f"tstr_trtr_{dataset_slug}_{metric.lower()}_{slugify_identifier(target_label)}.png"
    )
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {metric.upper()} bars for {evaluation_dataset} to {figure_path}")
    return figure_path


def build_prediction_dataframe(
    probabilities: np.ndarray,
    labels: Sequence[object],
    predictions: Sequence[object],
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Assemble a dataframe compatible with :func:`evaluate_predictions`."""

    prob_matrix = np.asarray(probabilities)
    class_names = list(class_names)
    if prob_matrix.ndim == 1:
        if len(class_names) == 2:
            negative_name, positive_name = class_names[0], class_names[-1]
            proba_dict = {
                f"pred_proba_{negative_name}": 1.0 - prob_matrix,
                f"pred_proba_{positive_name}": prob_matrix,
            }
        else:
            proba_dict = {"pred_proba_0": prob_matrix}
    else:
        if prob_matrix.shape[1] == len(class_names) and len(class_names) > 0:
            proba_dict = {
                f"pred_proba_{class_names[idx]}": prob_matrix[:, idx]
                for idx in range(prob_matrix.shape[1])
            }
        else:
            proba_dict = {
                f"pred_proba_{idx}": prob_matrix[:, idx]
                for idx in range(prob_matrix.shape[1])
            }

    base_df = pd.DataFrame(
        {
            "label": np.asarray(labels),
            "y_pred": np.asarray(predictions),
        }
    )
    if proba_dict:
        proba_df = pd.DataFrame(proba_dict)
        base_df = pd.concat([base_df.reset_index(drop=True), proba_df], axis=1)
    else:
        base_df = base_df.reset_index(drop=True)
    return base_df


def resolve_classification_loss_weight(params: Mapping[str, object]) -> Optional[float]:
    """Normalise ``classification_loss_weight`` from Optuna parameters."""

    use_weight = params.get("use_classification_loss_weight")
    if isinstance(use_weight, str):
        use_weight = use_weight.lower() in {"1", "true", "yes"}
    elif isinstance(use_weight, (np.bool_,)):
        use_weight = bool(use_weight)
    if not use_weight:
        return None
    weight = params.get("classification_loss_weight")
    if weight is None:
        return 1.0
    if isinstance(weight, (np.floating, np.integer)):
        return float(weight)
    return float(weight)


def build_suave_model(
    params: Mapping[str, object],
    schema: Optional[Schema],
    *,
    random_state: int,
) -> SUAVE:
    """Instantiate :class:`SUAVE` using Optuna-style parameters."""

    if schema is None:
        warnings.warn(
            "未提供 schema，正在使用 SUAVE 默认的自动推断模式。",
            RuntimeWarning,
            stacklevel=2,
        )

    hidden_key = str(params.get("hidden_dims", "medium"))
    head_hidden_key = str(params.get("head_hidden_dims", "medium"))
    hidden_dims = HIDDEN_DIMENSION_OPTIONS.get(
        hidden_key, HIDDEN_DIMENSION_OPTIONS["medium"]
    )
    head_hidden_dims = HEAD_HIDDEN_DIMENSION_OPTIONS.get(
        head_hidden_key, HEAD_HIDDEN_DIMENSION_OPTIONS["medium"]
    )
    classification_loss_weight = resolve_classification_loss_weight(params)
    return SUAVE(
        schema=schema,
        latent_dim=int(params.get("latent_dim", 16)),
        n_components=int(params.get("n_components", 1)),
        hidden_dims=hidden_dims,
        head_hidden_dims=head_hidden_dims,
        dropout=float(params.get("dropout", 0.1)),
        learning_rate=float(params.get("learning_rate", 1e-3)),
        batch_size=int(params.get("batch_size", 256)),
        beta=float(params.get("beta", 1.5)),
        classification_loss_weight=classification_loss_weight,
        random_state=random_state,
        behaviour="supervised",
    )

