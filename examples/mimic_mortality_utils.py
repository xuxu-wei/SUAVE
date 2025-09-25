"""Shared utilities for the MIMIC mortality modelling examples."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
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
BENCHMARK_COLUMNS = (
    "APS_III",
    "APACHE_IV",
    "SAPS_II",
    "OASIS",
)  # do not include in training. Only use for benchamrk validation.

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
    "mutual_information_feature",
    "plot_benchmark_curves",
    "plot_distribution_shift_diagnostics",
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


def _save_figure_with_formats(
    figure: "plt.Figure",
    base_path: Path,
    *,
    formats: Sequence[str] = ("png", "svg", "pdf", "jpg"),
    dpi: int = 300,
) -> Dict[str, Path]:
    """Persist ``figure`` to ``base_path`` across multiple file formats.

    Parameters
    ----------
    figure:
        Matplotlib figure instance to serialise.
    base_path:
        Destination path without an extension. Parent directories are created
        when required.
    formats:
        Iterable of file suffixes (without leading dots) that should be
        generated. The default includes ``png``, ``svg``, ``pdf`` and ``jpg``
        to provide both raster and vector representations.
    dpi:
        Rendering DPI applied to raster exports.

    Returns
    -------
    Dict[str, Path]
        Mapping from format suffix to the saved :class:`pathlib.Path`.
    """

    base_path = base_path.with_suffix("")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, Path] = {}
    for suffix in formats:
        target = base_path.with_suffix(f".{suffix}")
        figure.savefig(target, dpi=dpi, bbox_inches="tight", format=suffix)
        saved_paths[suffix] = target
    return saved_paths


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


def make_baseline_model_factories(
    random_state: int,
) -> Dict[str, Callable[[], Pipeline]]:
    """Return model factories for the supervised transfer comparison."""

    return {
        "Logistic regression": lambda: make_logistic_pipeline(random_state),
        "Random forest": lambda: make_random_forest_pipeline(random_state),
        "XGBoost": lambda: make_xgboost_pipeline(random_state),
    }


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
    return_raw: bool = False,
) -> Union[
    Dict[str, Tuple[pd.DataFrame, pd.Series]],
    Tuple[
        Dict[str, Tuple[pd.DataFrame, pd.Series]],
        Dict[str, Tuple[pd.DataFrame, pd.Series]],
    ],
]:
    """Construct real and synthetic training sets for TSTR/TRTR evaluation.

    Parameters
    ----------
    model:
        The fitted :class:`SUAVE` model used to generate synthetic samples.
    feature_columns:
        Ordered collection of feature column names to preserve in each dataset.
    real_features:
        Feature frame from the source domain (e.g., MIMIC train split).
    real_labels:
        Corresponding label series aligned with ``real_features``.
    random_state:
        Seed controlling synthetic sampling reproducibility.
    return_raw:
        When ``True``, also return schema-aligned (non-numeric) feature frames for
        each training set. These are required when re-training SUAVE models that
        expect categorical values rather than numeric casts.

    Returns
    -------
    datasets : Dict[str, Tuple[pd.DataFrame, pd.Series]]
        Mapping from dataset name to a tuple of numeric feature frame and label
        series.
    raw_datasets : Dict[str, Tuple[pd.DataFrame, pd.Series]]
        Only returned when ``return_raw`` is ``True``. Contains the same data as
        ``datasets`` but with schema-aligned feature frames prior to
        ``to_numeric_frame`` conversion.
    """

    feature_columns = list(feature_columns)
    raw_real_features = real_features.loc[:, feature_columns].reset_index(drop=True)
    real_label_series = pd.Series(real_labels).reset_index(drop=True)
    real_label_series.name = real_labels.name

    raw_datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
        "TRTR (real)": (
            raw_real_features.copy(),
            real_label_series.copy(),
        )
    }
    datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
        name: (
            to_numeric_frame(features).reset_index(drop=True),
            labels.copy(),
        )
        for name, (features, labels) in raw_datasets.items()
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
        return frame.reset_index(drop=True)

    # Conditional sampling that mirrors the empirical class distribution.
    unconditional_labels = np.random.default_rng(random_state).choice(
        label_array,
        size=n_train,
        replace=True,
    )
    synthesis_features = sample_features(
        n_train,
        conditional=True,
        labels=np.asarray(unconditional_labels),
    )
    synthesis_labels = pd.Series(unconditional_labels, name=real_label_series.name)
    datasets["TSTR synthesis"] = (
        to_numeric_frame(synthesis_features.copy()).reset_index(drop=True),
        synthesis_labels.copy(),
    )
    raw_datasets["TSTR synthesis"] = (
        synthesis_features,
        synthesis_labels,
    )

    balanced_labels = _generate_balanced_labels(
        label_array,
        n_train,
        random_state=random_state + 10,
    )
    balance_features = sample_features(
        len(balanced_labels),
        conditional=True,
        labels=np.asarray(balanced_labels),
    )
    balance_labels = pd.Series(balanced_labels, name=real_label_series.name)
    datasets["TSTR synthesis-balance"] = (
        to_numeric_frame(balance_features.copy()).reset_index(drop=True),
        balance_labels.copy(),
    )
    raw_datasets["TSTR synthesis-balance"] = (
        balance_features,
        balance_labels,
    )

    label_counts = real_label_series.value_counts().sort_index()
    target_count = int(label_counts.max()) if not label_counts.empty else 0
    augmented_features_raw = [raw_real_features.copy()]
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
        augmented_features_raw.append(synthetic_block)
        augmented_labels.append(pd.Series(class_labels, name=real_label_series.name))

    raw_augmented = pd.concat(augmented_features_raw, ignore_index=True)
    augmented_labels_series = pd.concat(
        augmented_labels, ignore_index=True
    ).reset_index(drop=True)
    datasets["TSTR synthesis-augment"] = (
        to_numeric_frame(raw_augmented).reset_index(drop=True),
        augmented_labels_series,
    )
    raw_datasets["TSTR synthesis-augment"] = (
        raw_augmented,
        augmented_labels_series.copy(),
    )

    five_x = n_train * 5
    five_x_labels = np.random.default_rng(random_state + 20).choice(
        label_array,
        size=five_x,
        replace=True,
    )
    five_x_features = sample_features(
        five_x,
        conditional=True,
        labels=np.asarray(five_x_labels),
    )
    five_x_series = pd.Series(five_x_labels, name=real_label_series.name)
    datasets["TSTR synthesis-5x"] = (
        to_numeric_frame(five_x_features.copy()).reset_index(drop=True),
        five_x_series.copy(),
    )
    raw_datasets["TSTR synthesis-5x"] = (
        five_x_features,
        five_x_series,
    )

    five_x_balanced = _generate_balanced_labels(
        label_array,
        five_x,
        random_state=random_state + 30,
    )
    five_x_balance_features = sample_features(
        len(five_x_balanced),
        conditional=True,
        labels=np.asarray(five_x_balanced),
    )
    five_x_balance_labels = pd.Series(five_x_balanced, name=real_label_series.name)
    datasets["TSTR synthesis-5x balance"] = (
        to_numeric_frame(five_x_balance_features.copy()).reset_index(drop=True),
        five_x_balance_labels.copy(),
    )
    raw_datasets["TSTR synthesis-5x balance"] = (
        five_x_balance_features,
        five_x_balance_labels,
    )

    if return_raw:
        return datasets, raw_datasets
    return datasets


def evaluate_transfer_baselines(
    training_sets: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
    evaluation_sets: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
    *,
    model_factories: Mapping[str, Callable[[], Pipeline]],
    bootstrap_n: int,
    random_state: int,
    raw_training_sets: Optional[Mapping[str, Tuple[pd.DataFrame, pd.Series]]] = None,
    raw_evaluation_sets: Optional[Mapping[str, Tuple[pd.DataFrame, pd.Series]]] = None,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]
]:
    """Train classical models on each training set and evaluate with bootstraps.

    Parameters
    ----------
    training_sets, evaluation_sets:
        Numeric feature frames paired with label series for each dataset.
    model_factories:
        Mapping from model name to a callable producing a scikit-learn compatible
        estimator.
    bootstrap_n:
        Number of bootstrap samples for metric confidence intervals.
    random_state:
        Seed for reproducible bootstrapping.
    raw_training_sets, raw_evaluation_sets:
        Optional schema-aligned feature frames keyed identically to
        ``training_sets`` and ``evaluation_sets``. Estimators declaring the
        attribute ``requires_schema_aligned_features`` will be trained and
        evaluated using these raw frames.
    """

    summary_rows: List[Dict[str, object]] = []
    long_rows: List[Dict[str, object]] = []
    nested_results: Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]] = {}

    for training_name, (train_X_numeric, train_y_numeric) in training_sets.items():
        nested_results.setdefault(training_name, {})
        raw_training = (
            raw_training_sets.get(training_name)
            if raw_training_sets is not None
            else None
        )
        for model_name, factory in model_factories.items():
            estimator = factory()
            use_raw_features = getattr(
                estimator, "requires_schema_aligned_features", False
            )
            if use_raw_features:
                if raw_training is None:
                    raise ValueError(
                        "Raw training data missing for estimator requiring schema-aligned"
                        f" features on training set '{training_name}'."
                    )
                train_X, train_y = raw_training
            else:
                train_X, train_y = train_X_numeric, train_y_numeric

            estimator.fit(train_X, train_y)
            nested_results[training_name].setdefault(model_name, {})
            train_columns = list(train_X.columns)
            classes = getattr(estimator, "classes_", None)
            if classes is None:
                classes = np.unique(np.asarray(train_y))
            class_names = [str(value) for value in classes]
            positive_label = class_names[-1] if len(class_names) == 2 else None

            for evaluation_name, (
                eval_X_numeric,
                eval_y_numeric,
            ) in evaluation_sets.items():
                if use_raw_features:
                    if (
                        raw_evaluation_sets is None
                        or evaluation_name not in raw_evaluation_sets
                    ):
                        raise ValueError(
                            "Raw evaluation data missing for estimator requiring schema-aligned"
                            f" features on evaluation set '{evaluation_name}'."
                        )
                    eval_X, eval_y = raw_evaluation_sets[evaluation_name]
                else:
                    eval_X, eval_y = eval_X_numeric, eval_y_numeric

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
) -> Optional[Path]:
    """Generate calibration curves with Brier scores annotated in the legend."""

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color="tab:gray", label="Perfect calibration"
    )

    plotted = False
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
        plotted = True

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration: {target_name}")
    ax.legend()
    fig.tight_layout()
    if not plotted:
        plt.close(fig)
        return None

    _save_figure_with_formats(fig, output_path)
    plt.close(fig)
    return output_path.with_suffix("")


def plot_latent_space(
    model: "SUAVE",
    feature_map: Mapping[str, pd.DataFrame],
    label_map: Mapping[str, Sequence[object]],
    *,
    target_name: str,
    output_path: Path,
) -> Optional[Path]:
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
        return None

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
    _save_figure_with_formats(fig, output_path)
    plt.close(fig)
    return output_path.with_suffix("")


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
    figure_base = output_dir / f"benchmark_curves_{dataset_slug}_{target_label}"
    _save_figure_with_formats(fig, figure_base)
    plt.close(fig)
    print(
        "Saved benchmark curves for {dataset} to {path}.[png/svg/pdf/jpg]".format(
            dataset=dataset_name,
            path=figure_base,
        )
    )
    return figure_base.with_suffix("")


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
    figure_base = (
        output_dir
        / f"tstr_trtr_{dataset_slug}_{metric.lower()}_{slugify_identifier(target_label)}"
    )
    _save_figure_with_formats(fig, figure_base)
    plt.close(fig)
    print(
        "Saved {metric} bars for {dataset} to {path}.[png/svg/pdf/jpg]".format(
            metric=metric.upper(),
            dataset=evaluation_dataset,
            path=figure_base,
        )
    )
    return figure_base.with_suffix("")


def _format_metric_label(column: str) -> str:
    """Return a human-readable label for ``column`` names."""

    tokens = column.replace("_", " ").split()
    formatted: List[str] = []
    for token in tokens:
        lower = token.lower()
        if lower in {"mmd", "auc"}:
            formatted.append(lower.upper())
        elif lower == "xgboost":
            formatted.append("XGBoost")
        elif lower == "logistic":
            formatted.append("Logistic")
        elif lower == "global":
            formatted.append("Global")
        elif lower == "ci":
            formatted.append("CI")
        elif lower == "energy":
            formatted.append("Energy")
        elif lower == "distance":
            formatted.append("Distance")
        else:
            formatted.append(token.capitalize())
    return " ".join(formatted).strip()


def plot_distribution_shift_diagnostics(
    overall_df: pd.DataFrame,
    per_feature_df: pd.DataFrame,
    *,
    output_dir: Path,
    target_label: str,
    top_n: int = 10,
) -> Dict[str, Path]:
    """Create companion figures for distribution shift reporting.

    Parameters
    ----------
    overall_df:
        Dataframe containing a single row with global distribution shift
        metrics such as MMD, energy distance, permutation ``p`` values, and
        classifier two-sample test (C2ST) scores.
    per_feature_df:
        Table of per-feature diagnostics that includes the columns ``feature``,
        ``mutual_information`` and, when available, ``mmd`` and
        ``energy_distance``.
    output_dir:
        Directory used to persist the generated figures.
    target_label:
        Name of the outcome under analysis; it is incorporated into the output
        filenames.
    top_n:
        Number of features ranked by mutual information to visualise.

    Returns
    -------
    Dict[str, Path]
        Mapping from figure identifier to the saved path stem (without file
        extension). Each stem corresponds to ``.png``, ``.svg``, ``.pdf`` and
        ``.jpg`` files on disk.

    Example
    -------
    >>> overall = pd.DataFrame(
    ...     [{"global_mmd": 0.02, "global_energy_distance": 0.15, "global_mmd_p_value": 0.4}]
    ... )
    >>> per_feature = pd.DataFrame(
    ...     {
    ...         "feature": ["age", "sofa"],
    ...         "mutual_information": [0.05, 0.02],
    ...         "mmd": [0.01, 0.02],
    ...         "energy_distance": [0.10, 0.20],
    ...     }
    ... )
    >>> from pathlib import Path
    >>> outputs = plot_distribution_shift_diagnostics(
    ...     overall,
    ...     per_feature,
    ...     output_dir=Path("./out"),
    ...     target_label="demo",
    ...     top_n=2,
    ... )
    >>> sorted(outputs.keys())
    ['global_overview', 'top_features']
    """

    figures: Dict[str, Path] = {}
    if overall_df.empty:
        return figures

    output_dir.mkdir(parents=True, exist_ok=True)
    overall_row = overall_df.iloc[0]

    metric_labels: List[str] = []
    metric_values: List[float] = []
    ci_offsets: List[Tuple[float, float]] = []
    for column, value in overall_row.items():
        if (
            column.endswith("_p_value")
            or column.endswith("_ci_low")
            or column.endswith("_ci_high")
        ):
            continue
        if column in {"target", "dataset", "Target", "Dataset"}:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        metric_labels.append(_format_metric_label(column))
        metric_values.append(numeric_value)
        low = overall_row.get(f"{column}_ci_low")
        high = overall_row.get(f"{column}_ci_high")
        if low is not None and high is not None:
            ci_offsets.append((float(numeric_value - low), float(high - numeric_value)))
        else:
            ci_offsets.append((float("nan"), float("nan")))

    pvalue_labels: List[str] = []
    pvalue_values: List[float] = []
    for column, value in overall_row.items():
        if not column.endswith("_p_value"):
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        pvalue_labels.append(_format_metric_label(column.replace("_p_value", "")))
        pvalue_values.append(numeric_value)

    if metric_labels:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        metric_ax, pvalue_ax = axes
        positions = np.arange(len(metric_labels))
        metric_ax.bar(positions, metric_values, color="tab:blue", alpha=0.85)
        for idx, (offset_low, offset_high) in enumerate(ci_offsets):
            if not np.isfinite(offset_low) or not np.isfinite(offset_high):
                continue
            metric_ax.errorbar(
                positions[idx],
                metric_values[idx],
                yerr=[[offset_low], [offset_high]],
                fmt="none",
                ecolor="black",
                elinewidth=1.2,
                capsize=4,
            )
        metric_ax.set_xticks(positions)
        metric_ax.set_xticklabels(metric_labels, rotation=20, ha="right")
        metric_ax.set_ylabel("Score")
        metric_ax.set_title("Global shift metrics")

        if pvalue_labels:
            pv_positions = np.arange(len(pvalue_labels))
            pvalue_ax.bar(pv_positions, pvalue_values, color="tab:orange", alpha=0.85)
            pvalue_ax.axhline(0.05, color="tab:red", linestyle="--", linewidth=1.0)
            pvalue_ax.set_xticks(pv_positions)
            pvalue_ax.set_xticklabels(pvalue_labels, rotation=20, ha="right")
            pvalue_ax.set_yscale("log")
            pvalue_ax.set_ylabel("p-value (log scale)")
            pvalue_ax.set_title("Permutation tests")
        else:
            pvalue_ax.axis("off")

        fig.suptitle(f"Distribution shift overview – {target_label}")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        global_base = (
            output_dir / f"distribution_shift_global_{slugify_identifier(target_label)}"
        )
        _save_figure_with_formats(fig, global_base)
        plt.close(fig)
        figures["global_overview"] = global_base.with_suffix("")

    if per_feature_df.empty or "mutual_information" not in per_feature_df.columns:
        return figures

    top_features = (
        per_feature_df.sort_values("mutual_information", ascending=False)
        .head(top_n)
        .copy()
    )
    if top_features.empty:
        return figures

    features = list(top_features["feature"].astype(str))
    scores = top_features["mutual_information"].astype(float).to_numpy()
    y_positions = np.arange(len(features))

    fig, ax = plt.subplots(figsize=(12, max(4.0, 0.5 * len(features))))
    ax.barh(
        y_positions, scores, color="tab:blue", alpha=0.85, label="Mutual information"
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Mutual information")
    ax.set_title(f"Top {len(features)} shift features – {target_label}")

    has_secondary = False
    secondary_ax = ax.twiny()
    if "mmd" in top_features.columns:
        secondary_ax.plot(
            top_features["mmd"].to_numpy(),
            y_positions,
            marker="o",
            color="tab:orange",
            linestyle="",
            label="MMD",
        )
        has_secondary = True
    if "energy_distance" in top_features.columns:
        secondary_ax.plot(
            top_features["energy_distance"].to_numpy(),
            y_positions,
            marker="s",
            color="tab:green",
            linestyle="",
            label="Energy distance",
        )
        has_secondary = True

    if has_secondary:
        secondary_ax.set_xlabel("MMD / Energy distance")
        handles, labels = ax.get_legend_handles_labels()
        sec_handles, sec_labels = secondary_ax.get_legend_handles_labels()
        combined_handles = handles + sec_handles
        combined_labels = labels + sec_labels
        ax.legend(combined_handles, combined_labels, loc="lower right")
    else:
        secondary_ax.axis("off")
        ax.legend(loc="lower right")

    fig.tight_layout()
    feature_base = (
        output_dir
        / f"distribution_shift_top_features_{slugify_identifier(target_label)}"
    )
    _save_figure_with_formats(fig, feature_base)
    plt.close(fig)
    figures["top_features"] = feature_base.with_suffix("")

    return figures


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
    schema: Schema,
    *,
    random_state: int,
) -> SUAVE:
    """Instantiate :class:`SUAVE` using Optuna-style parameters."""

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
