"""Schema inference utilities with multiple review modes."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

try:  # pragma: no cover - optional interactive dependency
    import matplotlib
    from matplotlib import pyplot as plt
except Exception:  # pragma: no cover - fallback when matplotlib is unavailable
    matplotlib = None
    plt = None

from .types import Schema


NUMERIC_CATEGORICAL_THRESHOLD = 0.05
MAX_CATEGORICAL_UNIQUE = 20
STD_REAL_FALLBACK = 0.1
RANGE_REAL_FALLBACK = 1.0
SMALL_STD_THRESHOLD = 1e-6
SMALL_RANGE_THRESHOLD = 1e-6
BINARY_VALUES = {0, 1}
ORDINAL_RANGE_THRESHOLD = 10
POSITIVE_SKEW_THRESHOLD = 1.0
POSITIVE_MAX_MEAN_RATIO = 5.0
REVIEW_RATIO_MARGIN = 0.02
REVIEW_SKEW_MARGIN = 0.25
REVIEW_UNIQUE_MARGIN = 2
MIN_NUMERIC_COVERAGE = 0.95


class InferenceConfidence(str, Enum):
    """Simple tri-state confidence used to annotate schema suggestions."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


_CONFIDENCE_PRIORITY = {
    InferenceConfidence.LOW: 0,
    InferenceConfidence.MEDIUM: 1,
    InferenceConfidence.HIGH: 2,
}


def _worst_confidence(*levels: InferenceConfidence) -> InferenceConfidence:
    """Return the lowest-confidence entry from ``levels``."""

    if not levels:
        return InferenceConfidence.HIGH
    return min(levels, key=_CONFIDENCE_PRIORITY.__getitem__)


SCHEMA_INFERENCE_MODES: Tuple[str, ...] = ("silent", "info", "interactive")
"""Valid options understood by :meth:`SchemaInferencer.infer`."""


@dataclass
class SchemaInferenceResult:
    """
    Container for the output of :meth:`SchemaInferencer.infer`.

    Parameters
    ----------
    schema : Schema
        The inferred :class:`Schema` object describing feature types and
        auxiliary attributes (e.g., ``n_classes`` for categorical/ordinal).
    mode : {"silent", "info", "interactive"}
        The mode in which inference was executed.
    review_columns : list of str
        Column names that were flagged for review based on proximity to
        heuristic decision boundaries or other diagnostics.
    column_notes : Mapping[str, str]
        Per-column notes describing why a feature was flagged or how it
        was treated during inference.
    messages : list of str
        Human-readable diagnostics and status messages suitable for logging
        or console display.
    column_confidence : Mapping[str, InferenceConfidence]
        Per-column confidence assessments produced during inference. These
        values are useful for colour-coding interactive review tools.

    See Also
    --------
    SchemaInferencer.infer : Runs schema inference and returns this object.
    """

    schema: Schema
    mode: str
    review_columns: List[str]
    column_notes: Mapping[str, str]
    messages: List[str]
    column_confidence: Mapping[str, InferenceConfidence]


class SchemaInferencer:
    """
    Infer a :class:`Schema` from a pandas DataFrame using simple descriptive
    statistics and heuristics, with optional human-in-the-loop review.

    Parameters
    ----------
    categorical_overrides : iterable of str, optional
        Columns that must always be treated as categorical regardless of the
        observed statistics. Useful for ID-like integers, booleans encoded
        as 0/1, or known enumerations.

    Notes
    -----
    The inferencer first attempts to recognize numeric features (including
    integer-like floats and strings convertible to numerics) and then decides
    among ``real``, ``pos``, ``count``, ``cat``, or ``ordinal`` based on
    distributional characteristics (cardinality, dispersion, skewness, range,
    and support). Columns close to decision thresholds are flagged for review.
    """

    def __init__(self, *, categorical_overrides: Optional[Iterable[str]] = None):
        """
        Construct a new :class:`SchemaInferencer`.

        Parameters
        ----------
        categorical_overrides : iterable of str, optional
            Column names to force as categorical irrespective of measured
            statistics.

        Examples
        --------
        >>> inferencer = SchemaInferencer(categorical_overrides={"sex", "CRRT"})
        """
        self._categorical_overrides = set(categorical_overrides or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def infer(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[Iterable[str]] = None,
        *,
        mode: str = "silent",
    ) -> SchemaInferenceResult:
        """
        Infer a :class:`Schema` from ``df`` under the requested ``mode``.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data frame containing candidate feature columns.
        feature_columns : iterable of str, optional
            Subset of columns to consider. If ``None``, all columns in ``df``
            are processed.
        mode : {"silent", "info", "interactive"}, default ``"silent"``
            Controls verbosity and whether an interactive review UI is
            attempted.

        Returns
        -------
        SchemaInferenceResult
            Object containing the inferred schema and review diagnostics.

        See Also
        --------
        SchemaInferencer._infer_column_schema : Column-level inference routine.

        Notes
        -----
        - In ``info`` and ``interactive`` modes, columns near decision
          thresholds are listed in ``result.review_columns`` and explanatory
          notes are attached in ``result.column_notes``.
        - In ``interactive`` mode, a GUI may be launched (environment
          permitting) to let users confirm/override inferred types.
        """

        columns = list(feature_columns) if feature_columns is not None else list(df.columns)

        schema_dict: Dict[str, MutableMapping[str, object]] = {}
        review_notes: Dict[str, str] = {}
        column_confidence: Dict[str, InferenceConfidence] = {}

        for column in columns:
            spec, notes, confidence = self._infer_column_schema(column, df[column])
            schema_dict[column] = dict(spec)
            if notes:
                review_notes[column] = notes
            column_confidence[column] = confidence

        messages: List[str] = []
        review_columns = list(review_notes.keys())
        mode_normalised = str(mode).lower()
        if mode_normalised not in SCHEMA_INFERENCE_MODES:
            raise ValueError(
                "mode must be one of {'silent', 'info', 'interactive'}; "
                f"got {mode!r}"
            )

        if mode_normalised in {"info", "interactive"} and review_notes:
            for name, note in review_notes.items():
                messages.append(f"Column '{name}' flagged for review: {note}")

        if mode_normalised == "interactive":
            browser_schema, browser_message = self._try_browser_schema_builder(df, columns)
            if browser_schema is not None:
                schema_dict = browser_schema.to_dict()
                review_notes = {}
                review_columns = []
                messages.append(
                    "Browser-based schema builder applied; returning the user-adjusted schema."
                )
            else:
                if browser_message:
                    messages.append(browser_message)
                if review_notes:
                    if self._can_launch_gui():
                        updated = self._interactive_review(df, schema_dict, review_notes)
                        if updated:
                            messages.append(
                                "Interactive review applied to: "
                                + ", ".join(sorted(updated))
                            )
                    else:
                        messages.append(
                            "Interactive review not available in the current environment; "
                            "returning the automatically inferred schema."
                        )

        result_schema = Schema(schema_dict)
        return SchemaInferenceResult(
            schema=result_schema,
            mode=mode_normalised,
            review_columns=review_columns,
            column_notes=review_notes,
            messages=messages,
            column_confidence=column_confidence,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _infer_column_schema(
        self, column: str, series: pd.Series
    ) -> Tuple[Mapping[str, object], str, InferenceConfidence]:
        if column in self._categorical_overrides:
            nunique = int(series.dropna().nunique())
            spec = self._categorical_spec(nunique)
            return spec, "Categorical override applied.", InferenceConfidence.HIGH

        if is_bool_dtype(series):
            nunique = int(series.dropna().nunique())
            return (
                self._categorical_spec(nunique),
                "Boolean column coerced to categorical.",
                InferenceConfidence.HIGH,
            )

        non_null = series.dropna()
        if non_null.empty:
            return (
                {"type": "real"},
                "Column empty after removing missing values.",
                InferenceConfidence.LOW,
            )

        if is_numeric_dtype(series):
            return self._infer_numeric_schema(non_null)

        convertible = pd.to_numeric(non_null, errors="coerce")
        coverage = float(convertible.notna().sum()) / float(len(non_null)) if len(non_null) else 0.0
        if coverage >= MIN_NUMERIC_COVERAGE:
            spec, notes, confidence = self._infer_numeric_schema(convertible.dropna())
            extra_note = (
                "Converted from non-numeric values with high numeric coverage."
            )
            full_note = f"{extra_note} {notes}".strip()
            adjusted = _worst_confidence(confidence, InferenceConfidence.MEDIUM)
            return spec, full_note, adjusted

        nunique = int(non_null.nunique())
        note = (
            "Treating as categorical; only "
            f"{coverage:.0%} of values can be safely parsed as numeric."
        )
        return (
            self._categorical_spec(nunique),
            note,
            InferenceConfidence.LOW,
        )

    def _infer_numeric_schema(
        self, series: pd.Series
    ) -> Tuple[Mapping[str, object], str, InferenceConfidence]:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return (
                {"type": "real"},
                "No numeric values available after coercion.",
                InferenceConfidence.LOW,
            )

        nunique = int(numeric.nunique())
        non_missing = numeric.size
        unique_ratio = nunique / non_missing if non_missing else 0.0
        std = float(numeric.std(ddof=0))
        value_range = float(numeric.max() - numeric.min())
        min_value = float(numeric.min())
        max_value = float(numeric.max())
        mean_value = float(numeric.mean()) if non_missing else 0.0
        ratio = (max_value / max(mean_value, 1e-6)) if mean_value > 0 else float("inf")
        skewness = float(numeric.skew()) if nunique > 2 else 0.0
        if np.isnan(skewness):
            skewness = 0.0

        review_reasons: List[str] = []

        def finalize(
            spec: Mapping[str, object],
            *,
            reasons: Optional[List[str]] = None,
            note: str = "",
            forced: Optional[InferenceConfidence] = None,
        ) -> Tuple[Mapping[str, object], str, InferenceConfidence]:
            text_note = note.strip() if note else " ".join(reasons or []).strip()
            if forced is not None:
                confidence = forced
            elif text_note:
                confidence = InferenceConfidence.MEDIUM
            else:
                confidence = InferenceConfidence.HIGH
            return spec, text_note, confidence

        unique_values: Optional[set[float]] = None

        def get_unique_values() -> set[float]:
            nonlocal unique_values
            if unique_values is None:
                unique_values = set(np.unique(numeric))
            return unique_values

        integer_like = bool(
            np.all(np.isfinite(numeric)) and np.allclose(numeric, np.round(numeric))
        )
        if integer_like:
            int_values = np.round(numeric).astype(int)
            sorted_unique = np.sort(np.unique(int_values))
            if sorted_unique.size <= 1:
                contiguous = bool(sorted_unique.size)
                ladder_span = 0
            else:
                diffs = np.diff(sorted_unique)
                contiguous = bool(np.all(diffs == 1))
                ladder_span = int(sorted_unique[-1] - sorted_unique[0])
            limited_range = (
                sorted_unique.size > 0
                and ladder_span <= ORDINAL_RANGE_THRESHOLD
            )
            non_negative = min_value >= 0
            starts_low = sorted_unique.size > 0 and sorted_unique[0] <= 1

            if nunique <= 2 and get_unique_values().issubset(BINARY_VALUES):
                if abs(unique_ratio - NUMERIC_CATEGORICAL_THRESHOLD) <= REVIEW_RATIO_MARGIN:
                    review_reasons.append("Binary ratio near categorical threshold.")
                return finalize(self._categorical_spec(nunique), reasons=review_reasons)

            if (
                non_negative
                and contiguous
                and limited_range
                and nunique <= MAX_CATEGORICAL_UNIQUE
                and unique_ratio <= NUMERIC_CATEGORICAL_THRESHOLD
            ):
                if abs(unique_ratio - NUMERIC_CATEGORICAL_THRESHOLD) <= REVIEW_RATIO_MARGIN:
                    review_reasons.append("Ordinal ladder near categorical ratio boundary.")
                if nunique >= MAX_CATEGORICAL_UNIQUE - REVIEW_UNIQUE_MARGIN:
                    review_reasons.append("Ordinal ladder near size threshold.")
                return finalize(self._ordinal_spec(nunique), reasons=review_reasons)

            if (
                non_negative
                and starts_low
                and (
                    unique_ratio > NUMERIC_CATEGORICAL_THRESHOLD
                    or nunique > MAX_CATEGORICAL_UNIQUE
                )
            ):
                if unique_ratio <= NUMERIC_CATEGORICAL_THRESHOLD + REVIEW_RATIO_MARGIN:
                    review_reasons.append("Count ladder close to categorical ratio boundary.")
                return finalize({"type": "count"}, reasons=review_reasons)

            if (
                non_negative
                and starts_low
                and contiguous
                and ladder_span > ORDINAL_RANGE_THRESHOLD
                and unique_ratio <= NUMERIC_CATEGORICAL_THRESHOLD
            ):
                if abs(unique_ratio - NUMERIC_CATEGORICAL_THRESHOLD) <= REVIEW_RATIO_MARGIN:
                    review_reasons.append("Count ladder close to categorical ratio boundary.")
                return finalize({"type": "count"}, reasons=review_reasons)

            if std <= SMALL_STD_THRESHOLD or value_range <= SMALL_RANGE_THRESHOLD:
                review_reasons.append("Dispersion too small; treating as categorical.")
                return finalize(
                    self._categorical_spec(nunique),
                    reasons=review_reasons,
                    forced=InferenceConfidence.LOW,
                )

            if nunique <= MAX_CATEGORICAL_UNIQUE and unique_ratio <= NUMERIC_CATEGORICAL_THRESHOLD:
                if std >= STD_REAL_FALLBACK or value_range >= RANGE_REAL_FALLBACK:
                    review_reasons.append(
                        "Dispersion suggests continuous behaviour despite low cardinality."
                    )
                    return finalize({"type": "real"}, reasons=review_reasons)
                if (
                    abs(unique_ratio - NUMERIC_CATEGORICAL_THRESHOLD) <= REVIEW_RATIO_MARGIN
                    or nunique >= MAX_CATEGORICAL_UNIQUE - REVIEW_UNIQUE_MARGIN
                ):
                    review_reasons.append("Discrete integer near categorical thresholds.")
                return finalize(self._categorical_spec(nunique), reasons=review_reasons)

            if non_negative and (
                skewness >= POSITIVE_SKEW_THRESHOLD or ratio >= POSITIVE_MAX_MEAN_RATIO
            ):
                if skewness <= POSITIVE_SKEW_THRESHOLD + REVIEW_SKEW_MARGIN:
                    review_reasons.append("Positive skew close to threshold.")
                return finalize({"type": "pos"}, reasons=review_reasons)

            if unique_ratio <= NUMERIC_CATEGORICAL_THRESHOLD + REVIEW_RATIO_MARGIN:
                review_reasons.append("Integer feature near categorical threshold.")
            return finalize({"type": "real"}, reasons=review_reasons)

        if std <= SMALL_STD_THRESHOLD or value_range <= SMALL_RANGE_THRESHOLD:
            review_reasons.append("Dispersion too small; defaulting to categorical.")
            return finalize(
                self._categorical_spec(nunique),
                reasons=review_reasons,
                forced=InferenceConfidence.LOW,
            )

        if nunique <= MAX_CATEGORICAL_UNIQUE and unique_ratio <= NUMERIC_CATEGORICAL_THRESHOLD:
            if std >= STD_REAL_FALLBACK or value_range >= RANGE_REAL_FALLBACK:
                review_reasons.append(
                    "Dispersion suggests continuous behaviour despite low cardinality."
                )
                return finalize({"type": "real"}, reasons=review_reasons)
            if non_missing and unique_ratio <= NUMERIC_CATEGORICAL_THRESHOLD + REVIEW_RATIO_MARGIN:
                review_reasons.append("Binary ratio near categorical threshold.")
            if (
                abs(unique_ratio - NUMERIC_CATEGORICAL_THRESHOLD) <= REVIEW_RATIO_MARGIN
                or nunique >= MAX_CATEGORICAL_UNIQUE - REVIEW_UNIQUE_MARGIN
            ):
                review_reasons.append("Floating feature near categorical thresholds.")
            return finalize(self._categorical_spec(nunique), reasons=review_reasons)

        positive_support = min_value > 0 or (min_value == 0 and max_value > 0)
        if positive_support and (
            skewness >= POSITIVE_SKEW_THRESHOLD or ratio >= POSITIVE_MAX_MEAN_RATIO
        ):
            if skewness <= POSITIVE_SKEW_THRESHOLD + REVIEW_SKEW_MARGIN:
                review_reasons.append("Positive skew close to threshold.")
            return finalize({"type": "pos"}, reasons=review_reasons)

        if abs(unique_ratio - NUMERIC_CATEGORICAL_THRESHOLD) <= REVIEW_RATIO_MARGIN:
            review_reasons.append("Continuous feature near categorical ratio boundary.")

        return finalize({"type": "real"}, reasons=review_reasons)
    # ------------------------------------------------------------------
    # Interactive helpers
    # ------------------------------------------------------------------
    def _try_browser_schema_builder(
        self, df: pd.DataFrame, columns: Iterable[str]
    ) -> Tuple[Optional[Schema], Optional[str]]:
        """Attempt to launch the browser-based schema editor if available."""

        try:  # pragma: no cover - optional dependency
            from .interactive import launch_schema_builder
            from .interactive.schema_builder import SchemaBuilderError
        except Exception:  # pragma: no cover - missing optional dependency
            return None, (
                "Browser-based schema builder unavailable; falling back to Matplotlib review."
            )

        subset = df.loc[:, list(columns)].copy()
        inferencer = SchemaInferencer(categorical_overrides=set(self._categorical_overrides))
        try:
            schema = launch_schema_builder(
                subset,
                feature_columns=columns,
                mode="info",
                inferencer=inferencer,
            )
        except SchemaBuilderError as error:
            return None, f"Browser-based schema builder failed: {error}"
        except Exception as error:  # pragma: no cover - unexpected runtime issues
            return None, f"Browser-based schema builder failed: {error}"

        return schema, None

    @staticmethod
    def _can_launch_gui() -> bool:
        if plt is None or matplotlib is None:
            return False
        backend = matplotlib.get_backend().lower()
        interactive_backends = {
            "qt5agg",
            "qt4agg",
            "tkagg",
            "macosx",
            "gtk3agg",
            "wxagg",
        }
        has_display = bool(os.environ.get("DISPLAY") or sys.platform in {"win32", "darwin"})
        return backend in interactive_backends and has_display

    def _interactive_review(
        self,
        df: pd.DataFrame,
        schema_dict: Dict[str, MutableMapping[str, object]],
        review_notes: Mapping[str, str],
    ) -> List[str]:
        if plt is None:
            return []

        updated: List[str] = []
        for column in review_notes:
            series = df[column].dropna()
            if series.empty:
                continue

            fig, ax = plt.subplots()
            self._plot_distribution(ax, series)
            ax.set_title(f"{column}: inferred {schema_dict[column]['type']}")
            fig.tight_layout()
            plt.show(block=True)
            plt.close(fig)

            while True:
                prompt = (
                    f"Column '{column}' inferred as {schema_dict[column]['type']}. "
                    "Enter a new type [real/pos/count/cat/ordinal] or press Enter to accept: "
                )
                user_input = input(prompt).strip().lower()
                if not user_input:
                    break
                if user_input in {"real", "pos", "count", "cat", "ordinal"}:
                    schema_dict[column] = self._spec_from_type(user_input, series)
                    updated.append(column)
                    break
                print("Unrecognised type. Please enter one of: real, pos, count, cat, ordinal.")

        return updated

    @staticmethod
    def _plot_distribution(ax, series: pd.Series) -> None:
        if is_numeric_dtype(series):
            ax.hist(series, bins=min(30, max(5, int(np.sqrt(series.size)))), color="tab:blue", alpha=0.75)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
        else:
            counts = series.astype(str).value_counts().sort_index()
            positions = np.arange(len(counts.index))
            ax.bar(positions, counts.values, color="tab:blue", alpha=0.75)
            ax.set_xticks(positions)
            ax.set_xticklabels(counts.index, rotation=45, ha="right")
            ax.set_ylabel("Count")

    @staticmethod
    def _spec_from_type(column_type: str, series: pd.Series) -> Mapping[str, object]:
        if column_type in {"cat", "ordinal"}:
            nunique = int(series.dropna().nunique())
            return {"type": column_type, "n_classes": max(nunique, 2)}
        return {"type": column_type}

    @staticmethod
    def _categorical_spec(nunique: int) -> Mapping[str, object]:
        return {"type": "cat", "n_classes": max(int(nunique), 2)}

    @staticmethod
    def _ordinal_spec(nunique: int) -> Mapping[str, object]:
        return {"type": "ordinal", "n_classes": max(int(nunique), 2)}


__all__ = [
    "SchemaInferencer",
    "SchemaInferenceResult",
    "InferenceConfidence",
    "SCHEMA_INFERENCE_MODES",
]
