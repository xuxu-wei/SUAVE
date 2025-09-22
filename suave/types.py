"""Type definitions and helpers for the SUAVE package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a single column in the :class:`Schema`.

    Parameters
    ----------
    type:
        Logical feature type. Supported values mirror the unsupervised branch of
        the model: ``"real"`` (Gaussian), ``"pos"`` (log-normal),
        ``"count"`` (Poisson), ``"cat"`` (categorical) and ``"ordinal"``
        (cumulative link).
    n_classes:
        Number of distinct categories for categorical features. It must be
        provided when ``type`` is ``"cat"``.
    """

    type: str
    n_classes: Optional[int] = None
    y_dim: Optional[int] = None


class Schema:
    """Container describing the tabular data layout expected by :class:`SUAVE`.

    The schema is intentionally lightweight: users provide a mapping from
    column name to a :class:`ColumnSpec`.  Automatic inference is outside the
    scope of this first iteration, which keeps the public API explicit and
    predictable.

    Parameters
    ----------
    columns:
        Mapping from column names to ``dict`` (or :class:`ColumnSpec`) objects
        containing the ``type`` key and, for categorical features, the
        ``n_classes`` key.

    Examples
    --------
    >>> from suave.types import Schema
    >>> schema = Schema(
    ...     {
    ...         "age": {"type": "real"},
    ...         "gender": {"type": "cat", "n_classes": 2},
    ...     }
    ... )
    >>> schema.feature_names
    ['age', 'gender']
    >>> schema.categorical_features
    ['gender']
    """

    _SUPPORTED_TYPES = {"real", "cat", "pos", "count", "ordinal"}

    def __init__(self, columns: Mapping[str, Mapping[str, object]]):
        self._columns: Dict[str, ColumnSpec] = {}
        for name, raw_spec in columns.items():
            spec_dict = dict(raw_spec)
            column_type = spec_dict.get("type")
            if column_type not in self._SUPPORTED_TYPES:
                raise ValueError(
                    f"Unsupported column type '{column_type}' for '{name}'. "
                    "Supported types: real, pos, count, cat, ordinal."
                )
            n_classes = spec_dict.get("n_classes")
            if column_type in {"cat", "ordinal"}:
                if n_classes is None:
                    raise ValueError(
                        f"Column '{name}' of type '{column_type}' requires 'n_classes'."
                    )
                if int(n_classes) <= 1:
                    raise ValueError(
                        f"Column '{name}' of type '{column_type}' must declare "
                        "'n_classes' greater than 1."
                    )
                n_classes = int(n_classes)
            elif n_classes is not None:
                raise ValueError(
                    f"Column '{name}' of type '{column_type}' should not provide 'n_classes'."
                )
            y_dim = spec_dict.get("y_dim")
            if y_dim is not None:
                y_dim = int(y_dim)
                if y_dim <= 0:
                    raise ValueError(
                        f"Column '{name}' must declare a positive 'y_dim' share"
                    )
            self._columns[name] = ColumnSpec(
                type=column_type, n_classes=n_classes, y_dim=y_dim
            )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def feature_names(self) -> Iterable[str]:
        """Return an iterable with the schema's feature names."""

        return tuple(self._columns.keys())

    @property
    def real_features(self) -> Iterable[str]:
        """Names of columns declared as ``"real"``."""

        return tuple(
            name for name, spec in self._columns.items() if spec.type == "real"
        )

    @property
    def categorical_features(self) -> Iterable[str]:
        """Names of columns declared as ``"cat"``."""

        return tuple(name for name, spec in self._columns.items() if spec.type == "cat")

    @property
    def positive_features(self) -> Iterable[str]:
        """Names of columns declared as ``"pos"``."""

        return tuple(name for name, spec in self._columns.items() if spec.type == "pos")

    @property
    def count_features(self) -> Iterable[str]:
        """Names of columns declared as ``"count"``."""

        return tuple(
            name for name, spec in self._columns.items() if spec.type == "count"
        )

    @property
    def ordinal_features(self) -> Iterable[str]:
        """Names of columns declared as ``"ordinal"``."""

        return tuple(
            name for name, spec in self._columns.items() if spec.type == "ordinal"
        )

    # ------------------------------------------------------------------
    # Mapping protocol helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, MutableMapping[str, object]]:
        """Return a JSON-serialisable representation of the schema."""

        return {
            name: {
                "type": spec.type,
                **({"n_classes": spec.n_classes} if spec.n_classes else {}),
                **({"y_dim": spec.y_dim} if spec.y_dim else {}),
            }
            for name, spec in self._columns.items()
        }

    def __contains__(self, item: str) -> bool:  # pragma: no cover - trivial
        return item in self._columns

    def __getitem__(self, item: str) -> ColumnSpec:  # pragma: no cover - trivial
        return self._columns[item]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._columns)

    def __iter__(self):  # pragma: no cover - trivial
        return iter(self._columns)

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------
    def update(self, other: Mapping[str, Mapping[str, object]]) -> None:
        """Update the schema with additional column specifications."""

        for name, spec in other.items():
            self._columns[name] = Schema({name: spec})[name]

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def require_columns(self, columns: Iterable[str]) -> None:
        """Validate that all ``columns`` are present in the schema."""

        missing = [column for column in columns if column not in self._columns]
        if missing:
            raise KeyError(f"Columns {missing} missing from schema")

    # ------------------------------------------------------------------
    # Latent partition helpers
    # ------------------------------------------------------------------
    def y_dimensions(self) -> Dict[str, int]:
        """Return the latent ``y`` allocation for every column."""

        return {
            name: (spec.y_dim if spec.y_dim is not None else self._default_y_dim(spec))
            for name, spec in self._columns.items()
        }

    @staticmethod
    def _default_y_dim(spec: ColumnSpec) -> int:
        """Return the fallback ``y`` share used when none is provided."""

        if spec.type in {"real", "pos", "count"}:
            return 1
        if spec.type in {"cat", "ordinal"}:
            if spec.n_classes is None:
                raise ValueError(
                    "Categorical and ordinal columns require 'n_classes' to infer y_dim"
                )
            return int(spec.n_classes)
        raise ValueError(f"Unsupported column type '{spec.type}'")
