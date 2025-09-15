"""Simple schema definitions for tabular data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ColumnSchema:
    name: str
    kind: str  # continuous, binary, categorical, count
    categories: Optional[List[str]] = None


@dataclass
class TableSchema:
    columns: List[ColumnSchema]

    def feature_names(self) -> List[str]:
        return [c.name for c in self.columns]
