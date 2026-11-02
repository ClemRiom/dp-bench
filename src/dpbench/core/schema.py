"""Schema of a table: the type and the public domain of every column.

Differential privacy needs to know the domain of the data *before* looking at it: the
sensitivity of a sum depends on the bounds of the column, a histogram needs the list of
categories, a one-hot encoding needs it too. A :class:`TableSchema` records this public
knowledge. When the bounds are computed from the data itself, ``bounds_from_data`` is set:
such bounds leak a little information about the records, and the dashboard says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NUMERIC = "numeric"
CATEGORICAL = "categorical"
IGNORED = "ignored"
KINDS = (NUMERIC, CATEGORICAL, IGNORED)


@dataclass
class ColumnSpec:
    """Type and public domain of one column."""

    name: str
    kind: str
    lower: float | None = None
    upper: float | None = None
    categories: list[str] | None = None
    bounds_from_data: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"column {self.name!r}: kind must be one of {KINDS}, got {self.kind!r}")
        if self.kind == NUMERIC:
            if self.lower is None or self.upper is None:
                raise ValueError(f"numeric column {self.name!r} needs a lower and an upper bound")
            self.lower, self.upper = float(self.lower), float(self.upper)
            if not (math.isfinite(self.lower) and math.isfinite(self.upper)) or self.lower > self.upper:
                raise ValueError(f"column {self.name!r}: invalid bounds [{self.lower}, {self.upper}]")
        if self.kind == CATEGORICAL:
            if not self.categories:
                raise ValueError(f"categorical column {self.name!r} needs its list of categories")
            self.categories = [str(c) for c in self.categories]

    @property
    def width(self) -> float:
        """``upper - lower`` for a numeric column."""
        return float(self.upper - self.lower) if self.kind == NUMERIC else math.nan

    @property
    def magnitude(self) -> float:
        """Largest absolute value of the domain: the sensitivity of a sum (add or remove one record)."""
        return max(abs(self.lower), abs(self.upper)) if self.kind == NUMERIC else math.nan


@dataclass
class TableSchema:
    """The columns of a table, and optionally the target and the sensitive attribute for machine learning."""

    columns: dict[str, ColumnSpec]
    target: str | None = None
    sensitive_attribute: str | None = None
    notes: list[str] = field(default_factory=list)

    def column(self, name: str) -> ColumnSpec:
        if name not in self.columns:
            raise KeyError(f"unknown column {name!r}; columns: {', '.join(self.columns)}")
        return self.columns[name]

    def names(self, kind: str | None = None) -> list[str]:
        return [n for n, c in self.columns.items() if kind is None or c.kind == kind]

    @property
    def bounds_from_data(self) -> list[str]:
        return [n for n, c in self.columns.items() if c.bounds_from_data]


def numeric_values(frame: pd.DataFrame, column: ColumnSpec) -> np.ndarray:
    """The values of a numeric column as float64, clipped to the column's bounds (missing values dropped)."""
    values = pd.to_numeric(frame[column.name], errors="coerce").dropna().to_numpy(dtype=np.float64)
    return np.clip(values, column.lower, column.upper)


def category_codes(frame: pd.DataFrame, column: ColumnSpec) -> np.ndarray:
    """Index of each record's category in ``column.categories`` (-1 for a missing or unknown value)."""
    mapping = {c: i for i, c in enumerate(column.categories or [])}
    return frame[column.name].astype(str).map(mapping).fillna(-1).to_numpy(dtype=np.int64)
