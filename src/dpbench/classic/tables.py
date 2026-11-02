"""Tables: the databases that the classic-DP queries are asked on.

A table is a pandas data frame plus its :class:`~dpbench.core.schema.TableSchema` (the
public domain of every column). Built-in tables: a large synthetic census
(20,000 people) and the small Diabetes study (442 patients) to see what DP costs on a
small database.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from dpbench.core.registry import register_table
from dpbench.core.schema import CATEGORICAL, NUMERIC, ColumnSpec, TableSchema


@dataclass
class Table:
    """A database: its records and the public domain of its columns."""

    name: str
    frame: pd.DataFrame
    schema: TableSchema
    description: str = ""
    source: str = ""

    @property
    def n(self) -> int:
        return len(self.frame)

    def column(self, name: str) -> ColumnSpec:
        return self.schema.column(name)

    def numeric_columns(self) -> list[str]:
        return self.schema.names(NUMERIC)

    def categorical_columns(self) -> list[str]:
        return self.schema.names(CATEGORICAL)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "records": self.n,
            "numeric_columns": self.numeric_columns(),
            "categorical_columns": self.categorical_columns(),
            "bounds_from_data": self.schema.bounds_from_data,
        }


class TableLoader(ABC):
    """Base class of the table plugins: :meth:`load` returns a :class:`Table`."""

    name: ClassVar[str]

    @abstractmethod
    def load(self) -> Table:
        """The whole table."""


REGIONS = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "Centre"]
EDUCATION = ["None", "Primary", "Secondary", "Vocational", "Bachelor", "Master or more"]


@register_table("census_synthetic")
class SyntheticCensus(TableLoader):
    """Synthetic census generated on the fly (default 20,000 people): age, sex, region, education, hours, income; offline."""

    def __init__(self, n: int = 20_000, seed: int = 0) -> None:
        self.n = int(n)
        self.seed = int(seed)

    def load(self) -> Table:
        rng = np.random.default_rng(self.seed)
        n = self.n
        age = np.clip(rng.gamma(9.0, 4.6, n), 18, 95).round()
        sex = rng.choice(["Female", "Male"], n, p=[0.51, 0.49])
        region = rng.choice(REGIONS, n, p=[0.22, 0.08, 0.1, 0.1, 0.16, 0.1, 0.14, 0.1])
        education = rng.choice(EDUCATION, n, p=[0.04, 0.12, 0.38, 0.18, 0.17, 0.11])
        level = np.array([EDUCATION.index(e) for e in education])
        employed = rng.random(n) < np.where((age > 22) & (age < 65), 0.82, 0.25)
        hours = np.where(employed, np.clip(rng.normal(37 + 2 * (sex == "Male"), 8, n), 1, 99), 0).round()
        income = np.where(
            employed,
            np.exp(rng.normal(10.0 + 0.18 * level + 0.012 * np.minimum(age, 60), 0.55, n)),
            np.exp(rng.normal(9.0, 0.8, n)),
        )
        frame = pd.DataFrame(
            {
                "age": age,
                "sex": sex,
                "region": region,
                "education": education,
                "employed": np.where(employed, "yes", "no"),
                "hours_per_week": hours,
                "income": np.clip(income, 0, 500_000).round(),
            }
        )
        columns = {
            "age": ColumnSpec("age", "numeric", lower=18, upper=95),
            "sex": ColumnSpec("sex", "categorical", categories=["Female", "Male"]),
            "region": ColumnSpec("region", "categorical", categories=REGIONS),
            "education": ColumnSpec("education", "categorical", categories=EDUCATION),
            "employed": ColumnSpec("employed", "categorical", categories=["no", "yes"]),
            "hours_per_week": ColumnSpec("hours_per_week", "numeric", lower=0, upper=99),
            "income": ColumnSpec("income", "numeric", lower=0, upper=500_000),
        }
        schema = TableSchema(columns, target="employed", sensitive_attribute="sex", notes=["Domains fixed a priori."])
        return Table(
            "census_synthetic",
            frame,
            schema,
            f"Synthetic census of {n:,} people (generated, seed {self.seed})",
            "DP-Bench generator",
        )


@register_table("diabetes")
class DiabetesTable(TableLoader):
    """Diabetes study bundled with scikit-learn: 442 patients only (age, sex, BMI, blood pressure, disease progression)."""

    def load(self) -> Table:
        from sklearn.datasets import load_diabetes

        data = load_diabetes(scaled=False)
        frame = pd.DataFrame(data.data[:, :4], columns=["age", "sex", "bmi", "blood_pressure"])
        frame["sex"] = np.where(frame["sex"] == 1, "1", "2")
        frame["progression"] = data.target
        columns = {
            "age": ColumnSpec("age", "numeric", lower=18, upper=90),
            "sex": ColumnSpec("sex", "categorical", categories=["1", "2"]),
            "bmi": ColumnSpec("bmi", "numeric", lower=15, upper=50),
            "blood_pressure": ColumnSpec("blood_pressure", "numeric", lower=50, upper=150),
            "progression": ColumnSpec("progression", "numeric", lower=0, upper=400),
        }
        schema = TableSchema(columns, notes=["Plausible human ranges fixed a priori, not computed on the data."])
        return Table(
            "diabetes",
            frame,
            schema,
            "Diabetes study (Efron et al. 2004): 442 patients",
            "scikit-learn load_diabetes",
        )
