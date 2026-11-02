"""Classification datasets uploaded in the dashboard (a CSV file, its schema and a target column).

The features are encoded with the *schema*, not with statistics of the data: numeric
columns are clipped to their bounds and scaled to [0, 1], categorical columns are one-hot
encoded with their list of categories. A missing numeric value becomes 0.5 (the middle of
the range) and a missing or unknown category a row of zeros.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dpbench.core.registry import DATASETS, register_dataset
from dpbench.core.schema import CATEGORICAL, NUMERIC, TableSchema, category_codes
from dpbench.core.types import Dataset
from dpbench.datasets.base import DatasetLoader


def encode_features(
    frame: pd.DataFrame, schema: TableSchema, exclude: tuple[str, ...] = ()
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix (float32) and feature names of a data frame, from its schema."""
    columns, names = [], []
    for name, spec in schema.columns.items():
        if name in exclude or name not in frame:
            continue
        if spec.kind == NUMERIC:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
            scaled = (np.clip(values, spec.lower, spec.upper) - spec.lower) / max(spec.width, 1e-12)
            columns.append(np.where(np.isnan(scaled), 0.5, scaled)[:, None])
            names.append(name)
        elif spec.kind == CATEGORICAL:
            codes = category_codes(frame, spec)
            one_hot = np.zeros((len(frame), len(spec.categories or [])))
            known = codes >= 0
            one_hot[np.flatnonzero(known), codes[known]] = 1.0
            columns.append(one_hot)
            names.extend(f"{name}={c}" for c in spec.categories or [])
    if not columns:
        raise ValueError("no feature: every column is the target or ignored")
    return np.concatenate(columns, axis=1).astype(np.float32), names


class UploadedDataset(DatasetLoader):
    """A dataset uploaded in the dashboard. The target column must be categorical."""

    csv_path: ClassVar[Path]
    schema_data: ClassVar[dict[str, Any]]
    description: ClassVar[str] = ""

    def __init__(
        self,
        test_size: float = 0.2,
        split_seed: int = 0,
        n_train: int | None = None,
        n_test: int | None = None,
        subset_seed: int = 0,
    ) -> None:
        super().__init__(n_train, n_test, subset_seed)
        self.test_size = test_size
        self.split_seed = split_seed

    def load_full(self) -> Dataset:
        frame = pd.read_csv(self.csv_path)
        schema = TableSchema.from_dict(self.schema_data)
        target = schema.target
        if target is None or schema.column(target).kind != CATEGORICAL:
            raise ValueError(f"dataset {self.name!r}: the target must be a categorical column")
        target_spec = schema.column(target)
        labels = category_codes(frame, target_spec)
        frame = frame[labels >= 0].reset_index(drop=True)
        labels = labels[labels >= 0]
        X, names = encode_features(frame, schema, exclude=(target,))
        groups = group_names = None
        sensitive = schema.sensitive_attribute
        if sensitive and sensitive != target and schema.column(sensitive).kind == CATEGORICAL:
            group_names = list(schema.column(sensitive).categories or [])
            groups = np.maximum(category_codes(frame, schema.column(sensitive)), 0)
        counts = np.bincount(labels)
        stratify = labels if len(labels) and counts[counts > 0].min() >= 2 else None
        arrays = [X, labels] + ([groups] if groups is not None else [])
        split = train_test_split(*arrays, test_size=self.test_size, random_state=self.split_seed, stratify=stratify)
        X_tr, X_te, y_tr, y_te = split[:4]
        g_tr, g_te = (split[4], split[5]) if groups is not None else (None, None)
        return Dataset(
            name=self.name,
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            y_test=y_te,
            n_classes=len(target_spec.categories or []),
            class_names=list(target_spec.categories or []),
            feature_names=names,
            sensitive_attribute=sensitive if groups is not None else None,
            groups_train=g_tr,
            groups_test=g_te,
            group_names=group_names,
            description=self.description or f"Uploaded file {self.csv_path.name}",
            source=f"uploaded file {self.csv_path.name}",
        )


def register_uploaded_dataset(name: str, csv_path: Path, schema: dict[str, Any], description: str = "") -> type:
    """Register (or replace) the dataset plugin of an uploaded file."""
    cls = type(
        f"UploadedDataset_{name}",
        (UploadedDataset,),
        {"csv_path": Path(csv_path), "schema_data": schema, "description": description, "__doc__": description},
    )
    DATASETS.unregister(name)
    return register_dataset(name)(cls)
