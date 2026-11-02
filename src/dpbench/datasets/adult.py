"""UCI Adult census income dataset (tabular, binary task, sensitive attributes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dpbench.core.registry import register_dataset
from dpbench.core.types import Dataset
from dpbench.datasets.base import DatasetLoader, DownloadError, cache_dir, download

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

# Fixed bounds taken from the dataset documentation (census top-coding), not computed on the
# training data: preprocessing therefore reveals nothing about the private records.
NUMERIC = {
    "age": (17.0, 90.0, "linear"),
    "fnlwgt": (1.0e4, 1.5e6, "log"),
    "education_num": (1.0, 16.0, "linear"),
    "capital_gain": (0.0, 1.0e5, "log1p"),
    "capital_loss": (0.0, 5.0e3, "log1p"),
    "hours_per_week": (1.0, 99.0, "linear"),
}

# Category lists of the UCI documentation (adult.names), so the one-hot encoding is public too.
CATEGORIES = {
    "workclass": [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ],
    "education": [
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ],
    "marital_status": [
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native_country": [
        "United-States",
        "Cambodia",
        "England",
        "Puerto-Rico",
        "Canada",
        "Germany",
        "Outlying-US(Guam-USVI-etc)",
        "India",
        "Japan",
        "Greece",
        "South",
        "China",
        "Cuba",
        "Iran",
        "Honduras",
        "Philippines",
        "Italy",
        "Poland",
        "Jamaica",
        "Vietnam",
        "Mexico",
        "Portugal",
        "Ireland",
        "France",
        "Dominican-Republic",
        "Laos",
        "Ecuador",
        "Taiwan",
        "Haiti",
        "Columbia",
        "Hungary",
        "Guatemala",
        "Nicaragua",
        "Scotland",
        "Thailand",
        "Yugoslavia",
        "El-Salvador",
        "Trinadad&Tobago",
        "Peru",
        "Hong",
        "Holand-Netherlands",
    ],
}

CSV_MIRRORS = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv",)
CSV_MD5 = "779a6204aa629e559c8f48d0897526cf"
UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"


def _read(path: Path, skiprows: int = 0) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, names=COLUMNS, na_values="?", skipinitialspace=True, skiprows=skiprows)
    frame["income"] = frame["income"].astype(str).str.rstrip(".")
    return frame


def read_adult(directory: Path | None = None) -> pd.DataFrame:
    """The 48,842 raw records (the UCI training and test files together)."""
    directory = directory or cache_dir("adult")
    try:
        return _read(download("adult-all.csv", CSV_MIRRORS, directory, md5=CSV_MD5))
    except DownloadError as csv_error:
        try:
            train = _read(download("adult.data", [UCI_BASE], directory))
            test = _read(download("adult.test", [UCI_BASE], directory), skiprows=1)
        except DownloadError:
            raise csv_error from None
        return pd.concat([train, test], ignore_index=True)


def encode_features(frame: pd.DataFrame, drop: tuple[str, ...] = ()) -> tuple[np.ndarray, list[str]]:
    """Numeric features scaled to [0, 1] with fixed bounds, categorical features one-hot encoded."""
    columns, names = [], []
    for column, (low, high, transform) in NUMERIC.items():
        if column in drop:
            continue
        values = frame[column].to_numpy(dtype=np.float64).clip(low, high)
        if transform == "log":
            values, low, high = np.log(values), np.log(low), np.log(high)
        elif transform == "log1p":
            values, low, high = np.log1p(values), np.log1p(low), np.log1p(high)
        columns.append(((values - low) / (high - low))[:, None])
        names.append(column)
    for column, categories in CATEGORIES.items():
        if column in drop:
            continue
        values = frame[column].to_numpy()
        columns.append(np.stack([values == c for c in categories], axis=1).astype(np.float64))
        names.extend(f"{column}={c}" for c in categories)
    return np.concatenate(columns, axis=1).astype(np.float32), names


@register_dataset("adult")
class Adult(DatasetLoader):
    """UCI Adult census income: 45,222 records with no missing value, task income > 50K, sensitive attribute sex or race.

    Numeric features are clipped to documented bounds and scaled to [0, 1]; categorical
    features are one-hot encoded with the documented categories (105 features in total).
    """

    def __init__(
        self,
        sensitive_attribute: str = "sex",
        drop_sensitive: bool = False,
        test_size: float = 0.2,
        split_seed: int = 0,
        n_train: int | None = None,
        n_test: int | None = None,
        subset_seed: int = 0,
    ) -> None:
        super().__init__(n_train, n_test, subset_seed)
        if sensitive_attribute not in ("sex", "race"):
            raise ValueError(f"sensitive_attribute must be 'sex' or 'race', got {sensitive_attribute!r}")
        self.sensitive_attribute = sensitive_attribute
        self.drop_sensitive = drop_sensitive
        self.test_size = test_size
        self.split_seed = split_seed

    def load_full(self) -> Dataset:
        frame = read_adult().dropna().reset_index(drop=True)
        drop = (self.sensitive_attribute,) if self.drop_sensitive else ()
        X, names = encode_features(frame, drop)
        y = (frame["income"] == ">50K").to_numpy(dtype=np.int64)
        group_names = CATEGORIES[self.sensitive_attribute]
        groups = frame[self.sensitive_attribute].map({g: i for i, g in enumerate(group_names)}).to_numpy()
        X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
            X, y, groups, test_size=self.test_size, random_state=self.split_seed, stratify=y
        )
        return Dataset(
            name="adult",
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            y_test=y_te,
            n_classes=2,
            class_names=["<=50K", ">50K"],
            feature_names=names,
            sensitive_attribute=self.sensitive_attribute,
            groups_train=g_tr,
            groups_test=g_te,
            group_names=list(group_names),
            description="UCI Adult census income (records with missing values removed)",
            source="UCI Machine Learning Repository (Becker & Kohavi, 1996)",
        )
