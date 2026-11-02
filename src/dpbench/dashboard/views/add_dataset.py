"""Add a dataset: drag and drop (or browse for) a data file, describe its columns, save it in the workspace."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from dpbench import workspace
from dpbench.core.schema import CATEGORICAL, IGNORED, KINDS, NUMERIC, ColumnSpec, TableSchema, infer_column
from dpbench.dashboard import common

FILE_TYPES = ["csv", "tsv", "txt", "xlsx", "xls", "parquet", "json"]


def read_file(name: str, data: bytes, separator: str) -> pd.DataFrame:
    """Read an uploaded file into a data frame (the separator of a text file is guessed when 'auto')."""
    suffix = Path(name).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        try:
            return pd.read_excel(io.BytesIO(data))
        except ImportError as error:
            raise ImportError("reading Excel files needs openpyxl: pip install openpyxl") from error
    if suffix == ".parquet":
        return pd.read_parquet(io.BytesIO(data))
    if suffix == ".json":
        return pd.read_json(io.BytesIO(data))
    text = data.decode("utf-8-sig", errors="replace")
    if separator == "auto":
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    return pd.read_csv(io.StringIO(text), sep={"comma": ",", "semicolon": ";", "tab": "\t"}[separator])


def suggested_name(filename: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(filename).stem).strip("_").lower() or "my_dataset"
    return stem if stem[0].isalpha() else f"data_{stem}"


def _schema_editor(frame: pd.DataFrame, key: str) -> TableSchema | None:
    inferred = {str(c): infer_column(str(c), frame[c]) for c in frame.columns}
    rows = []
    for name, spec in inferred.items():
        rows.append(
            {
                "Column": name,
                "Type": spec.kind,
                "Lower bound": spec.lower,
                "Upper bound": spec.upper,
                "Categories": len(spec.categories or []) if spec.kind == CATEGORICAL else None,
                "Example values": ", ".join(map(str, frame[name].dropna().astype(str).unique()[:4])),
            }
        )
    edited = st.data_editor(
        pd.DataFrame(rows),
        hide_index=True,
        disabled=["Column", "Categories", "Example values"],
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=list(KINDS), required=True),
            "Lower bound": st.column_config.NumberColumn("Lower bound"),
            "Upper bound": st.column_config.NumberColumn("Upper bound"),
        },
        key=key,
        width="stretch",
    )
    columns: dict[str, ColumnSpec] = {}
    problems = []
    for _, row in edited.iterrows():
        name, kind = row["Column"], row["Type"]
        guess = inferred[name]
        try:
            if kind == NUMERIC:
                values = pd.to_numeric(frame[name], errors="coerce")
                if values.notna().sum() == 0:
                    raise ValueError("no numeric value")
                lower = float(row["Lower bound"]) if pd.notna(row["Lower bound"]) else float(values.min())
                upper = float(row["Upper bound"]) if pd.notna(row["Upper bound"]) else float(values.max())
                from_data = guess.kind == NUMERIC and guess.lower == lower and guess.upper == upper
                columns[name] = ColumnSpec(
                    name, NUMERIC, lower=lower, upper=upper, bounds_from_data=from_data or guess.kind != NUMERIC
                )
            elif kind == CATEGORICAL:
                categories = (
                    guess.categories if guess.kind == CATEGORICAL else sorted(frame[name].dropna().astype(str).unique())
                )
                if len(categories) > 500:
                    raise ValueError(f"{len(categories)} distinct values: too many for a categorical column")
                columns[name] = ColumnSpec(name, CATEGORICAL, categories=list(categories), bounds_from_data=True)
            else:
                columns[name] = ColumnSpec(name, IGNORED)
        except ValueError as error:
            problems.append(f"`{name}`: {error}")
    if problems:
        st.error("Fix these columns: " + "; ".join(problems))
        return None
    return TableSchema(columns, notes=["Uploaded in the dashboard; domains reviewed by the user."])


def _existing() -> None:
    datasets = workspace.list_datasets()
    st.subheader("Datasets in the workspace")
    if not datasets:
        st.info("No dataset added yet.")
        return
    for meta in datasets:
        with st.container(border=True):
            left, right = st.columns([5, 1])
            target = (meta.get("schema") or {}).get("target")
            uses = "classic DP queries" + (f" and machine learning (target `{target}`)" if target else "")
            left.markdown(
                f"**{meta['name']}** — {meta.get('rows', '?'):,} rows · usable for {uses}"
                + (f"  \n{meta['description']}" if meta.get("description") else "")
                + (f"  \n:red[{meta['error']}]" if meta.get("error") else "")
            )
            if right.button("Delete", key=f"delete_dataset_{meta['name']}"):
                workspace.delete_dataset(meta["name"])
                st.cache_data.clear()
                st.rerun()


def render() -> None:
    st.title("Add a dataset")
    uploaded = st.file_uploader("Data file (CSV, TSV, Excel, Parquet or JSON)", type=FILE_TYPES)
    if uploaded is None:
        _existing()
        return
    separator = "auto"
    if Path(uploaded.name).suffix.lower() in (".csv", ".tsv", ".txt"):
        separator = st.segmented_control("Separator", ["auto", "comma", "semicolon", "tab"], default="auto") or "auto"
    try:
        frame = read_file(uploaded.name, uploaded.getvalue(), separator)
    except Exception as error:
        st.error(f"Cannot read {uploaded.name}: {error}")
        return
    frame.columns = [str(c).strip() for c in frame.columns]
    st.markdown(f"**{uploaded.name}**: {len(frame):,} rows × {frame.shape[1]} columns")
    common.table(frame.head(10))
    st.markdown("#### Columns and their public domains")
    schema = _schema_editor(frame, key=f"schema_{uploaded.name}")
    if schema is None:
        return
    categorical = schema.names(CATEGORICAL)
    c1, c2 = st.columns(2)
    target = c1.selectbox("Target column (machine learning)", ["(none: classic DP only)", *categorical])
    sensitive = c2.selectbox("Sensitive attribute (fairness metrics, optional)", ["(none)", *categorical])
    schema.target = None if target.startswith("(none") else target
    schema.sensitive_attribute = None if sensitive == "(none)" or sensitive == target else sensitive
    c1, c2 = st.columns(2)
    name = c1.text_input("Name (letters, digits and _)", suggested_name(uploaded.name))
    description = c2.text_input("Description", f"Uploaded from {uploaded.name}")
    from_data = schema.bounds_from_data
    if from_data:
        st.warning(
            f"{len(from_data)} column(s) keep domains computed from the data ({', '.join(from_data[:8])}"
            f"{'...' if len(from_data) > 8 else ''}): fine to experiment, but not fixed in advance.",
        )
    if schema.target:
        counts = frame[schema.target].astype(str).value_counts()
        st.markdown(f"Classes of `{schema.target}`: " + ", ".join(f"{k} ({v})" for k, v in counts.head(10).items()))
    existing = {m["name"] for m in workspace.list_datasets()}
    overwrite = False
    if name in existing:
        overwrite = st.checkbox(f"Replace the existing dataset {name!r}", value=False)
    if st.button("Save to the workspace", type="primary", disabled=name in existing and not overwrite):
        try:
            keep = [c for c in frame.columns if schema.columns[c].kind != IGNORED]
            meta = workspace.save_dataset(name, frame[keep], schema, description, uploaded.name, overwrite=overwrite)
        except Exception as error:
            st.error(f"Not saved: {error}")
            return
        st.cache_data.clear()
        uses = " and ".join(f"**{kind}** `{name}`" for kind in meta["registered_as"])
        st.success(f"Saved. Available as {uses}.")
        if "dataset" in meta["registered_as"]:
            from dpbench.core.registry import DATASETS

            try:
                data = DATASETS.get(name)().load()
                st.markdown(
                    f"For training: {data.n_train} training and {data.n_test} test examples, "
                    f"{data.n_features} features after encoding, {data.n_classes} classes."
                )
            except Exception as error:
                st.error(f"The machine-learning view of this dataset fails: {error}")
    _existing()
