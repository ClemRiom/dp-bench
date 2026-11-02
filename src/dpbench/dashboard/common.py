"""Helpers shared by the dashboard pages: charts and tables with DP-Bench's styling."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PLOTLY_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "svg"}}


def plot(fig: go.Figure, key: str | None = None) -> None:
    """``st.plotly_chart`` keeping DP-Bench's styling, with a fallback for older Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None, config=PLOTLY_CONFIG, key=key)


def table(frame: pd.DataFrame | Any, **kwargs: Any) -> None:
    kwargs.setdefault("hide_index", True)
    try:
        st.dataframe(frame, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(frame, use_container_width=True, **kwargs)
