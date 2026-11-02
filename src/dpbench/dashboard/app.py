"""DP-Bench dashboard (Streamlit). Start it with ``dpbench dashboard``.

At this step the dashboard has one page: queries on databases (classic DP).
"""

from __future__ import annotations

import streamlit as st

from dpbench import __version__
from dpbench.dashboard.views import classic

st.set_page_config(page_title="DP-Bench", page_icon=":material/shield_lock:", layout="wide")

navigation = st.navigation(
    {
        "Classic DP": [
            st.Page(classic.render, title="Queries on databases", icon=":material/database:", url_path="classic")
        ],
    }
)
st.sidebar.caption(f"DP-Bench {__version__} · benchmark of differentially private algorithms")
navigation.run()
