"""Colours and labels shared by the figures and the dashboard."""

from __future__ import annotations

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#7b4fd6", "#e34948", "#c99000", "#e87ba4", "#008300", "#0f8fa8"]
GRID_COLOR = "#ececec"
FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def eps_text(epsilon: float) -> str:
    return "∞" if epsilon == float("inf") else f"{epsilon:g}"
