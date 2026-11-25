"""Shared matplotlib styling for Olink plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def set_plot_theme(ax: Axes | None = None, font: str = "DejaVu Sans") -> Axes:
    """Apply a light, publication-style theme to ``ax`` or the current axes."""
    target = ax or plt.gca()
    target.spines["top"].set_visible(False)
    target.spines["right"].set_visible(False)
    target.tick_params(labelsize=9)
    target.set_axisbelow(True)
    target.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#cccccc")
    for label in target.get_xticklabels() + target.get_yticklabels():
        label.set_fontname(font)
    if target.xaxis.label:
        target.xaxis.label.set_fontname(font)
    if target.yaxis.label:
        target.yaxis.label.set_fontname(font)
    if target.title:
        target.title.set_fontname(font)
    return target


def apply_theme_to_figure(fig: Figure, font: str = "DejaVu Sans") -> Figure:
    for ax in fig.axes:
        set_plot_theme(ax, font=font)
    fig.tight_layout()
    return fig
