"""Olink color palette."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.colors as mcolors
import numpy as np

OLINK_COLORS = {
    "red": "#FE1F04",
    "orange": "#FF8C22",
    "yellow": "#FFC700",
    "green": "#27AE55",
    "teal": "#077183",
    "turquoise": "#00C7E1",
    "lightblue": "#A2D9F5",
    "darkblue": "#00559E",
    "purple": "#6A27AE",
    "pink": "#FF51B8",
}

DEFAULT_ORDER = (
    "turquoise",
    "red",
    "darkblue",
    "yellow",
    "teal",
    "pink",
    "green",
    "purple",
    "orange",
    "lightblue",
)


def olink_pal(
    n: int | None = None,
    coloroption: Sequence[str] | None = None,
    alpha: float = 1.0,
) -> list[str]:
    """Return ``n`` Olink palette colors as RGBA hex strings."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha in wrong range")

    if coloroption is not None:
        missing = [name for name in coloroption if name not in OLINK_COLORS]
        if missing:
            raise ValueError(f"Unknown coloroption value(s): {missing}")
        colors = [OLINK_COLORS[name] for name in coloroption]
        if n is None:
            n = len(colors)
        if n <= len(colors):
            return [_with_alpha(c, alpha) for c in colors[:n]]
        return _ramp(colors, n, alpha)

    ordered = [OLINK_COLORS[name] for name in DEFAULT_ORDER]
    if n is None:
        n = len(ordered)
    if n <= len(ordered):
        return [_with_alpha(c, alpha) for c in ordered[:n]]
    spectrum = [OLINK_COLORS[name] for name in OLINK_COLORS]
    return _ramp(spectrum, n, alpha)


def _with_alpha(hex_color: str, alpha: float) -> str:
    rgb = mcolors.to_rgb(hex_color)
    return mcolors.to_hex((*rgb, alpha), keep_alpha=True)


def _ramp(colors: Sequence[str], n: int, alpha: float) -> list[str]:
    cmap = mcolors.LinearSegmentedColormap.from_list("olink", list(colors))
    return [_with_alpha(mcolors.to_hex(cmap(x)), alpha) for x in np.linspace(0, 1, n)]
