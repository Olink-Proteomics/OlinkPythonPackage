"""Volcano plot of t-test results."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from olinkanalyze._utils import require_dataframe
from olinkanalyze.plots.colors import olink_pal
from olinkanalyze.plots.theme import apply_theme_to_figure


def olink_volcano_plot(
    p_val_tbl: pd.DataFrame,
    x_lab: str = "Estimate",
    olinkid_list: Sequence[str] | None = None,
) -> Figure:
    """Plot -log10(p-value) versus effect size from :func:`olink_ttest`."""
    require_dataframe(p_val_tbl, name="p.val_tbl")
    required = {"estimate", "p.value", "Threshold", "OlinkID", "Assay"}
    missing = required.difference(p_val_tbl.columns)
    if missing:
        raise ValueError(f"p.val_tbl is missing columns: {sorted(missing)}")

    plot_df = p_val_tbl.copy()
    plot_df["neg_log10_p"] = -np.log10(plot_df["p.value"].clip(lower=np.finfo(float).tiny))
    if olinkid_list is None:
        labels = plot_df.loc[plot_df["Threshold"] == "Significant", "OlinkID"]
    else:
        labels = pd.Series(list(olinkid_list))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    palette = {
        "Significant": olink_pal(coloroption=["red"])[0],
        "Non-significant": olink_pal(coloroption=["turquoise"])[0],
    }
    sns.scatterplot(
        data=plot_df,
        x="estimate",
        y="neg_log10_p",
        hue="Threshold",
        palette=palette,
        ax=ax,
        s=28,
        edgecolor="none",
    )
    ax.axhline(-np.log10(0.05), linestyle=":", color="grey", linewidth=1)
    ax.set_xlabel(x_lab)
    ax.set_ylabel("-log10(p-value)")

    labeled = plot_df.loc[plot_df["OlinkID"].isin(set(labels.tolist()))]
    for _, row in labeled.iterrows():
        ax.annotate(
            str(row["Assay"]),
            (row["estimate"], row["neg_log10_p"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    apply_theme_to_figure(fig)
    return fig
