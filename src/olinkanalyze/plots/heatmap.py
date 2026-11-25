"""Sample × assay NPX heatmap."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from olinkanalyze._utils import col, require_dataframe
from olinkanalyze.plots.theme import apply_theme_to_figure
from olinkanalyze.qc.clean_npx import prepare_plot_npx


def olink_heatmap_plot(
    df: pd.DataFrame,
    check_log: Mapping | None = None,
    *,
    n_assays: int | None = 20,
    fontsize: int = 8,
) -> Figure:
    """Heatmap of samples versus assays.

    Assays are chosen by NPX variance when ``n_assays`` is set. Values are
    median-centered per assay.
    """
    require_dataframe(df)
    plot_df, check_log = prepare_plot_npx(df, check_log=check_log)
    sample_col = col(check_log, "sample_id")
    quant_col = col(check_log, "quant")
    oid_col = col(check_log, "olink_id")
    assay_col = col(check_log, "assay")

    wide = plot_df.pivot_table(
        index=sample_col,
        columns=oid_col,
        values=quant_col,
        aggfunc="mean",
        observed=True,
    )
    wide = wide.dropna(axis=1, how="all")
    if wide.empty:
        raise ValueError("No NPX values available for a heatmap.")
    if n_assays is not None and n_assays < wide.shape[1]:
        variance = wide.var(axis=0, skipna=True)
        keep = variance.nlargest(n_assays).index
        wide = wide.loc[:, keep]
    centered = wide.sub(wide.median(axis=0), axis=1)
    oid_to_assay = plot_df.groupby(oid_col, observed=True)[assay_col].first()
    centered.columns = [str(oid_to_assay.get(oid, oid)) for oid in centered.columns]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * centered.shape[1] + 2), 6.0))
    sns.heatmap(
        centered,
        ax=ax,
        cmap="vlag",
        center=0.0,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "NPX (median-centered)"},
    )
    ax.set_xlabel("Assay")
    ax.set_ylabel("SampleID")
    ax.tick_params(axis="x", labelsize=fontsize, labelrotation=90)
    ax.tick_params(axis="y", labelsize=fontsize)
    apply_theme_to_figure(fig)
    return fig
