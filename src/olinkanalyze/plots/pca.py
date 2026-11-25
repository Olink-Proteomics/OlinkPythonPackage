"""PCA scatter plot of samples from long-format NPX data."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from olinkanalyze._utils import col, require_dataframe
from olinkanalyze.plots.colors import olink_pal
from olinkanalyze.plots.theme import apply_theme_to_figure, figure_legend_outside
from olinkanalyze.qc.clean_npx import prepare_plot_npx


def olink_pca_plot(
    df: pd.DataFrame,
    check_log: Mapping | None = None,
    color_g: str | None = "QC_Warning",
    *,
    n_loadings: int = 0,
) -> Figure:
    """Plot PC1 versus PC2 for samples. NPX is pivoted to samples × assays.

    Missing values are filled with the assay median after dropping assays that
    are entirely missing. Optional ``n_loadings`` labels the assays with the
    largest loadings on PC1.
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
    if wide.shape[0] < 3 or wide.shape[1] < 2:
        raise ValueError("PCA needs at least 3 samples and 2 assays with data.")
    filled = wide.fillna(wide.median(axis=0))
    matrix = filled.to_numpy(dtype=float)
    matrix = matrix - np.nanmean(matrix, axis=0)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    scores = u[:, :2] * s[:2]
    var = s**2
    pct = var / var.sum() * 100

    color_values = None
    if color_g is not None:
        if color_g not in plot_df.columns:
            raise ValueError(f"Column {color_g!r} is not present in df.")
        color_values = plot_df.groupby(sample_col, sort=False, observed=True)[color_g].first()
        color_values = color_values.reindex(filled.index)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    if color_values is None:
        ax.scatter(scores[:, 0], scores[:, 1], s=36, color=olink_pal(n=1)[0], edgecolor="none")
    else:
        levels = [level for level in pd.unique(color_values) if pd.notna(level)]
        palette = olink_pal(n=max(len(levels), 1))
        palette_map = {level: palette[idx] for idx, level in enumerate(levels)}
        for level in levels:
            mask = color_values.astype(object).eq(level).to_numpy()
            ax.scatter(
                scores[mask, 0],
                scores[mask, 1],
                s=36,
                color=palette_map[level],
                label=str(level),
                edgecolor="none",
            )
        handles, labels = ax.get_legend_handles_labels()
        figure_legend_outside(fig, handles, labels, title=color_g)

    if n_loadings > 0:
        loading = pd.Series(np.abs(vt[0]), index=filled.columns).sort_values(ascending=False)
        oid_to_assay = plot_df.groupby(oid_col, observed=True)[assay_col].first()
        scale = np.max(np.abs(scores[:, :2])) / max(np.max(np.abs(vt[0, : n_loadings + 1])), 1e-8)
        for oid in loading.head(n_loadings).index:
            col_idx = list(filled.columns).index(oid)
            ax.annotate(
                str(oid_to_assay.get(oid, oid)),
                (vt[0, col_idx] * scale * 0.3, vt[1, col_idx] * scale * 0.3),
                fontsize=8,
            )

    ax.set_xlabel(f"PC1 ({pct[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({pct[1]:.1f}%)")
    ax.set_title("PCA")
    apply_theme_to_figure(fig)
    fig._olink_pca_scores = pd.DataFrame(  # type: ignore[attr-defined]
        scores, index=filled.index, columns=["PC1", "PC2"]
    )
    return fig
