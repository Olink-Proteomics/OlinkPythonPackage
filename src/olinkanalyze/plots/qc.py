"""Sample-level IQR versus median QC scatter plots."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from olinkanalyze._utils import col, require_dataframe
from olinkanalyze.plots.colors import olink_pal
from olinkanalyze.plots.theme import apply_theme_to_figure
from olinkanalyze.qc.clean_npx import prepare_plot_npx


def olink_qc_plot(
    df: pd.DataFrame,
    check_log: Mapping | None = None,
    color_g: str = "QC_Warning",
    *,
    label_outliers: bool = False,
    iqr_outlier_def: float = 3.0,
    median_outlier_def: float = 3.0,
    outlier_lines: bool = True,
    facet_nrow: int | None = None,
    facet_ncol: int | None = None,
) -> Figure:
    """Scatterplot of sample IQR versus median NPX, faceted by Panel."""
    require_dataframe(df)
    cleaned, check_log = prepare_plot_npx(df, check_log=check_log)
    sample_col = col(check_log, "sample_id")
    quant_col = col(check_log, "quant")
    panel_col = col(check_log, "panel")
    if color_g not in cleaned.columns:
        raise ValueError(f"Column {color_g!r} is not present in df.")

    plot_df = cleaned.copy()
    plot_df[panel_col] = plot_df[panel_col].astype(str).str.replace("Olink ", "", regex=False)
    grouped = plot_df.groupby([panel_col, sample_col], dropna=False, sort=False)
    summary = grouped.agg(
        sample_median=(quant_col, "median"),
        **{color_g: (color_g, "first")},
    )
    summary["IQR"] = grouped[quant_col].quantile(0.75) - grouped[quant_col].quantile(0.25)
    summary = summary.reset_index()

    def _limits(frame: pd.DataFrame) -> pd.DataFrame:
        med_mean = frame["sample_median"].mean()
        med_sd = frame["sample_median"].std(ddof=1)
        iqr_mean = frame["IQR"].mean()
        iqr_sd = frame["IQR"].std(ddof=1)
        med_sd = 0.0 if pd.isna(med_sd) else med_sd
        iqr_sd = 0.0 if pd.isna(iqr_sd) else iqr_sd
        frame = frame.copy()
        frame["median_low"] = med_mean - median_outlier_def * med_sd
        frame["median_high"] = med_mean + median_outlier_def * med_sd
        frame["iqr_low"] = iqr_mean - iqr_outlier_def * iqr_sd
        frame["iqr_high"] = iqr_mean + iqr_outlier_def * iqr_sd
        in_range = (
            (frame["sample_median"] < frame["median_high"])
            & (frame["sample_median"] > frame["median_low"])
            & (frame["IQR"] > frame["iqr_low"])
            & (frame["IQR"] < frame["iqr_high"])
        )
        frame["Outlier"] = np.where(in_range, 0, 1)
        return frame

    summary = pd.concat(
        [_limits(group) for _, group in summary.groupby(panel_col, dropna=False, sort=False)],
        ignore_index=True,
    )
    panels = list(summary[panel_col].dropna().unique())
    n_panels = max(len(panels), 1)
    ncols = facet_ncol or n_panels
    nrows = facet_nrow or int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows), squeeze=False)
    palette = olink_pal(n=max(summary[color_g].nunique(), 1))

    for idx, panel in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        subset = summary.loc[summary[panel_col] == panel]
        sns.scatterplot(
            data=subset,
            x="sample_median",
            y="IQR",
            hue=color_g,
            palette=palette,
            ax=ax,
            s=40,
        )
        if outlier_lines and not subset.empty:
            row = subset.iloc[0]
            ax.axhline(row["iqr_low"], linestyle="--", color="grey", linewidth=0.8)
            ax.axhline(row["iqr_high"], linestyle="--", color="grey", linewidth=0.8)
            ax.axvline(row["median_low"], linestyle="--", color="grey", linewidth=0.8)
            ax.axvline(row["median_high"], linestyle="--", color="grey", linewidth=0.8)
        if label_outliers:
            outliers = subset.loc[subset["Outlier"] == 1]
            for _, point in outliers.iterrows():
                ax.annotate(
                    str(point[sample_col]),
                    (point["sample_median"], point["IQR"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )
        ax.set_title(str(panel))
        ax.set_xlabel("Sample Median")
        if ax.legend_ is not None and idx != 0:
            ax.legend_.remove()

    for extra in range(len(panels), nrows * ncols):
        axes[extra // ncols][extra % ncols].set_visible(False)

    apply_theme_to_figure(fig)
    fig._olink_qc_data = summary  # type: ignore[attr-defined]
    return fig
