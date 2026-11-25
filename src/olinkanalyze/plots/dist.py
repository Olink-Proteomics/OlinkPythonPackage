"""NPX distribution boxplots per sample and panel."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cbook import boxplot_stats
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from olinkanalyze._utils import col, require_dataframe, series_str_contains
from olinkanalyze.plots.colors import olink_pal
from olinkanalyze.plots.theme import apply_theme_to_figure, figure_legend_outside
from olinkanalyze.qc.clean_npx import prepare_plot_npx


def olink_dist_plot(
    df: pd.DataFrame,
    check_log: Mapping | None = None,
    color_g: str = "QC_Warning",
) -> Figure:
    """Boxplots of NPX versus SampleID, faceted by Panel.

    Box statistics are computed per sample, then drawn with matplotlib, so
    seaborn never sees every assay-level NPX row.
    """
    require_dataframe(df)
    plot_df, check_log = prepare_plot_npx(df, check_log=check_log)
    sample_col = col(check_log, "sample_id")
    quant_col = col(check_log, "quant")
    panel_col = col(check_log, "panel")
    if color_g not in plot_df.columns:
        raise ValueError(f"Column {color_g!r} is not present in df.")

    plot_df[panel_col] = plot_df[panel_col].astype(str).str.replace("Olink ", "", regex=False)
    if color_g == check_log["col_names"].get("qc_warning", color_g):
        plot_df = _summarize_qc_warning(plot_df, check_log, color_g)

    panels = list(plot_df[panel_col].dropna().unique())
    n_panels = max(len(panels), 1)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(5.5 * n_panels, 4.5),
        squeeze=False,
        layout="constrained",
    )
    hue_levels = [level for level in pd.unique(plot_df[color_g]) if pd.notna(level)]
    palette = olink_pal(n=max(len(hue_levels), 1))
    palette_map = {level: palette[idx] for idx, level in enumerate(hue_levels)}

    for ax, panel in zip(axes[0], panels, strict=False):
        subset = plot_df.loc[plot_df[panel_col] == panel, [sample_col, quant_col, color_g]]
        _draw_sample_boxes(
            ax,
            subset,
            sample_col=sample_col,
            quant_col=quant_col,
            color_g=color_g,
            palette_map=palette_map,
            default_color=palette[0],
        )
        ax.set_title(str(panel))
        ax.set_xlabel("Samples")
        ax.set_ylabel(quant_col)
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)

    if palette_map:
        handles = [
            Patch(facecolor=color, edgecolor="0.2", label=str(level))
            for level, color in palette_map.items()
        ]
        figure_legend_outside(fig, handles=handles, title=color_g)

    apply_theme_to_figure(fig)
    return fig


def _draw_sample_boxes(
    ax,
    subset: pd.DataFrame,
    *,
    sample_col: str,
    quant_col: str,
    color_g: str,
    palette_map: dict,
    default_color: str,
) -> None:
    grouped = subset.groupby(sample_col, sort=False, observed=True, dropna=False)
    sample_hue = grouped[color_g].first()
    arrays: list[np.ndarray] = []
    labels: list[object] = []
    medians: list[float] = []
    for sample, npx in grouped[quant_col]:
        values = npx.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        arrays.append(values)
        labels.append(sample)
        medians.append(float(np.median(values)))
    if not arrays:
        return

    order = np.argsort(medians, kind="mergesort")
    labels = [labels[idx] for idx in order]
    arrays = [arrays[idx] for idx in order]
    stats = boxplot_stats(arrays, labels=[str(label) for label in labels], whis=1.5)
    drawn = ax.bxp(
        stats,
        patch_artist=True,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 2,
            "markerfacecolor": "none",
            "markeredgecolor": "0.3",
        },
        medianprops={"color": "black", "linewidth": 1},
        whiskerprops={"color": "0.3", "linewidth": 0.8},
        capprops={"color": "0.3", "linewidth": 0.8},
        boxprops={"linewidth": 0.8},
    )
    for box, sample in zip(drawn["boxes"], labels, strict=True):
        hue = sample_hue.get(sample, sample_hue.iloc[0] if not sample_hue.empty else None)
        box.set_facecolor(palette_map.get(hue, default_color))
        box.set_edgecolor("0.2")


def _summarize_qc_warning(df: pd.DataFrame, check_log: Mapping, color_g: str) -> pd.DataFrame:
    sample_col = col(check_log, "sample_id")
    panel_col = col(check_log, "panel")
    fail = series_str_contains(df[color_g], "fail|warn", case=False)
    summary = (
        df.assign(_fail=fail)
        .groupby([sample_col, panel_col], dropna=False, sort=False)["_fail"]
        .any()
        .rename("_any_fail")
        .reset_index()
    )
    out = df.merge(summary, on=[sample_col, panel_col], how="left")
    out[color_g] = np.where(out["_any_fail"], "FAIL", "PASS")
    return out.drop(columns=["_any_fail"])
