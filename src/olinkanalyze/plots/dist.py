"""NPX distribution boxplots per sample and panel."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from olinkanalyze._utils import col, get_check_log, require_dataframe
from olinkanalyze.plots.colors import olink_pal
from olinkanalyze.plots.theme import apply_theme_to_figure
from olinkanalyze.qc.clean_npx import clean_npx


def olink_dist_plot(
    df: pd.DataFrame,
    check_log: Mapping | None = None,
    color_g: str = "QC_Warning",
) -> Figure:
    """Boxplots of NPX versus SampleID, faceted by Panel."""
    require_dataframe(df)
    check_log = get_check_log(df, check_log=check_log)
    cleaned = clean_npx(
        df,
        check_log=check_log,
        remove_assay_na=True,
        remove_invalid_oid=True,
        remove_dup_sample_id=False,
        remove_control_assay=False,
        remove_control_sample=False,
        remove_qc_warning=False,
        remove_assay_warning=False,
        convert_nonunique_uniprot=True,
        verbose=False,
    )
    check_log = get_check_log(cleaned)
    sample_col = col(check_log, "sample_id")
    quant_col = col(check_log, "quant")
    panel_col = col(check_log, "panel")
    if color_g not in cleaned.columns:
        raise ValueError(f"Column {color_g!r} is not present in df.")

    plot_df = cleaned.copy()
    plot_df[panel_col] = plot_df[panel_col].astype(str).str.replace("Olink ", "", regex=False)
    if color_g == check_log["col_names"].get("qc_warning", color_g):
        plot_df = _summarize_qc_warning(plot_df, check_log, color_g)

    panels = list(plot_df[panel_col].dropna().unique())
    n_panels = max(len(panels), 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5), squeeze=False)
    palette = olink_pal(n=max(plot_df[color_g].nunique(), 1))

    for ax, panel in zip(axes[0], panels, strict=False):
        subset = plot_df.loc[plot_df[panel_col] == panel]
        order = subset.groupby(sample_col)[quant_col].median().sort_values().index.tolist()
        sns.boxplot(
            data=subset,
            x=sample_col,
            y=quant_col,
            hue=color_g,
            order=order,
            palette=palette,
            ax=ax,
            fliersize=2,
        )
        ax.set_title(str(panel))
        ax.set_xlabel("Samples")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        if ax.legend_ is not None and panel != panels[0]:
            ax.legend_.remove()

    apply_theme_to_figure(fig)
    return fig


def _summarize_qc_warning(df: pd.DataFrame, check_log: Mapping, color_g: str) -> pd.DataFrame:
    sample_col = col(check_log, "sample_id")
    panel_col = col(check_log, "panel")
    fail = df[color_g].astype(str).str.contains("fail|warn", case=False, na=False)
    summary = (
        df.assign(_fail=fail)
        .groupby([sample_col, panel_col], dropna=False)["_fail"]
        .any()
        .rename("_any_fail")
        .reset_index()
    )
    out = df.merge(summary, on=[sample_col, panel_col], how="left")
    out[color_g] = out["_any_fail"].map(lambda x: "FAIL" if x else "PASS")
    return out.drop(columns=["_any_fail"])
