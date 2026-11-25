"""Flag NPX values below the assay LOD."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from olinkanalyze._utils import attach_check_log, col, extract_check_log, get_check_log, require_dataframe


def olink_lod(
    df: pd.DataFrame,
    check_log: Mapping | None = None,
) -> pd.DataFrame:
    """Add a ``BelowLOD`` column where NPX is strictly less than ``LOD``.

    Rows with missing NPX or LOD are not flagged. The attached check_log is
    preserved when present.
    """
    require_dataframe(df)
    check_log = get_check_log(df, check_log=check_log)
    if "lod" not in check_log["col_names"]:
        raise ValueError("LOD column is required for olink_lod().")
    quant_col = col(check_log, "quant")
    lod_col = col(check_log, "lod")
    out = df.copy()
    out["BelowLOD"] = out[quant_col].lt(out[lod_col]) & out[quant_col].notna() & out[lod_col].notna()
    attached = extract_check_log(df)
    if attached is not None:
        return attach_check_log(out, attached, copy=False)
    return attach_check_log(out, check_log, copy=False)


def olink_lod_summary(
    df: pd.DataFrame,
    check_log: Mapping | None = None,
) -> pd.DataFrame:
    """Per-assay fraction of measurements below LOD."""
    flagged = olink_lod(df, check_log=check_log)
    check_log = get_check_log(flagged, check_log=check_log)
    oid_col = col(check_log, "olink_id")
    assay_col = col(check_log, "assay")
    panel_col = col(check_log, "panel")
    grouped = flagged.groupby(oid_col, sort=False, dropna=False, observed=True)
    summary = grouped.agg(
        Assay=(assay_col, "first"),
        Panel=(panel_col, "first"),
        n=("BelowLOD", "size"),
        n_below=("BelowLOD", "sum"),
    )
    summary["OlinkID"] = summary.index
    summary["fraction_below"] = summary["n_below"] / summary["n"]
    return summary.reset_index(drop=True)[["Assay", "OlinkID", "Panel", "n", "n_below", "fraction_below"]]
