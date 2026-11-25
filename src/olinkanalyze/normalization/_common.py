"""Shared helpers for project-level NPX adjustments."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import pandas as pd

from olinkanalyze._utils import attach_check_log, col
from olinkanalyze.qc.check_npx import check_npx
from olinkanalyze.qc.column_names import categorize_npx_columns


def plain(df: pd.DataFrame) -> pd.DataFrame:
    """Copy a frame without attrs so pandas concat/merge can compare metadata."""
    out = df.copy()
    out.attrs = {}
    return out


def apply_project_adjustment(
    *,
    ref_df: pd.DataFrame,
    not_ref_df: pd.DataFrame,
    adj: pd.DataFrame,
    ref_log: Mapping,
    not_ref_log: Mapping,
    ref_name: str,
    not_ref_name: str,
    warn_bridges: bool = False,
) -> pd.DataFrame:
    oid_ref = col(ref_log, "olink_id")
    oid_not = col(not_ref_log, "olink_id")
    quant_not = col(not_ref_log, "quant")

    ref_out = plain(ref_df)
    ref_out["Project"] = ref_name
    ref_out["Adj_factor"] = 0.0

    not_ref_out = plain(not_ref_df)
    not_ref_out = not_ref_out.merge(
        adj.rename(columns={oid_ref: oid_not}),
        on=oid_not,
        how="left",
    )
    not_ref_out["Adj_factor"] = not_ref_out["Adj_factor"].fillna(0.0)
    not_ref_out[quant_not] = not_ref_out[quant_not] + not_ref_out["Adj_factor"]
    if "lod" in not_ref_log["col_names"] and col(not_ref_log, "lod") in not_ref_out.columns:
        lod_col = col(not_ref_log, "lod")
        not_ref_out[lod_col] = not_ref_out[lod_col] + not_ref_out["Adj_factor"]
    not_ref_out["Project"] = not_ref_name

    combined = pd.concat([ref_out, not_ref_out], ignore_index=True, sort=False)
    if warn_bridges:
        warnings.warn(
            "Output includes two sets of bridging samples. "
            "We recommend retaining only bridge samples from the reference dataset "
            "for downstream analysis.",
            UserWarning,
            stacklevel=3,
        )
    log = check_npx(combined)
    combined = categorize_npx_columns(combined, col_names=log["col_names"])
    return attach_check_log(combined, log, copy=False)
