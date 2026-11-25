"""Intensity (median) normalization between NPX projects."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from olinkanalyze._utils import col, get_check_log, require_dataframe
from olinkanalyze.normalization._common import apply_project_adjustment, plain


def olink_normalization_intensity(
    project_1_df: pd.DataFrame,
    project_2_df: pd.DataFrame,
    project_1_name: str = "P1",
    project_2_name: str = "P2",
    reference_project: str = "P1",
    project_1_check_log: Mapping | None = None,
    project_2_check_log: Mapping | None = None,
) -> pd.DataFrame:
    """Shift the non-reference project so per-assay medians match the reference."""
    require_dataframe(project_1_df, name="df1")
    require_dataframe(project_2_df, name="df2")
    log1 = get_check_log(project_1_df, check_log=project_1_check_log)
    log2 = get_check_log(project_2_df, check_log=project_2_check_log)
    if reference_project == project_1_name:
        ref_df, not_ref_df = project_1_df, project_2_df
        ref_log, not_ref_log = log1, log2
        ref_name, not_ref_name = project_1_name, project_2_name
    else:
        ref_df, not_ref_df = project_2_df, project_1_df
        ref_log, not_ref_log = log2, log1
        ref_name, not_ref_name = project_2_name, project_1_name

    adj = _intensity_adjustment_factors(ref_df, not_ref_df, ref_log, not_ref_log)
    return apply_project_adjustment(
        ref_df=ref_df,
        not_ref_df=not_ref_df,
        adj=adj,
        ref_log=ref_log,
        not_ref_log=not_ref_log,
        ref_name=ref_name,
        not_ref_name=not_ref_name,
    )


def _intensity_adjustment_factors(
    ref_df: pd.DataFrame,
    not_ref_df: pd.DataFrame,
    ref_log: Mapping,
    not_ref_log: Mapping,
) -> pd.DataFrame:
    oid_ref = col(ref_log, "olink_id")
    oid_not = col(not_ref_log, "olink_id")
    quant_ref = col(ref_log, "quant")
    quant_not = col(not_ref_log, "quant")
    ref_med = plain(ref_df).groupby(oid_ref, dropna=False, observed=True)[quant_ref].median()
    not_med = plain(not_ref_df).groupby(oid_not, dropna=False, observed=True)[quant_not].median()
    adj = (ref_med - not_med.reindex(ref_med.index)).rename("Adj_factor").reset_index()
    adj["Adj_factor"] = adj["Adj_factor"].fillna(0.0)
    return adj
