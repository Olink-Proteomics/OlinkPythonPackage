"""Subset (paired-sample) normalization between NPX projects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from olinkanalyze._utils import col, get_check_log, require_dataframe
from olinkanalyze.normalization._common import apply_project_adjustment, plain


def olink_normalization_subset(
    project_1_df: pd.DataFrame,
    project_2_df: pd.DataFrame,
    overlapping_samples_df1: Sequence[str],
    overlapping_samples_df2: Sequence[str],
    project_1_name: str = "P1",
    project_2_name: str = "P2",
    reference_project: str = "P1",
    project_1_check_log: Mapping | None = None,
    project_2_check_log: Mapping | None = None,
) -> pd.DataFrame:
    """Normalize using paired subset samples (matched by position in the two lists)."""
    require_dataframe(project_1_df, name="df1")
    require_dataframe(project_2_df, name="df2")
    samples_1 = list(overlapping_samples_df1)
    samples_2 = list(overlapping_samples_df2)
    if len(samples_1) != len(samples_2):
        raise ValueError("overlapping_samples_df1 and overlapping_samples_df2 must have the same length.")
    if len(samples_1) == 0:
        raise ValueError("overlapping sample lists must not be empty.")

    log1 = get_check_log(project_1_df, check_log=project_1_check_log)
    log2 = get_check_log(project_2_df, check_log=project_2_check_log)
    if reference_project == project_1_name:
        ref_df, not_ref_df = project_1_df, project_2_df
        ref_log, not_ref_log = log1, log2
        ref_name, not_ref_name = project_1_name, project_2_name
        ref_samples, not_samples = samples_1, samples_2
    else:
        ref_df, not_ref_df = project_2_df, project_1_df
        ref_log, not_ref_log = log2, log1
        ref_name, not_ref_name = project_2_name, project_1_name
        ref_samples, not_samples = samples_2, samples_1

    adj = _subset_adjustment_factors(
        ref_df=ref_df,
        not_ref_df=not_ref_df,
        ref_log=ref_log,
        not_ref_log=not_ref_log,
        ref_samples=ref_samples,
        not_samples=not_samples,
        ref_name=ref_name,
        not_ref_name=not_ref_name,
    )
    return apply_project_adjustment(
        ref_df=ref_df,
        not_ref_df=not_ref_df,
        adj=adj,
        ref_log=ref_log,
        not_ref_log=not_ref_log,
        ref_name=ref_name,
        not_ref_name=not_ref_name,
    )


def _subset_adjustment_factors(
    *,
    ref_df: pd.DataFrame,
    not_ref_df: pd.DataFrame,
    ref_log: Mapping,
    not_ref_log: Mapping,
    ref_samples: Sequence[str],
    not_samples: Sequence[str],
    ref_name: str,
    not_ref_name: str,
) -> pd.DataFrame:
    sample_ref = col(ref_log, "sample_id")
    sample_not = col(not_ref_log, "sample_id")
    oid_ref = col(ref_log, "olink_id")
    oid_not = col(not_ref_log, "olink_id")
    quant_ref = col(ref_log, "quant")
    quant_not = col(not_ref_log, "quant")

    pair_map = pd.DataFrame({"SampleID_ref": list(ref_samples), "SampleID_not": list(not_samples)})
    ref_cols = [sample_ref, oid_ref, quant_ref]
    not_cols = [sample_not, oid_not, quant_not]
    ref_bridge = plain(ref_df.loc[ref_df[sample_ref].isin(ref_samples), ref_cols])
    not_bridge = plain(not_ref_df.loc[not_ref_df[sample_not].isin(not_samples), not_cols])
    ref_bridge = ref_bridge.rename(
        columns={sample_ref: "SampleID_ref", oid_ref: "OlinkID", quant_ref: ref_name}
    )
    not_bridge = not_bridge.rename(
        columns={sample_not: "SampleID_not", oid_not: "OlinkID", quant_not: not_ref_name}
    )
    ref_bridge = ref_bridge.merge(pair_map, on="SampleID_ref", how="inner")
    paired = ref_bridge.merge(not_bridge, on=["SampleID_not", "OlinkID"], how="inner")
    paired["quant_diff"] = paired[ref_name] - paired[not_ref_name]
    adj = paired.groupby("OlinkID", dropna=False, observed=True)["quant_diff"].median().rename("Adj_factor")
    adj = adj.reset_index()
    adj["Adj_factor"] = adj["Adj_factor"].fillna(0.0)
    return adj.rename(columns={"OlinkID": oid_ref})
