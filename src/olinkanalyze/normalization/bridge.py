"""Bridge (reference-sample) normalization between two NPX datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from olinkanalyze._utils import col, get_check_log, require_dataframe
from olinkanalyze.normalization._common import apply_project_adjustment, plain


def olink_normalization(
    df1: pd.DataFrame,
    df2: pd.DataFrame | None = None,
    overlapping_samples_df1: Sequence[str] | None = None,
    overlapping_samples_df2: Sequence[str] | None = None,
    df1_project_nr: str = "P1",
    df2_project_nr: str = "P2",
    reference_project: str = "P1",
    reference_medians: pd.DataFrame | None = None,
    df1_check_log: Mapping | None = None,
    df2_check_log: Mapping | None = None,
) -> pd.DataFrame:
    """Normalize two Olink datasets.

    Dispatches to bridge, subset, or intensity normalization. Cross-product
    and reference-median tables are not implemented.
    """
    if reference_medians is not None:
        raise NotImplementedError(
            "Reference-median and cross-product normalization are not implemented."
        )
    if df2 is None:
        raise ValueError("df2 is required for normalization.")
    if reference_project not in {df1_project_nr, df2_project_nr}:
        raise ValueError("reference_project must be one of df1_project_nr and df2_project_nr.")
    if overlapping_samples_df1 is not None and overlapping_samples_df2 is not None:
        from olinkanalyze.normalization.subset import olink_normalization_subset

        return olink_normalization_subset(
            project_1_df=df1,
            project_2_df=df2,
            overlapping_samples_df1=overlapping_samples_df1,
            overlapping_samples_df2=overlapping_samples_df2,
            project_1_name=df1_project_nr,
            project_2_name=df2_project_nr,
            reference_project=reference_project,
            project_1_check_log=df1_check_log,
            project_2_check_log=df2_check_log,
        )
    if overlapping_samples_df1 is not None:
        return olink_normalization_bridge(
            project_1_df=df1,
            project_2_df=df2,
            overlapping_samples_df1=overlapping_samples_df1,
            project_1_name=df1_project_nr,
            project_2_name=df2_project_nr,
            reference_project=reference_project,
            project_1_check_log=df1_check_log,
            project_2_check_log=df2_check_log,
        )
    if overlapping_samples_df2 is not None:
        raise ValueError("overlapping_samples_df2 requires overlapping_samples_df1 for subset normalization.")
    from olinkanalyze.normalization.intensity import olink_normalization_intensity

    return olink_normalization_intensity(
        project_1_df=df1,
        project_2_df=df2,
        project_1_name=df1_project_nr,
        project_2_name=df2_project_nr,
        reference_project=reference_project,
        project_1_check_log=df1_check_log,
        project_2_check_log=df2_check_log,
    )


def olink_normalization_bridge(
    project_1_df: pd.DataFrame,
    project_2_df: pd.DataFrame,
    overlapping_samples_df1: Sequence[str],
    project_1_name: str = "P1",
    project_2_name: str = "P2",
    reference_project: str = "P1",
    project_1_check_log: Mapping | None = None,
    project_2_check_log: Mapping | None = None,
) -> pd.DataFrame:
    """Bridge-normalize ``project_2_df`` toward ``project_1_df`` or vice versa."""
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

    adj = _bridge_adjustment_factors(
        ref_df=ref_df,
        not_ref_df=not_ref_df,
        ref_log=ref_log,
        not_ref_log=not_ref_log,
        bridge_samples=list(overlapping_samples_df1),
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
        warn_bridges=True,
    )


def _bridge_adjustment_factors(
    *,
    ref_df: pd.DataFrame,
    not_ref_df: pd.DataFrame,
    ref_log: Mapping,
    not_ref_log: Mapping,
    bridge_samples: Sequence[str],
    ref_name: str,
    not_ref_name: str,
) -> pd.DataFrame:
    sample_ref = col(ref_log, "sample_id")
    sample_not = col(not_ref_log, "sample_id")
    oid_ref = col(ref_log, "olink_id")
    oid_not = col(not_ref_log, "olink_id")
    quant_ref = col(ref_log, "quant")
    quant_not = col(not_ref_log, "quant")

    ref_bridge = plain(ref_df.loc[ref_df[sample_ref].isin(bridge_samples), [sample_ref, oid_ref, quant_ref]])
    not_bridge = plain(
        not_ref_df.loc[not_ref_df[sample_not].isin(bridge_samples), [sample_not, oid_not, quant_not]]
    )
    ref_bridge = ref_bridge.rename(columns={sample_ref: "SampleID", oid_ref: "OlinkID", quant_ref: ref_name})
    not_bridge = not_bridge.rename(
        columns={sample_not: "SampleID", oid_not: "OlinkID", quant_not: not_ref_name}
    )

    paired = ref_bridge.merge(not_bridge, on=["SampleID", "OlinkID"], how="inner")
    paired["quant_diff"] = paired[ref_name] - paired[not_ref_name]
    adj = paired.groupby("OlinkID", dropna=False)["quant_diff"].median().rename("Adj_factor").reset_index()
    adj["Adj_factor"] = adj["Adj_factor"].fillna(0.0)
    return adj.rename(columns={"OlinkID": oid_ref})
