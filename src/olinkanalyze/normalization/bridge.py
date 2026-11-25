"""Bridge (reference-sample) normalization between two NPX datasets."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import pandas as pd

from olinkanalyze._utils import attach_check_log, col, get_check_log, require_dataframe
from olinkanalyze.qc.check_npx import check_npx


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

    This Python slice implements **bridge normalization** only: the
    non-reference project is shifted by the per-assay median of pair-wise
    NPX differences on overlapping samples.
    """
    if overlapping_samples_df2 is not None or reference_medians is not None:
        raise NotImplementedError(
            "Subset, intensity, reference-median, and cross-product "
            "normalization are not implemented in this Python slice. "
            "Provide overlapping_samples_df1 only for bridge normalization."
        )
    if df2 is None:
        raise ValueError("df2 is required for bridge normalization.")
    if overlapping_samples_df1 is None:
        raise ValueError("overlapping_samples_df1 is required.")
    if reference_project not in {df1_project_nr, df2_project_nr}:
        raise ValueError("reference_project must be one of df1_project_nr and df2_project_nr.")
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

    bridge_samples = list(overlapping_samples_df1)
    adj = _bridge_adjustment_factors(
        ref_df=ref_df,
        not_ref_df=not_ref_df,
        ref_log=ref_log,
        not_ref_log=not_ref_log,
        bridge_samples=bridge_samples,
        ref_name=ref_name,
        not_ref_name=not_ref_name,
    )

    ref_out = _plain(ref_df)
    ref_out["Project"] = ref_name
    ref_out["Adj_factor"] = 0.0

    not_ref_out = _plain(not_ref_df)
    oid_ref = col(ref_log, "olink_id")
    oid_not = col(not_ref_log, "olink_id")
    quant_not = col(not_ref_log, "quant")
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
    warnings.warn(
        "Output includes two sets of bridging samples. "
        "We recommend retaining only bridge samples from the reference dataset "
        "for downstream analysis.",
        UserWarning,
        stacklevel=2,
    )
    return attach_check_log(combined, check_npx(combined))


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

    ref_bridge = _plain(ref_df.loc[ref_df[sample_ref].isin(bridge_samples), [sample_ref, oid_ref, quant_ref]])
    not_bridge = _plain(
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


def _plain(df: pd.DataFrame) -> pd.DataFrame:
    """Copy a frame without attrs so pandas concat/merge can compare metadata."""
    out = df.copy()
    out.attrs = {}
    return out
