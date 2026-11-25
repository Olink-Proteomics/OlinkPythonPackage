import pandas as pd

from olinkanalyze import olink_normalization, olink_normalization_intensity, olink_normalization_subset
from olinkanalyze._constants import BRIDGE_SAMPLE_IDS


def test_bridge_normalization_shifts_non_reference(data1: pd.DataFrame, data2: pd.DataFrame):
    bridge = [
        sample
        for sample in BRIDGE_SAMPLE_IDS
        if sample in set(data1["SampleID"]) and sample in set(data2["SampleID"])
    ]
    assert bridge
    normalized = olink_normalization(
        df1=data1,
        df2=data2,
        overlapping_samples_df1=bridge,
        df1_project_nr="P1",
        df2_project_nr="P2",
        reference_project="P1",
    )
    assert {"Project", "Adj_factor"} <= set(normalized.columns)
    ref = normalized.loc[normalized["Project"] == "P1"]
    other = normalized.loc[normalized["Project"] == "P2"]
    assert (ref["Adj_factor"] == 0).all()
    assert not (other["Adj_factor"] == 0).all()

    oid = "OID20006"
    ref_mean = data1.loc[data1["OlinkID"] == oid, "NPX"].mean()
    other_before = data2.loc[data2["OlinkID"] == oid, "NPX"].mean()
    other_after = other.loc[other["OlinkID"] == oid, "NPX"].mean()
    # data2 is generated with a +0.8 project shift; bridging should pull it back.
    assert other_before > ref_mean
    assert abs(other_after - ref_mean) < abs(other_before - ref_mean)


def test_intensity_normalization_aligns_medians(data1: pd.DataFrame, data2: pd.DataFrame):
    normalized = olink_normalization_intensity(
        project_1_df=data1,
        project_2_df=data2,
        project_1_name="P1",
        project_2_name="P2",
        reference_project="P1",
    )
    oid = "OID20006"
    ref_med = data1.loc[data1["OlinkID"] == oid, "NPX"].median()
    other_before = data2.loc[data2["OlinkID"] == oid, "NPX"].median()
    other_after = normalized.loc[
        (normalized["Project"] == "P2") & (normalized["OlinkID"] == oid), "NPX"
    ].median()
    assert other_before > ref_med
    assert abs(other_after - ref_med) < abs(other_before - ref_med)


def test_subset_normalization_pairs_samples(data1: pd.DataFrame, data2: pd.DataFrame):
    bridge = [
        sample
        for sample in BRIDGE_SAMPLE_IDS
        if sample in set(data1["SampleID"]) and sample in set(data2["SampleID"])
    ]
    normalized = olink_normalization_subset(
        project_1_df=data1,
        project_2_df=data2,
        overlapping_samples_df1=bridge,
        overlapping_samples_df2=bridge,
        project_1_name="P1",
        project_2_name="P2",
        reference_project="P1",
    )
    assert {"Project", "Adj_factor"} <= set(normalized.columns)
    assert not normalized.loc[normalized["Project"] == "P2", "Adj_factor"].eq(0).all()


def test_olink_normalization_dispatches_intensity(data1: pd.DataFrame, data2: pd.DataFrame):
    normalized = olink_normalization(df1=data1, df2=data2, df1_project_nr="P1", df2_project_nr="P2")
    assert "Adj_factor" in normalized.columns
