import pandas as pd
import pytest

from olinkanalyze import clean_npx, olink_ttest


def test_olink_ttest_treatment_effect(data1: pd.DataFrame):
    cleaned = clean_npx(data1)
    result = olink_ttest(cleaned, variable="Treatment")
    assert {"estimate", "p.value", "Adjusted_pval", "Threshold", "Treated", "Untreated"} <= set(
        result.columns
    )
    assert result["Threshold"].isin(["Significant", "Non-significant"]).all()
    effect = result.loc[result["OlinkID"].isin(["OID20001", "OID20002", "OID20003"])]
    assert (effect["estimate"] > 0).all()
    assert (effect["Treated"] > effect["Untreated"]).all()


def test_olink_ttest_requires_two_levels(data1: pd.DataFrame):
    cleaned = clean_npx(data1)
    with pytest.raises(ValueError, match="number of levels"):
        olink_ttest(cleaned, variable="Site")


def test_olink_ttest_paired(data1: pd.DataFrame):
    cleaned = clean_npx(data1)
    paired = cleaned.loc[cleaned["Time"].isin(["Baseline", "Week.6"])].copy()
    result = olink_ttest(paired, variable="Time", pair_id="Subject")
    assert "method" in result.columns
    assert result["method"].str.contains("Paired").all()
