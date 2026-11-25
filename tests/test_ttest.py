import pandas as pd
import pytest
from scipy import stats

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


def test_olink_ttest_matches_scipy_per_assay(data1: pd.DataFrame):
    cleaned = clean_npx(data1)
    result = olink_ttest(cleaned, variable="Treatment")
    oid = result["OlinkID"].iloc[0]
    subset = cleaned.loc[cleaned["OlinkID"] == oid]
    levels = sorted(subset["Treatment"].dropna().astype(str).unique())
    g1 = subset.loc[subset["Treatment"].astype(str) == levels[0], "NPX"].dropna().to_numpy(dtype=float)
    g2 = subset.loc[subset["Treatment"].astype(str) == levels[1], "NPX"].dropna().to_numpy(dtype=float)
    expected = stats.ttest_ind(g1, g2, equal_var=False)
    ci = expected.confidence_interval(confidence_level=0.95)
    row = result.loc[result["OlinkID"] == oid].iloc[0]
    assert row["statistic"] == pytest.approx(float(expected.statistic))
    assert row["p.value"] == pytest.approx(float(expected.pvalue))
    assert row["parameter"] == pytest.approx(float(expected.df))
    assert row["conf.low"] == pytest.approx(float(ci.low))
    assert row["conf.high"] == pytest.approx(float(ci.high))
    assert row["estimate"] == pytest.approx(float(g1.mean() - g2.mean()))


def test_olink_ttest_paired(data1: pd.DataFrame):
    cleaned = clean_npx(data1)
    paired = cleaned.loc[cleaned["Time"].isin(["Baseline", "Week.6"])].copy()
    result = olink_ttest(paired, variable="Time", pair_id="Subject")
    assert "method" in result.columns
    assert result["method"].str.contains("Paired").all()
