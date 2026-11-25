import pandas as pd
import pytest
from scipy import stats

from olinkanalyze import clean_npx, olink_anova, olink_anova_posthoc, olink_wilcox


def test_olink_wilcox_treatment_effect(data1: pd.DataFrame):
    cleaned = clean_npx(data1)
    result = olink_wilcox(cleaned, variable="Treatment")
    assert {"estimate", "p.value", "Adjusted_pval", "Threshold", "Treated", "Untreated"} <= set(
        result.columns
    )
    effect = result.loc[result["OlinkID"].isin(["OID20001", "OID20002", "OID20003"])]
    assert (effect["estimate"] > 0).all()
    assert result["method"].str.contains("rank-sum").all()


def test_olink_wilcox_matches_scipy(data1: pd.DataFrame):
    cleaned = clean_npx(data1)
    result = olink_wilcox(cleaned, variable="Treatment")
    oid = result["OlinkID"].iloc[0]
    subset = cleaned.loc[cleaned["OlinkID"] == oid]
    levels = sorted(subset["Treatment"].dropna().astype(str).unique())
    g1 = subset.loc[subset["Treatment"].astype(str) == levels[0], "NPX"].dropna().to_numpy(dtype=float)
    g2 = subset.loc[subset["Treatment"].astype(str) == levels[1], "NPX"].dropna().to_numpy(dtype=float)
    expected = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    row = result.loc[result["OlinkID"] == oid].iloc[0]
    assert row["statistic"] == pytest.approx(float(expected.statistic))
    assert row["p.value"] == pytest.approx(float(expected.pvalue))


def test_olink_wilcox_paired(data1: pd.DataFrame):
    cleaned = clean_npx(data1)
    paired = cleaned.loc[cleaned["Time"].isin(["Baseline", "Week.6"])].copy()
    result = olink_wilcox(paired, variable="Time", pair_id="Subject")
    assert result["method"].str.contains("signed-rank").all()


def test_olink_anova_site_has_results(data1: pd.DataFrame):
    cleaned = clean_npx(data1)
    result = olink_anova(cleaned, variable="Site")
    assert {"statistic", "p.value", "Adjusted_pval", "Threshold"} <= set(result.columns)
    assert result["method"].eq("One-way ANOVA").all()
    assert not result.empty


def test_olink_anova_posthoc_pairwise(data1: pd.DataFrame):
    cleaned = clean_npx(data1)
    posthoc = olink_anova_posthoc(cleaned, variable="Site")
    assert "contrast" in posthoc.columns
    assert posthoc["contrast"].nunique() >= 2
    assert {"p.value", "Adjusted_pval", "Threshold"} <= set(posthoc.columns)
