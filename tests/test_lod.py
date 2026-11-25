import pandas as pd

from olinkanalyze import clean_npx, olink_lod, olink_lod_summary


def test_olink_lod_flags_values_below_threshold(data1: pd.DataFrame):
    cleaned = clean_npx(data1)
    df = cleaned.copy()
    df.loc[df.index[0], "NPX"] = df.loc[df.index[0], "LOD"] - 1.0
    flagged = olink_lod(df)
    assert flagged["BelowLOD"].sum() >= 1
    assert flagged.loc[df.index[0], "BelowLOD"]


def test_olink_lod_summary_has_fractions(data1: pd.DataFrame):
    summary = olink_lod_summary(clean_npx(data1))
    assert {"Assay", "OlinkID", "n", "n_below", "fraction_below"} <= set(summary.columns)
    assert (summary["fraction_below"] >= 0).all()
    assert (summary["fraction_below"] <= 1).all()
