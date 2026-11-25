import pandas as pd
import pytest

from olinkanalyze import check_npx, clean_npx, extract_check_log


def test_check_npx_detects_invalid_oid(data1: pd.DataFrame):
    df = data1.copy()
    df["OlinkID"] = df["OlinkID"].astype(str)
    df.loc[df.index[0], "OlinkID"] = "NOT_AN_OID"
    with pytest.warns(UserWarning, match="Unrecognized OlinkID"):
        log = check_npx(df)
    assert "NOT_AN_OID" in log["oid_invalid"]


def test_check_npx_detects_all_na_assay(data1: pd.DataFrame):
    df = data1.copy()
    oid = df["OlinkID"].iloc[0]
    df.loc[df["OlinkID"] == oid, "NPX"] = pd.NA
    with pytest.warns(UserWarning, match="NA for all samples"):
        log = check_npx(df)
    assert oid in log["assay_na"]


@pytest.mark.filterwarnings("default::UserWarning")
def test_check_npx_detects_duplicate_sample_ids(data1: pd.DataFrame):
    with pytest.warns(UserWarning, match="Duplicate SampleID"):
        log = check_npx(data1)
    assert any("CONTROL" in sample for sample in log["sample_id_dups"])


def test_clean_npx_removes_invalid_oid_and_duplicates(data1: pd.DataFrame):
    df = data1.copy()
    df["OlinkID"] = df["OlinkID"].astype(str)
    df.loc[df.index[0], "OlinkID"] = "BADID"
    cleaned = clean_npx(df)
    assert "BADID" not in set(cleaned["OlinkID"])
    assert not any("CONTROL" in str(s) for s in cleaned["SampleID"].unique())
    log = extract_check_log(cleaned)
    assert log is not None
    assert log["oid_invalid"] == []
    assert log["sample_id_dups"] == []


def test_clean_npx_can_keep_duplicates(data1: pd.DataFrame):
    cleaned = clean_npx(
        data1,
        remove_dup_sample_id=False,
        remove_control_sample=False,
        remove_qc_warning=False,
    )
    assert any("CONTROL" in str(s) for s in cleaned["SampleID"].unique())


def test_clean_npx_removes_fail_rows(data1: pd.DataFrame):
    cleaned = clean_npx(data1, remove_control_sample=False, remove_dup_sample_id=False)
    assert not cleaned["QC_Warning"].astype(str).str.contains("fail", case=False).any()


def test_clean_npx_skips_recheck_when_already_clean(data1: pd.DataFrame):
    cleaned = clean_npx(data1)
    again = clean_npx(cleaned)
    assert again is cleaned
    log = extract_check_log(again)
    assert log is not None
    assert log["oid_invalid"] == []
    assert log["sample_id_dups"] == []


def test_preferred_names_quant_column(data1: pd.DataFrame):
    df = data1.copy()
    df["PCNormalizedNPX"] = df["NPX"] + 1
    log = check_npx(df, preferred_names={"quant": "PCNormalizedNPX"})
    assert log["col_names"]["quant"] == "PCNormalizedNPX"
