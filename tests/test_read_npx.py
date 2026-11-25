from pathlib import Path

import pandas as pd
import pytest

from olinkanalyze import read_NPX, read_npx
from olinkanalyze._utils import extract_check_log
from olinkanalyze.data import load_example_csv


def test_read_npx_long_csv():
    path = load_example_csv()
    df = read_npx(path)
    assert not df.empty
    assert "SampleID" in df.columns
    assert "OlinkID" in df.columns
    assert "NPX" in df.columns
    log = extract_check_log(df)
    assert log is not None
    assert log["col_names"]["quant"] == "NPX"
    assert log["col_names"]["sample_id"] == "SampleID"


def test_read_npx_alias():
    df = read_NPX(load_example_csv())
    assert isinstance(df, pd.DataFrame)


def test_read_npx_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_npx(tmp_path / "missing.csv")


def test_read_npx_unsupported_extension(tmp_path: Path):
    path = tmp_path / "data.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported NPX file extension"):
        read_npx(path)


def test_read_npx_excel_and_parquet_roundtrip(data1: pd.DataFrame, tmp_path: Path):
    frame = data1.drop(columns=[], errors="ignore").head(40).copy()
    frame.attrs.clear()
    xlsx = tmp_path / "npx.xlsx"
    parquet = tmp_path / "npx.parquet"
    frame.to_excel(xlsx, index=False)
    frame.to_parquet(parquet, index=False)

    from_excel = read_npx(xlsx)
    from_parquet = read_npx(parquet)
    assert len(from_excel) == len(frame)
    assert len(from_parquet) == len(frame)
    assert extract_check_log(from_excel) is not None
    assert extract_check_log(from_parquet) is not None
