"""Read long-format Olink NPX files."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from olinkanalyze._constants import ACCEPTED_FIELD_SEP, ACCEPTED_NPX_FILE_EXT
from olinkanalyze._utils import attach_check_log
from olinkanalyze.qc.check_npx import check_npx


def read_npx(
    filename: str | Path,
    preferred_names: Mapping[str, str] | None = None,
    *,
    attach_log: bool = True,
) -> pd.DataFrame:
    """Import a long-format Olink NPX file (CSV/TSV, Excel, or Parquet).

    Wide qPCR layouts, ZIP archives, and the R legacy reader are not
    supported in this Python slice.
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"NPX file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in ACCEPTED_NPX_FILE_EXT["excel"]:
        df = pd.read_excel(path)
    elif suffix in ACCEPTED_NPX_FILE_EXT["parquet"]:
        df = pd.read_parquet(path)
    elif suffix in ACCEPTED_NPX_FILE_EXT["delim"]:
        df = _read_delimited(path)
    else:
        supported = (
            ACCEPTED_NPX_FILE_EXT["excel"] + ACCEPTED_NPX_FILE_EXT["delim"] + ACCEPTED_NPX_FILE_EXT["parquet"]
        )
        raise ValueError(f"Unsupported NPX file extension {suffix!r}. Expected one of {supported}.")

    if attach_log:
        check_log = check_npx(df, preferred_names=preferred_names)
        df = attach_check_log(df, check_log)
    return df


def read_NPX(
    filename: str | Path,
    preferred_names: Mapping[str, str] | None = None,
    *,
    attach_log: bool = True,
) -> pd.DataFrame:
    """Alias for :func:`read_npx` matching the R function name."""
    return read_npx(filename, preferred_names=preferred_names, attach_log=attach_log)


def _read_delimited(path: Path) -> pd.DataFrame:
    delimiter = _detect_delimiter(path)
    return pd.read_csv(path, sep=delimiter)


def _detect_delimiter(path: Path) -> str:
    with path.open(encoding="utf-8", errors="replace") as handle:
        sample = handle.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(ACCEPTED_FIELD_SEP))
        return dialect.delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        counts = {sep: first_line.count(sep) for sep in ACCEPTED_FIELD_SEP}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","
