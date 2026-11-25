"""Validate long-format Olink NPX data."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import pandas as pd

from olinkanalyze._constants import (
    CHECK_NPX_LST_NAMES,
    COLUMN_SPEC_BY_KEY,
    OLINK_ID_PATTERN,
    OUTDATED_DARID,
    OUTDATED_DARID_MIN_VERSION,
)
from olinkanalyze._utils import require_dataframe
from olinkanalyze.qc.column_names import resolve_column_names


def check_npx(
    df: pd.DataFrame,
    preferred_names: Mapping[str, str] | None = None,
) -> dict:
    """Run format and QC checks on NPX data.

    Returns a named dictionary that downstream functions consume as
    ``check_log``. The input frame is not modified.
    """
    require_dataframe(df)
    col_names = resolve_column_names(df, preferred_names=preferred_names)

    check_log = {
        "col_names": col_names,
        "oid_invalid": _check_olinkid(df, col_names),
        "assay_na": _check_all_na_assays(df, col_names),
        "sample_id_dups": _check_duplicate_sample_ids(df, col_names),
        "sample_id_na": _check_all_na_samples(df, col_names),
        "col_class": _check_col_class(df, col_names),
        "assay_qc": _check_assay_qc(df, col_names),
        "non_unique_uniprot": _check_nonunique_uniprot(df, col_names),
        "darid_invalid": _check_darid(df, col_names),
    }
    # Keep a stable key order for check_log consumers.
    return {key: check_log[key] for key in CHECK_NPX_LST_NAMES}


def _series(df: pd.DataFrame, col_names: dict[str, str], key: str) -> pd.Series:
    return df[col_names[key]]


def _check_olinkid(df: pd.DataFrame, col_names: dict[str, str]) -> list[str]:
    ids = _series(df, col_names, "olink_id").dropna().astype(str).unique()
    invalid = sorted(oid for oid in ids if not OLINK_ID_PATTERN.match(oid))
    if invalid:
        warnings.warn(
            f"Unrecognized OlinkID(s) detected: {invalid}. Consider running clean_npx().",
            UserWarning,
            stacklevel=3,
        )
    return invalid


def _check_all_na_assays(df: pd.DataFrame, col_names: dict[str, str]) -> list[str]:
    oid = col_names["olink_id"]
    quant = col_names["quant"]
    na_share = df.groupby(oid, dropna=False)[quant].apply(lambda s: s.isna().all())
    all_na = sorted(na_share[na_share].index.astype(str).tolist())
    if all_na:
        warnings.warn(
            f"{all_na} have {quant} = NA for all samples. Consider running clean_npx().",
            UserWarning,
            stacklevel=3,
        )
    return all_na


def _check_duplicate_sample_ids(df: pd.DataFrame, col_names: dict[str, str]) -> list[str]:
    sample = col_names["sample_id"]
    oid = col_names["olink_id"]
    counts = df.groupby([sample, oid], dropna=False).size()
    duplicates = sorted({str(sample_id) for sample_id, _oid in counts[counts > 1].index})
    if duplicates:
        warnings.warn(
            f"Duplicate SampleID(s) detected: {duplicates}. Consider running clean_npx().",
            UserWarning,
            stacklevel=3,
        )
    return duplicates


def _check_all_na_samples(df: pd.DataFrame, col_names: dict[str, str]) -> list[str]:
    sample = col_names["sample_id"]
    quant = col_names["quant"]
    na_share = df.groupby(sample, dropna=False)[quant].apply(lambda s: s.isna().all())
    all_na = sorted(na_share[na_share].index.astype(str).tolist())
    if all_na:
        warnings.warn(
            f"{all_na} have {quant} = NA for all assays. Consider running clean_npx().",
            UserWarning,
            stacklevel=3,
        )
    return all_na


def _pandas_class(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        # object columns that are all strings count as character
        if series.dropna().map(lambda x: isinstance(x, str)).all():
            return "character"
        if pd.api.types.is_string_dtype(series):
            return "character"
    if pd.api.types.is_bool_dtype(series):
        return "other"
    return "other"


def _check_col_class(df: pd.DataFrame, col_names: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for key, name in col_names.items():
        spec = COLUMN_SPEC_BY_KEY[key]
        if not spec["col_class_check"]:
            continue
        detected = _pandas_class(df[name])
        expected = spec["col_class"]
        if detected != expected:
            rows.append(
                {
                    "col_key": key,
                    "col_name": name,
                    "col_class": detected,
                    "expected_col_class": expected,
                }
            )
    result = pd.DataFrame(rows, columns=["col_key", "col_name", "col_class", "expected_col_class"])
    if not result.empty:
        details = "; ".join(
            f"{row.col_name}: expected {row.expected_col_class}, detected {row.col_class}"
            for row in result.itertuples()
        )
        warnings.warn(
            f"Detected columns with incorrect data type(s): {details}. "
            "Use clean_npx() or convert the columns before downstream analyses.",
            UserWarning,
            stacklevel=3,
        )
    return result


def _check_assay_qc(df: pd.DataFrame, col_names: dict[str, str]) -> list[str]:
    if "assay_warn" not in col_names:
        return []
    warn_col = col_names["assay_warn"]
    flagged = (
        df.loc[
            df[warn_col].astype(str).str.contains("warn", case=False, na=False),
            col_names["olink_id"],
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    flagged = sorted(flagged)
    if flagged:
        warnings.warn(
            f"{len(flagged)} assay(s) exhibited assay QC warnings in column "
            f"{warn_col}: {flagged}. Consider running clean_npx().",
            UserWarning,
            stacklevel=3,
        )
    return flagged


def _check_nonunique_uniprot(df: pd.DataFrame, col_names: dict[str, str]) -> list[str]:
    oid = col_names["olink_id"]
    uniprot = col_names["uniprot"]
    n_ids = df[[oid, uniprot]].drop_duplicates().groupby(oid)[uniprot].nunique()
    dups = sorted(n_ids[n_ids > 1].index.astype(str).tolist())
    if dups:
        warnings.warn(
            f"Detected multiple UniProt identifiers for assay(s): {dups}. Consider running clean_npx().",
            UserWarning,
            stacklevel=3,
        )
    return dups


def _parse_version(value: object) -> tuple[int, ...]:
    text = str(value)
    parts: list[int] = []
    for token in text.replace("-", ".").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def _check_darid(df: pd.DataFrame, col_names: dict[str, str]) -> pd.DataFrame:
    panel_version = col_names.get("panel_version")
    empty = pd.DataFrame(
        columns=[panel_version or "panel_version", col_names.get("qc_version", "qc_version")]
    )
    if "qc_version" not in col_names or panel_version is None:
        return empty

    subset = df[[panel_version, col_names["qc_version"]]].drop_duplicates()
    mask = subset[panel_version].astype(str).isin(OUTDATED_DARID)
    outdated = subset.loc[mask].copy()
    if outdated.empty:
        return empty

    too_old = outdated[outdated[col_names["qc_version"]].map(_parse_version) < OUTDATED_DARID_MIN_VERSION]
    if not too_old.empty:
        warnings.warn(
            "Outdated DataAnalysisRefID and PanelDataArchiveVersion combinations "
            f"detected: {too_old.to_dict(orient='records')}.",
            UserWarning,
            stacklevel=3,
        )
    return too_old.reset_index(drop=True)
