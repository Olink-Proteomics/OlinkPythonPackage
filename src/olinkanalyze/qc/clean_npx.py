"""Clean NPX data using a check_npx() log."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import pandas as pd

from olinkanalyze._constants import OLINK_ASSAY_TYPES, OLINK_SAMPLE_TYPES
from olinkanalyze._utils import attach_check_log, col, get_check_log, require_dataframe
from olinkanalyze.qc.check_npx import check_npx


def clean_npx(
    df: pd.DataFrame,
    check_log: Mapping | None = None,
    *,
    remove_assay_na: bool = True,
    remove_invalid_oid: bool = True,
    remove_dup_sample_id: bool = True,
    remove_control_assay: bool | Sequence[str] = True,
    remove_control_sample: bool | Sequence[str] = True,
    remove_qc_warning: bool = True,
    remove_assay_warning: bool = True,
    control_sample_ids: Sequence[str] | None = None,
    convert_df_cols: bool = True,
    convert_nonunique_uniprot: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Remove records that should not go into downstream analysis."""
    require_dataframe(df)
    check_log = get_check_log(df, check_log=check_log)
    out = df.copy()

    out = _clean_invalid_oid(out, check_log, remove_invalid_oid, verbose)
    out = _clean_assay_na(out, check_log, remove_assay_na, verbose)
    out = _clean_duplicate_sample_id(out, check_log, remove_dup_sample_id, verbose)
    out = _clean_sample_type(out, check_log, remove_control_sample, verbose)
    out = _clean_control_sample_id(out, check_log, control_sample_ids, verbose)
    out = _clean_qc_warning(out, check_log, remove_qc_warning, verbose)
    out = _clean_assay_type(out, check_log, remove_control_assay, verbose)
    out = _clean_assay_warning(out, check_log, remove_assay_warning, verbose)
    out = _clean_col_class(out, check_log, convert_df_cols, verbose)
    out = _clean_nonunique_uniprot(out, check_log, convert_nonunique_uniprot, verbose)

    quant_name = str(check_log["col_names"].get("quant", ""))
    if "quantified" in quant_name.lower():
        warnings.warn(
            f"Detected absolute quantification in column {quant_name}. "
            "A log2 transformation is recommended before analysis.",
            UserWarning,
            stacklevel=2,
        )

    updated_log = check_npx(out)
    return attach_check_log(out, updated_log)


def _inform(verbose: bool, message: str) -> None:
    if verbose:
        warnings.warn(message, UserWarning, stacklevel=4)


def _clean_invalid_oid(df: pd.DataFrame, check_log: Mapping, enabled: bool, verbose: bool) -> pd.DataFrame:
    if not enabled or len(check_log["oid_invalid"]) == 0:
        _inform(verbose, "Skipping or not needed: invalid OlinkID filter.")
        return df
    oid_col = col(check_log, "olink_id")
    return df.loc[~df[oid_col].isin(check_log["oid_invalid"])].copy()


def _clean_assay_na(df: pd.DataFrame, check_log: Mapping, enabled: bool, verbose: bool) -> pd.DataFrame:
    if not enabled or len(check_log["assay_na"]) == 0:
        _inform(verbose, "Skipping or not needed: all-NA assay filter.")
        return df
    oid_col = col(check_log, "olink_id")
    return df.loc[~df[oid_col].isin(check_log["assay_na"])].copy()


def _clean_duplicate_sample_id(
    df: pd.DataFrame, check_log: Mapping, enabled: bool, verbose: bool
) -> pd.DataFrame:
    if not enabled or len(check_log["sample_id_dups"]) == 0:
        _inform(verbose, "Skipping or not needed: duplicate SampleID filter.")
        return df
    sample_col = col(check_log, "sample_id")
    return df.loc[~df[sample_col].isin(check_log["sample_id_dups"])].copy()


def _control_values(
    flag: bool | Sequence[str],
    type_map: dict[str, tuple[str, ...]],
    keep_key: str,
    arg_name: str,
) -> list[str]:
    if isinstance(flag, bool):
        if not flag:
            return []
        return [value for key, values in type_map.items() if key != keep_key for value in values]

    selected = list(flag)
    known = [key for key in selected if key in type_map]
    unknown = [key for key in selected if key not in type_map]
    if not known:
        raise ValueError(f"{arg_name} must be a boolean or one or more of {list(type_map)}.")
    if unknown:
        warnings.warn(
            f"Unexpected entries {unknown} in {arg_name}. Proceeding with {known}.",
            UserWarning,
            stacklevel=4,
        )
    return [value for key in known for value in type_map[key]]


def _clean_sample_type(
    df: pd.DataFrame,
    check_log: Mapping,
    remove_control_sample: bool | Sequence[str],
    verbose: bool,
) -> pd.DataFrame:
    ctrl = _control_values(remove_control_sample, OLINK_SAMPLE_TYPES, "sample", "remove_control_sample")
    if not ctrl:
        _inform(verbose, "Skipping exclusion of control samples.")
        return df
    if "sample_type" not in check_log["col_names"]:
        _inform(
            True,
            "No column marking control samples in dataset. "
            "Ensure exclusion of control samples for downstream analysis.",
        )
        return df
    sample_type_col = col(check_log, "sample_type")
    values = df[sample_type_col].astype(str).str.upper()
    return df.loc[~values.isin({item.upper() for item in ctrl})].copy()


def _clean_assay_type(
    df: pd.DataFrame,
    check_log: Mapping,
    remove_control_assay: bool | Sequence[str],
    verbose: bool,
) -> pd.DataFrame:
    ctrl = _control_values(remove_control_assay, OLINK_ASSAY_TYPES, "assay", "remove_control_assay")
    if not ctrl:
        _inform(verbose, "Skipping exclusion of control assays.")
        return df
    if "assay_type" not in check_log["col_names"]:
        _inform(
            True,
            "No column marking assay type in dataset. "
            "Ensure internal control assays are removed before analysis.",
        )
        return df
    assay_type_col = col(check_log, "assay_type")
    values = df[assay_type_col].astype(str).str.lower()
    return df.loc[~values.isin({item.lower() for item in ctrl})].copy()


def _clean_control_sample_id(
    df: pd.DataFrame,
    check_log: Mapping,
    control_sample_ids: Sequence[str] | None,
    verbose: bool,
) -> pd.DataFrame:
    if not control_sample_ids:
        _inform(verbose, "No extra control SampleIDs requested.")
        return df
    sample_col = col(check_log, "sample_id")
    present = df[sample_col].isin(list(control_sample_ids))
    if not present.any():
        _inform(verbose, "None of the requested control SampleIDs were present.")
        return df
    return df.loc[~present].copy()


def _clean_qc_warning(df: pd.DataFrame, check_log: Mapping, enabled: bool, verbose: bool) -> pd.DataFrame:
    if not enabled:
        _inform(verbose, "Skipping exclusion of samples flagged FAIL.")
        return df
    qc_col = col(check_log, "qc_warning")
    fail = df[qc_col].astype(str).str.contains("fail", case=False, na=False)
    if not fail.any():
        _inform(verbose, "No samples flagged FAIL.")
        return df
    return df.loc[~fail].copy()


def _clean_assay_warning(df: pd.DataFrame, check_log: Mapping, enabled: bool, verbose: bool) -> pd.DataFrame:
    if not enabled:
        _inform(verbose, "Skipping exclusion of assays flagged WARN.")
        return df
    if "assay_warn" not in check_log["col_names"]:
        _inform(
            True,
            "No column marking assay warnings in dataset. "
            "Ensure assays with QC warnings are removed before analysis.",
        )
        return df
    warn_col = col(check_log, "assay_warn")
    warn = df[warn_col].astype(str).str.contains("warn", case=False, na=False)
    if not warn.any():
        _inform(verbose, "No assays flagged WARN.")
        return df
    return df.loc[~warn].copy()


def _clean_col_class(df: pd.DataFrame, check_log: Mapping, enabled: bool, verbose: bool) -> pd.DataFrame:
    if not enabled:
        _inform(verbose, "Skipping column type conversion.")
        return df
    col_class = check_log["col_class"]
    if col_class is None or len(col_class) == 0:
        _inform(verbose, "Columns are already in the expected format.")
        return df
    if isinstance(col_class, pd.DataFrame):
        rows = col_class.to_dict(orient="records")
    else:
        rows = list(col_class)
    out = df.copy()
    for row in rows:
        name = row["col_name"]
        expected = row["expected_col_class"]
        if expected == "numeric":
            out[name] = pd.to_numeric(out[name], errors="coerce")
        elif expected == "character":
            out[name] = out[name].astype(str)
    return out


def _clean_nonunique_uniprot(
    df: pd.DataFrame, check_log: Mapping, enabled: bool, verbose: bool
) -> pd.DataFrame:
    if not enabled or len(check_log["non_unique_uniprot"]) == 0:
        _inform(verbose, "Skipping or not needed: UniProt unification.")
        return df
    oid_col = col(check_log, "olink_id")
    uniprot_col = col(check_log, "uniprot")
    out = df.copy()
    for oid in check_log["non_unique_uniprot"]:
        mask = out[oid_col] == oid
        keep = out.loc[mask, uniprot_col].dropna().astype(str).iloc[0]
        out.loc[mask, uniprot_col] = keep
    return out
