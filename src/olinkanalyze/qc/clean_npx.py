"""Clean NPX data using a check_npx() log."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import pandas as pd

from olinkanalyze._constants import OLINK_ASSAY_TYPES, OLINK_SAMPLE_TYPES
from olinkanalyze._utils import (
    attach_check_log,
    col,
    extract_check_log,
    get_check_log,
    require_dataframe,
    series_str_contains,
    series_str_transform,
)
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
    keep = pd.Series(True, index=df.index)

    keep &= _mask_invalid_oid(df, check_log, remove_invalid_oid, verbose)
    keep &= _mask_assay_na(df, check_log, remove_assay_na, verbose)
    keep &= _mask_duplicate_sample_id(df, check_log, remove_dup_sample_id, verbose)
    keep &= _mask_sample_type(df, check_log, remove_control_sample, verbose)
    keep &= _mask_control_sample_id(df, check_log, control_sample_ids, verbose)
    keep &= _mask_qc_warning(df, check_log, remove_qc_warning, verbose)
    keep &= _mask_assay_type(df, check_log, remove_control_assay, verbose)
    keep &= _mask_assay_warning(df, check_log, remove_assay_warning, verbose)

    class_rows = _col_class_rows(check_log) if convert_df_cols else []
    needs_filter = not bool(keep.all())
    needs_types = convert_df_cols and bool(class_rows)
    needs_uniprot = convert_nonunique_uniprot and len(check_log["non_unique_uniprot"]) > 0

    if not convert_df_cols:
        _inform(verbose, "Skipping column type conversion.")
    elif not class_rows:
        _inform(verbose, "Columns are already in the expected format.")
    if not needs_uniprot:
        _inform(verbose, "Skipping or not needed: UniProt unification.")

    quant_name = str(check_log["col_names"].get("quant", ""))
    if "quantified" in quant_name.lower():
        warnings.warn(
            f"Detected absolute quantification in column {quant_name}. "
            "A log2 transformation is recommended before analysis.",
            UserWarning,
            stacklevel=2,
        )

    if not needs_filter and not needs_types and not needs_uniprot:
        if extract_check_log(df) is not None:
            return df
        return attach_check_log(df, check_log)

    out = df.loc[keep].copy()
    if needs_types:
        out = _apply_col_class(out, class_rows)
    if needs_uniprot:
        out = _apply_nonunique_uniprot(out, check_log)
    return attach_check_log(out, check_npx(out), copy=False)


def prepare_plot_npx(
    df: pd.DataFrame,
    check_log: Mapping | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Drop invalid OlinkIDs and all-NA assays without re-running ``check_npx``."""
    check_log = get_check_log(df, check_log=check_log)
    oid_col = col(check_log, "olink_id")
    drop_oids = set(check_log.get("oid_invalid") or ()) | set(check_log.get("assay_na") or ())
    out = df.loc[~df[oid_col].isin(drop_oids)].copy() if drop_oids else df.copy()
    if check_log.get("non_unique_uniprot"):
        out = _apply_nonunique_uniprot(out, check_log)
    return out, check_log


def _inform(verbose: bool, message: str) -> None:
    if verbose:
        warnings.warn(message, UserWarning, stacklevel=4)


def _keep_all(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index)


def _mask_invalid_oid(df: pd.DataFrame, check_log: Mapping, enabled: bool, verbose: bool) -> pd.Series:
    if not enabled or len(check_log["oid_invalid"]) == 0:
        _inform(verbose, "Skipping or not needed: invalid OlinkID filter.")
        return _keep_all(df)
    oid_col = col(check_log, "olink_id")
    return ~df[oid_col].isin(check_log["oid_invalid"])


def _mask_assay_na(df: pd.DataFrame, check_log: Mapping, enabled: bool, verbose: bool) -> pd.Series:
    if not enabled or len(check_log["assay_na"]) == 0:
        _inform(verbose, "Skipping or not needed: all-NA assay filter.")
        return _keep_all(df)
    oid_col = col(check_log, "olink_id")
    return ~df[oid_col].isin(check_log["assay_na"])


def _mask_duplicate_sample_id(
    df: pd.DataFrame, check_log: Mapping, enabled: bool, verbose: bool
) -> pd.Series:
    if not enabled or len(check_log["sample_id_dups"]) == 0:
        _inform(verbose, "Skipping or not needed: duplicate SampleID filter.")
        return _keep_all(df)
    sample_col = col(check_log, "sample_id")
    return ~df[sample_col].isin(check_log["sample_id_dups"])


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


def _mask_sample_type(
    df: pd.DataFrame,
    check_log: Mapping,
    remove_control_sample: bool | Sequence[str],
    verbose: bool,
) -> pd.Series:
    ctrl = _control_values(remove_control_sample, OLINK_SAMPLE_TYPES, "sample", "remove_control_sample")
    if not ctrl:
        _inform(verbose, "Skipping exclusion of control samples.")
        return _keep_all(df)
    if "sample_type" not in check_log["col_names"]:
        _inform(
            True,
            "No column marking control samples in dataset. "
            "Ensure exclusion of control samples for downstream analysis.",
        )
        return _keep_all(df)
    sample_type_col = col(check_log, "sample_type")
    values = series_str_transform(df[sample_type_col], "upper")
    return ~values.isin({item.upper() for item in ctrl})


def _mask_assay_type(
    df: pd.DataFrame,
    check_log: Mapping,
    remove_control_assay: bool | Sequence[str],
    verbose: bool,
) -> pd.Series:
    ctrl = _control_values(remove_control_assay, OLINK_ASSAY_TYPES, "assay", "remove_control_assay")
    if not ctrl:
        _inform(verbose, "Skipping exclusion of control assays.")
        return _keep_all(df)
    if "assay_type" not in check_log["col_names"]:
        _inform(
            True,
            "No column marking assay type in dataset. "
            "Ensure internal control assays are removed before analysis.",
        )
        return _keep_all(df)
    assay_type_col = col(check_log, "assay_type")
    values = series_str_transform(df[assay_type_col], "lower")
    return ~values.isin({item.lower() for item in ctrl})


def _mask_control_sample_id(
    df: pd.DataFrame,
    check_log: Mapping,
    control_sample_ids: Sequence[str] | None,
    verbose: bool,
) -> pd.Series:
    if not control_sample_ids:
        _inform(verbose, "No extra control SampleIDs requested.")
        return _keep_all(df)
    sample_col = col(check_log, "sample_id")
    present = df[sample_col].isin(list(control_sample_ids))
    if not present.any():
        _inform(verbose, "None of the requested control SampleIDs were present.")
        return _keep_all(df)
    return ~present


def _mask_qc_warning(df: pd.DataFrame, check_log: Mapping, enabled: bool, verbose: bool) -> pd.Series:
    if not enabled:
        _inform(verbose, "Skipping exclusion of samples flagged FAIL.")
        return _keep_all(df)
    qc_col = col(check_log, "qc_warning")
    fail = series_str_contains(df[qc_col], "fail", case=False)
    if not fail.any():
        _inform(verbose, "No samples flagged FAIL.")
        return _keep_all(df)
    return ~fail


def _mask_assay_warning(df: pd.DataFrame, check_log: Mapping, enabled: bool, verbose: bool) -> pd.Series:
    if not enabled:
        _inform(verbose, "Skipping exclusion of assays flagged WARN.")
        return _keep_all(df)
    if "assay_warn" not in check_log["col_names"]:
        _inform(
            True,
            "No column marking assay warnings in dataset. "
            "Ensure assays with QC warnings are removed before analysis.",
        )
        return _keep_all(df)
    warn_col = col(check_log, "assay_warn")
    warn = series_str_contains(df[warn_col], "warn", case=False)
    if not warn.any():
        _inform(verbose, "No assays flagged WARN.")
        return _keep_all(df)
    return ~warn


def _col_class_rows(check_log: Mapping) -> list[dict]:
    col_class = check_log["col_class"]
    if col_class is None or len(col_class) == 0:
        return []
    if isinstance(col_class, pd.DataFrame):
        return col_class.to_dict(orient="records")
    return list(col_class)


def _apply_col_class(df: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    for row in rows:
        name = row["col_name"]
        expected = row["expected_col_class"]
        if expected == "numeric":
            df[name] = pd.to_numeric(df[name], errors="coerce")
        elif expected == "character":
            df[name] = df[name].astype(str)
    return df


def _apply_nonunique_uniprot(df: pd.DataFrame, check_log: Mapping) -> pd.DataFrame:
    oid_col = col(check_log, "olink_id")
    uniprot_col = col(check_log, "uniprot")
    for oid in check_log["non_unique_uniprot"]:
        mask = df[oid_col] == oid
        keep = df.loc[mask, uniprot_col].dropna().astype(str).iloc[0]
        df.loc[mask, uniprot_col] = keep
    return df
