"""Small shared helpers."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import pandas as pd

from olinkanalyze._constants import CHECK_LOG_ATTR, CHECK_NPX_LST_NAMES


def require_dataframe(df: Any, *, name: str = "df") -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(df).__name__}.")
    return df


def validate_check_log(check_log: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(check_log, Mapping):
        raise TypeError("check_log must be a mapping returned by check_npx().")
    missing = [key for key in CHECK_NPX_LST_NAMES if key not in check_log]
    if missing:
        raise ValueError(
            "check_log is missing expected keys: " + ", ".join(missing) + ". Pass the output of check_npx()."
        )
    return dict(check_log)


def extract_check_log(df: pd.DataFrame) -> dict[str, Any] | None:
    log = df.attrs.get(CHECK_LOG_ATTR)
    if log is None:
        return None
    return _restore_check_log(validate_check_log(log))


def _restore_check_log(check_log: dict[str, Any]) -> dict[str, Any]:
    restored = dict(check_log)
    for key in ("col_class", "darid_invalid"):
        value = restored.get(key)
        if isinstance(value, list):
            restored[key] = pd.DataFrame(value)
    return restored


def _sanitize_check_log(check_log: dict[str, Any]) -> dict[str, Any]:
    """Keep check_log JSON-like so pandas can copy/compare DataFrame.attrs."""
    sanitized = dict(check_log)
    col_class = sanitized.get("col_class")
    if isinstance(col_class, pd.DataFrame):
        sanitized["col_class"] = col_class.to_dict(orient="records")
    darid = sanitized.get("darid_invalid")
    if isinstance(darid, pd.DataFrame):
        sanitized["darid_invalid"] = darid.to_dict(orient="records")
    return sanitized


def attach_check_log(df: pd.DataFrame, check_log: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out.attrs[CHECK_LOG_ATTR] = _sanitize_check_log(validate_check_log(check_log))
    return out


def get_check_log(
    df: pd.DataFrame,
    check_log: Mapping[str, Any] | None = None,
    preferred_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Use an attached log, an explicit log, or run check_npx()."""
    attached = extract_check_log(df)
    if attached is not None:
        return attached
    if check_log is not None:
        return validate_check_log(check_log)

    from olinkanalyze.qc.check_npx import check_npx

    warnings.warn(
        "No attached or supplied check_log found. Running check_npx(). "
        "Call check_npx() directly when you want to inspect the full data validity check output.",
        UserWarning,
        stacklevel=2,
    )
    return check_npx(df, preferred_names=preferred_names)


def col(check_log: Mapping[str, Any], key: str) -> str:
    try:
        return check_log["col_names"][key]
    except KeyError as exc:
        raise KeyError(f"Column key {key!r} is not available in check_log['col_names'].") from exc
