"""Resolve Olink column aliases to internal keys."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import pandas as pd

from olinkanalyze._constants import COLUMN_NAME_DICT, COLUMN_SPEC_BY_KEY


def resolve_column_names(
    df: pd.DataFrame,
    preferred_names: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Map internal column keys to names present in ``df``.

    Parameters
    ----------
    df
        Long-format NPX data.
    preferred_names
        Optional mapping of internal keys (for example ``"quant"``) to the
        column that should be used when several aliases exist.
    """
    preferred = dict(preferred_names) if preferred_names else {}
    unknown = [key for key in preferred if key not in COLUMN_SPEC_BY_KEY]
    if unknown:
        expected = ", ".join(spec["col_key"] for spec in COLUMN_NAME_DICT)
        raise ValueError(
            f"Unexpected names in preferred_names: {unknown}. Expected one or more of: {expected}."
        )
    if len(preferred) != len(set(preferred)):
        raise ValueError("preferred_names must have unique keys.")

    columns = list(df.columns)
    column_set = set(columns)
    selected: dict[str, str] = {}
    missing_required: list[str] = []
    multi_ties: list[str] = []

    for spec in COLUMN_NAME_DICT:
        key = spec["col_key"]
        if key in preferred:
            name = preferred[key]
            if name not in column_set:
                raise ValueError(f"preferred_names[{key!r}] = {name!r} is not a column of df.")
            selected[key] = name
            continue

        matches = [name for name in spec["col_names"] if name in column_set]
        if not matches:
            if not spec["col_miss"]:
                missing_required.append(f"{key} (expected one of {list(spec['col_names'])})")
            continue

        if len(matches) > 1 and not spec["col_multi"] and not spec["col_order"]:
            raise ValueError(
                f"More than one column in df is associated with key {key!r}: "
                f"{matches}. Use preferred_names to break the tie."
            )

        chosen = matches[0]
        selected[key] = chosen
        if len(matches) > 1:
            multi_ties.append(f"{key}: {chosen!r} was selected. Options were {matches}.")

    if missing_required:
        raise ValueError("Required NPX columns are missing from df: " + "; ".join(missing_required))

    if multi_ties:
        warnings.warn(
            "More than one column name in df was associated with certain keys. "
            "One was selected based on an ordered list. "
            + " ".join(multi_ties)
            + " Use preferred_names to select a different column.",
            UserWarning,
            stacklevel=2,
        )

    return selected


_EXTRA_CATEGORY_COLUMNS = ("Treatment", "Subject", "Site", "Time", "Project")


def categorize_npx_columns(
    df: pd.DataFrame,
    col_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Convert repeated label columns to ``category`` dtypes.

    Numeric quantification columns are left unchanged. ``check_npx`` should
    run on the original dtypes first so QC type checks stay meaningful.
    """
    names: set[str] = set()
    if col_names:
        for key, name in col_names.items():
            spec = COLUMN_SPEC_BY_KEY.get(key)
            if spec and spec["col_class"] == "character":
                names.add(name)
    else:
        for spec in COLUMN_NAME_DICT:
            if spec["col_class"] == "character":
                names.update(name for name in spec["col_names"] if name in df.columns)
    names.update(name for name in _EXTRA_CATEGORY_COLUMNS if name in df.columns)

    out = df
    copied = False
    for name in names:
        series = out[name]
        if isinstance(series.dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_numeric_dtype(series):
            continue
        if not copied:
            out = df.copy()
            copied = True
        out[name] = series.astype("category")
    return out


def drop_unused_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unused category levels after row filters."""
    updates = {
        name: df[name].cat.remove_unused_categories()
        for name in df.columns
        if isinstance(df[name].dtype, pd.CategoricalDtype)
    }
    if not updates:
        return df
    return df.assign(**updates)
