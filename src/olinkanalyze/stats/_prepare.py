"""Shared setup for per-assay statistical tests."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from olinkanalyze._constants import OLINK_ID_PATTERN
from olinkanalyze._utils import LOGGER, col, get_check_log, require_dataframe


@dataclass(frozen=True)
class StatWork:
    work: pd.DataFrame
    levels: list
    check_log: dict
    oid_col: str
    sample_col: str
    quant_col: str
    assay_col: str
    uniprot_col: str
    panel_col: str
    variable: str


def prepare_stat_work(
    df: pd.DataFrame,
    variable: str,
    check_log: Mapping | None = None,
    *,
    pair_id: str | None = None,
    exact_levels: int | None = None,
    min_levels: int = 2,
) -> StatWork:
    require_dataframe(df)
    if variable not in df.columns:
        raise ValueError("The df and variable arguments need to be specified.")
    check_log = get_check_log(df, check_log=check_log)

    oid_col = col(check_log, "olink_id")
    sample_col = col(check_log, "sample_id")
    quant_col = col(check_log, "quant")
    assay_col = col(check_log, "assay")
    uniprot_col = col(check_log, "uniprot")
    panel_col = col(check_log, "panel")

    oid_as_str = df[oid_col].astype(str)
    unique_oids = pd.Index(oid_as_str.unique())
    valid_oids = unique_oids[unique_oids.str.fullmatch(OLINK_ID_PATTERN.pattern, na=False)]
    work = df.loc[oid_as_str.isin(valid_oids)].copy()

    missing_var = work[variable].isna()
    if missing_var.any():
        removed = work.loc[missing_var, sample_col].unique().tolist()
        warnings.warn(
            "Samples removed due to missing variable levels: " + ", ".join(map(str, removed)),
            UserWarning,
            stacklevel=3,
        )
        work = work.loc[~missing_var]

    if pair_id is not None:
        if pair_id not in work.columns:
            raise ValueError(f"Column {pair_id} not found.")
        missing_pair = work[pair_id].isna()
        if missing_pair.any():
            removed = work.loc[missing_pair, sample_col].unique().tolist()
            warnings.warn(
                "Samples removed due to missing pair ID: " + ", ".join(map(str, removed)),
                UserWarning,
                stacklevel=3,
            )
            work = work.loc[~missing_pair]

    if isinstance(work[variable].dtype, pd.CategoricalDtype):
        levels = [level for level in work[variable].cat.categories if level in set(work[variable])]
    else:
        if not pd.api.types.is_string_dtype(work[variable]) and work[variable].dtype != object:
            raise TypeError(
                f"The grouping variable {variable} is neither factor nor character. "
                "Only character and factor variable types allowed."
            )
        LOGGER.info("Variable converted from character to factor: %s", variable)
        levels = sorted(work[variable].dropna().astype(str).unique().tolist())
        work[variable] = pd.Categorical(work[variable].astype(str), categories=levels)

    if exact_levels is not None and len(levels) != exact_levels:
        raise ValueError(
            f"The number of levels in the factor needs to be {exact_levels}. "
            f"Your factor has {len(levels)} levels."
        )
    if len(levels) < min_levels:
        raise ValueError(
            f"The number of levels in the factor needs to be at least {min_levels}. "
            f"Your factor has {len(levels)} levels."
        )

    n_bad = work.groupby(sample_col, dropna=False, sort=False)[variable].nunique(dropna=True).gt(1).sum()
    if n_bad:
        raise ValueError(
            f"There are {int(n_bad)} samples that do not have a unique level "
            "for your variable. Only one level per sample is allowed."
        )

    assay_na = set(check_log.get("assay_na") or [])
    tested = work.loc[~work[oid_col].isin(assay_na)]
    level_na = (
        tested[quant_col]
        .isna()
        .groupby([tested[oid_col], tested[variable]], dropna=False, sort=False)
        .all()
    )
    skip_assays = set(level_na[level_na].index.get_level_values(oid_col).astype(str))
    if skip_assays:
        warnings.warn(
            "The assays "
            + ", ".join(sorted(skip_assays))
            + " have too few datapoints in one level of the factor. They will not be tested.",
            UserWarning,
            stacklevel=3,
        )

    work = work.loc[~work[oid_col].isin(assay_na | skip_assays)]
    if work.empty:
        raise ValueError("No assays passing initial check. The test will not be performed.")

    return StatWork(
        work=work,
        levels=list(levels),
        check_log=check_log,
        oid_col=oid_col,
        sample_col=sample_col,
        quant_col=quant_col,
        assay_col=assay_col,
        uniprot_col=uniprot_col,
        panel_col=panel_col,
        variable=variable,
    )


def assay_meta(work: StatWork) -> pd.DataFrame:
    return work.work.groupby(work.oid_col, sort=False, dropna=False).agg(
        Assay=(work.assay_col, "first"),
        UniProt=(work.uniprot_col, "first"),
        Panel=(work.panel_col, "first"),
    )


def apply_fdr(result: pd.DataFrame) -> pd.DataFrame:
    _rejected, p_adj, _alphac_sidak, _alphac_bonf = multipletests(
        result["p.value"].to_numpy(), method="fdr_bh"
    )
    result = result.copy()
    result["Adjusted_pval"] = p_adj
    result["Threshold"] = np.where(result["Adjusted_pval"] < 0.05, "Significant", "Non-significant")
    return result.sort_values("p.value", kind="mergesort").reset_index(drop=True)


def wide_by_level(work: StatWork) -> tuple[pd.DataFrame, pd.Series]:
    wide = work.work.pivot_table(
        index=work.oid_col,
        columns=work.sample_col,
        values=work.quant_col,
        aggfunc="mean",
        observed=True,
    )
    sample_level = work.work.groupby(work.sample_col, sort=False, dropna=False)[work.variable].first()
    return wide, sample_level.reindex(wide.columns)
