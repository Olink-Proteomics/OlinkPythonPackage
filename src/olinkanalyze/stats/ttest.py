"""Per-assay t-tests on long-format NPX data."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from olinkanalyze._constants import OLINK_ID_PATTERN
from olinkanalyze._utils import LOGGER, col, get_check_log, require_dataframe


def olink_ttest(
    df: pd.DataFrame,
    variable: str,
    pair_id: str | None = None,
    check_log: Mapping | None = None,
    *,
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Welch two-sample or paired t-test for every assay.

    ``variable`` must have exactly two levels. Multiple-testing correction
    uses Benjamini-Hochberg FDR. The estimate is mean(level1) - mean(level2)
    using sorted unique levels.
    """
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
            stacklevel=2,
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
                stacklevel=2,
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

    if len(levels) != 2:
        raise ValueError(
            f"The number of levels in the factor needs to be 2. Your factor has {len(levels)} levels."
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
            stacklevel=2,
        )

    work = work.loc[~work[oid_col].isin(assay_na | skip_assays)]
    if work.empty:
        raise ValueError("No assays passing initial check. T-test will not be performed.")

    level1, level2 = levels
    meta = _assay_meta(work, oid_col, assay_col, uniprot_col, panel_col)
    if pair_id is None:
        LOGGER.info("T-test is performed on %s - %s.", level1, level2)
        result = _welch_matrix(
            work,
            meta=meta,
            oid_col=oid_col,
            sample_col=sample_col,
            quant_col=quant_col,
            variable=variable,
            level1=level1,
            level2=level2,
            alternative=alternative,
            confidence_level=confidence_level,
        )
    else:
        LOGGER.info("Paired t-test is performed on %s - %s.", level1, level2)
        pair_counts = (
            work.dropna(subset=[variable])
            .groupby([oid_col, pair_id], dropna=False, sort=False)[sample_col]
            .nunique()
        )
        too_many = pair_counts[pair_counts > 2]
        if not too_many.empty:
            bad_pairs = too_many.index.get_level_values(pair_id).unique().tolist()
            raise ValueError(
                "Each pair identifier must identify no more than 2 unique samples. "
                f"Check pairs: {', '.join(map(str, bad_pairs))}"
            )
        result = _paired_matrix(
            work,
            meta=meta,
            oid_col=oid_col,
            pair_id=pair_id,
            quant_col=quant_col,
            variable=variable,
            level1=level1,
            level2=level2,
            alternative=alternative,
            confidence_level=confidence_level,
        )

    if result.empty:
        raise ValueError("No assays passing initial check. T-test will not be performed.")

    _rejected, p_adj, _alphac_sidak, _alphac_bonf = multipletests(
        result["p.value"].to_numpy(), method="fdr_bh"
    )
    result["Adjusted_pval"] = p_adj
    result["Threshold"] = np.where(result["Adjusted_pval"] < 0.05, "Significant", "Non-significant")
    return result.sort_values("p.value", kind="mergesort").reset_index(drop=True)


def _assay_meta(
    work: pd.DataFrame,
    oid_col: str,
    assay_col: str,
    uniprot_col: str,
    panel_col: str,
) -> pd.DataFrame:
    return work.groupby(oid_col, sort=False, dropna=False).agg(
        Assay=(assay_col, "first"),
        UniProt=(uniprot_col, "first"),
        Panel=(panel_col, "first"),
    )


def _result_frame(
    meta: pd.DataFrame,
    index,
    *,
    level1,
    level2,
    mean1: np.ndarray,
    mean2: np.ndarray,
    test,
    confidence_level: float,
    method: str,
    alternative: str,
) -> pd.DataFrame:
    ci = test.confidence_interval(confidence_level=confidence_level)
    out = meta.reindex(index).copy()
    out["OlinkID"] = out.index
    out["estimate"] = mean1 - mean2
    out[str(level1)] = mean1
    out[str(level2)] = mean2
    out["statistic"] = np.asarray(test.statistic, dtype=float)
    out["p.value"] = np.asarray(test.pvalue, dtype=float)
    out["parameter"] = np.asarray(test.df, dtype=float)
    out["conf.low"] = np.asarray(ci.low, dtype=float)
    out["conf.high"] = np.asarray(ci.high, dtype=float)
    out["method"] = method
    out["alternative"] = alternative
    columns = [
        "Assay",
        "OlinkID",
        "UniProt",
        "Panel",
        "estimate",
        str(level1),
        str(level2),
        "statistic",
        "p.value",
        "parameter",
        "conf.low",
        "conf.high",
        "method",
        "alternative",
    ]
    return out.loc[:, columns].reset_index(drop=True)


def _welch_matrix(
    work: pd.DataFrame,
    *,
    meta: pd.DataFrame,
    oid_col: str,
    sample_col: str,
    quant_col: str,
    variable: str,
    level1,
    level2,
    alternative: str,
    confidence_level: float,
) -> pd.DataFrame:
    wide = work.pivot_table(
        index=oid_col,
        columns=sample_col,
        values=quant_col,
        aggfunc="mean",
        observed=True,
    )
    sample_level = work.groupby(sample_col, sort=False, dropna=False)[variable].first()
    aligned = sample_level.reindex(wide.columns)
    x1 = wide.loc[:, aligned.astype(str) == str(level1)].to_numpy(dtype=float, copy=False)
    x2 = wide.loc[:, aligned.astype(str) == str(level2)].to_numpy(dtype=float, copy=False)
    n1 = np.isfinite(x1).sum(axis=1)
    n2 = np.isfinite(x2).sum(axis=1)
    ok = (n1 >= 2) & (n2 >= 2)
    if not ok.any():
        return pd.DataFrame()
    test = stats.ttest_ind(
        x1[ok],
        x2[ok],
        axis=1,
        equal_var=False,
        nan_policy="omit",
        alternative=alternative,
    )
    return _result_frame(
        meta,
        wide.index[ok],
        level1=level1,
        level2=level2,
        mean1=np.nanmean(x1[ok], axis=1),
        mean2=np.nanmean(x2[ok], axis=1),
        test=test,
        confidence_level=confidence_level,
        method="Welch Two Sample t-test",
        alternative=alternative,
    )


def _paired_matrix(
    work: pd.DataFrame,
    *,
    meta: pd.DataFrame,
    oid_col: str,
    pair_id: str,
    quant_col: str,
    variable: str,
    level1,
    level2,
    alternative: str,
    confidence_level: float,
) -> pd.DataFrame:
    cell = work.pivot_table(
        index=[oid_col, pair_id],
        columns=variable,
        values=quant_col,
        aggfunc="mean",
        observed=True,
    ).reindex(columns=[level1, level2])
    x = cell[level1].unstack(pair_id)
    y = cell[level2].unstack(pair_id)
    x, y = x.align(y, join="outer", axis=1)
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    both = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = np.where(both, x_arr, np.nan)
    y_arr = np.where(both, y_arr, np.nan)
    ok = both.sum(axis=1) >= 2
    if not ok.any():
        return pd.DataFrame()
    test = stats.ttest_rel(
        x_arr[ok],
        y_arr[ok],
        axis=1,
        nan_policy="omit",
        alternative=alternative,
    )
    return _result_frame(
        meta,
        x.index[ok],
        level1=level1,
        level2=level2,
        mean1=np.nanmean(x_arr[ok], axis=1),
        mean2=np.nanmean(y_arr[ok], axis=1),
        test=test,
        confidence_level=confidence_level,
        method="Paired t-test",
        alternative=alternative,
    )
