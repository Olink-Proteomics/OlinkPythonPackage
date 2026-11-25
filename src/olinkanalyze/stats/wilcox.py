"""Per-assay Wilcoxon tests on long-format NPX data."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from olinkanalyze._utils import LOGGER
from olinkanalyze.stats._prepare import apply_fdr, assay_meta, prepare_stat_work, wide_by_level


def olink_wilcox(
    df: pd.DataFrame,
    variable: str,
    pair_id: str | None = None,
    check_log: Mapping | None = None,
    *,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Wilcoxon rank-sum or signed-rank test for every assay.

    ``variable`` must have exactly two levels. Multiple-testing correction
    uses Benjamini-Hochberg FDR. The estimate is median(level1) - median(level2)
    using sorted unique levels.
    """
    prepared = prepare_stat_work(df, variable, check_log=check_log, pair_id=pair_id, exact_levels=2)
    level1, level2 = prepared.levels
    meta = assay_meta(prepared)
    if pair_id is None:
        LOGGER.info("Wilcoxon rank-sum test is performed on %s - %s.", level1, level2)
        result = _rank_sum(prepared, meta=meta, level1=level1, level2=level2, alternative=alternative)
    else:
        LOGGER.info("Wilcoxon signed-rank test is performed on %s - %s.", level1, level2)
        pair_counts = (
            prepared.work.dropna(subset=[prepared.variable])
            .groupby([prepared.oid_col, pair_id], dropna=False, sort=False)[prepared.sample_col]
            .nunique()
        )
        too_many = pair_counts[pair_counts > 2]
        if not too_many.empty:
            bad_pairs = too_many.index.get_level_values(pair_id).unique().tolist()
            raise ValueError(
                "Each pair identifier must identify no more than 2 unique samples. "
                f"Check pairs: {', '.join(map(str, bad_pairs))}"
            )
        result = _signed_rank(
            prepared,
            meta=meta,
            pair_id=pair_id,
            level1=level1,
            level2=level2,
            alternative=alternative,
        )
    if result.empty:
        raise ValueError("No assays passing initial check. The test will not be performed.")
    return apply_fdr(result)


def _rank_sum(
    prepared,
    *,
    meta: pd.DataFrame,
    level1,
    level2,
    alternative: str,
) -> pd.DataFrame:
    wide, aligned = wide_by_level(prepared)
    x1 = wide.loc[:, aligned.astype(str) == str(level1)].to_numpy(dtype=float, copy=False)
    x2 = wide.loc[:, aligned.astype(str) == str(level2)].to_numpy(dtype=float, copy=False)
    n1 = np.isfinite(x1).sum(axis=1)
    n2 = np.isfinite(x2).sum(axis=1)
    ok = (n1 >= 2) & (n2 >= 2)
    if not ok.any():
        return pd.DataFrame()
    test = stats.mannwhitneyu(x1[ok], x2[ok], axis=1, nan_policy="omit", alternative=alternative)
    return _two_level_frame(
        meta,
        wide.index[ok],
        level1=level1,
        level2=level2,
        loc1=np.nanmedian(x1[ok], axis=1),
        loc2=np.nanmedian(x2[ok], axis=1),
        statistic=np.asarray(test.statistic, dtype=float),
        pvalue=np.asarray(test.pvalue, dtype=float),
        method="Wilcoxon rank-sum test",
        alternative=alternative,
    )


def _signed_rank(
    prepared,
    *,
    meta: pd.DataFrame,
    pair_id: str,
    level1,
    level2,
    alternative: str,
) -> pd.DataFrame:
    cell = prepared.work.pivot_table(
        index=[prepared.oid_col, pair_id],
        columns=prepared.variable,
        values=prepared.quant_col,
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
    test = stats.wilcoxon(
        x_arr[ok],
        y_arr[ok],
        axis=1,
        nan_policy="omit",
        alternative=alternative,
        zero_method="wilcox",
    )
    return _two_level_frame(
        meta,
        x.index[ok],
        level1=level1,
        level2=level2,
        loc1=np.nanmedian(x_arr[ok], axis=1),
        loc2=np.nanmedian(y_arr[ok], axis=1),
        statistic=np.asarray(test.statistic, dtype=float),
        pvalue=np.asarray(test.pvalue, dtype=float),
        method="Wilcoxon signed-rank test",
        alternative=alternative,
    )


def _two_level_frame(
    meta: pd.DataFrame,
    index,
    *,
    level1,
    level2,
    loc1: np.ndarray,
    loc2: np.ndarray,
    statistic: np.ndarray,
    pvalue: np.ndarray,
    method: str,
    alternative: str,
) -> pd.DataFrame:
    out = meta.reindex(index).copy()
    out["OlinkID"] = out.index
    out["estimate"] = loc1 - loc2
    out[str(level1)] = loc1
    out[str(level2)] = loc2
    out["statistic"] = statistic
    out["p.value"] = pvalue
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
        "method",
        "alternative",
    ]
    return out.loc[:, columns].reset_index(drop=True)
