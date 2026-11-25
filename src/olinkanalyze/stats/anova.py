"""Per-assay one-way ANOVA on long-format NPX data."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from olinkanalyze._utils import LOGGER
from olinkanalyze.stats._prepare import apply_fdr, assay_meta, prepare_stat_work, wide_by_level
from olinkanalyze.stats.ttest import olink_ttest


def olink_anova(
    df: pd.DataFrame,
    variable: str,
    check_log: Mapping | None = None,
) -> pd.DataFrame:
    """One-way ANOVA for every assay.

    ``variable`` must have at least two levels. Multiple-testing correction
    uses Benjamini-Hochberg FDR.
    """
    prepared = prepare_stat_work(df, variable, check_log=check_log, min_levels=2)
    LOGGER.info("ANOVA is performed on %s (%s levels).", variable, len(prepared.levels))
    wide, aligned = wide_by_level(prepared)
    groups = []
    counts = []
    for level in prepared.levels:
        block = wide.loc[:, aligned.astype(str) == str(level)].to_numpy(dtype=float, copy=False)
        groups.append(block)
        counts.append(np.isfinite(block).sum(axis=1))
    ok = np.ones(len(wide), dtype=bool)
    for n in counts:
        ok &= n >= 2
    if not ok.any():
        raise ValueError("No assays passing initial check. The test will not be performed.")
    test = stats.f_oneway(*(block[ok] for block in groups), axis=1, nan_policy="omit")
    meta = assay_meta(prepared)
    out = meta.reindex(wide.index[ok]).copy()
    out["OlinkID"] = out.index
    out["statistic"] = np.asarray(test.statistic, dtype=float)
    out["p.value"] = np.asarray(test.pvalue, dtype=float)
    out["method"] = "One-way ANOVA"
    out["variable"] = prepared.variable
    columns = ["Assay", "OlinkID", "UniProt", "Panel", "statistic", "p.value", "method", "variable"]
    return apply_fdr(out.loc[:, columns].reset_index(drop=True))


def olink_anova_posthoc(
    df: pd.DataFrame,
    variable: str,
    check_log: Mapping | None = None,
    *,
    effect: str = "pairwise",
) -> pd.DataFrame:
    """Pairwise Welch t-tests after :func:`olink_anova`.

    ``effect`` must be ``"pairwise"``. Results are FDR-corrected across all
    assay-by-contrast tests.
    """
    if effect != "pairwise":
        raise ValueError("Only effect='pairwise' is implemented.")
    prepared = prepare_stat_work(df, variable, check_log=check_log, min_levels=2)
    frames = []
    for level1, level2 in combinations(prepared.levels, 2):
        keep = {str(level1), str(level2)}
        subset = prepared.work.loc[prepared.work[prepared.variable].astype(str).isin(keep)]
        pairwise = olink_ttest(subset, variable=prepared.variable, check_log=prepared.check_log)
        pairwise["contrast"] = f"{level1} - {level2}"
        frames.append(pairwise)
    if not frames:
        raise ValueError("No pairwise contrasts to test.")
    combined = pd.concat(frames, ignore_index=True)
    return apply_fdr(combined.drop(columns=["Adjusted_pval", "Threshold"], errors="ignore"))
