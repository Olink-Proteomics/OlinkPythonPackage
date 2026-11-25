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
    if pair_id is None:
        LOGGER.info("T-test is performed on %s - %s.", level1, level2)
        rows = [
            _unpaired_row(
                group,
                oid=oid,
                level1=level1,
                level2=level2,
                variable=variable,
                quant_col=quant_col,
                assay_col=assay_col,
                uniprot_col=uniprot_col,
                panel_col=panel_col,
                alternative=alternative,
                confidence_level=confidence_level,
            )
            for oid, group in work.groupby(oid_col, dropna=False, sort=False)
        ]
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
        rows = [
            _paired_row(
                group,
                oid=oid,
                level1=level1,
                level2=level2,
                variable=variable,
                pair_id=pair_id,
                quant_col=quant_col,
                assay_col=assay_col,
                uniprot_col=uniprot_col,
                panel_col=panel_col,
                alternative=alternative,
                confidence_level=confidence_level,
            )
            for oid, group in work.groupby(oid_col, dropna=False, sort=False)
        ]

    result = pd.DataFrame([row for row in rows if row is not None])
    if result.empty:
        raise ValueError("No assays passing initial check. T-test will not be performed.")

    _rejected, p_adj, _alphac_sidak, _alphac_bonf = multipletests(
        result["p.value"].to_numpy(), method="fdr_bh"
    )
    result["Adjusted_pval"] = p_adj
    result["Threshold"] = np.where(result["Adjusted_pval"] < 0.05, "Significant", "Non-significant")
    return result.sort_values("p.value", kind="mergesort").reset_index(drop=True)


def _meta(group: pd.DataFrame, assay_col: str, uniprot_col: str, panel_col: str, oid: object) -> dict:
    return {
        "Assay": group[assay_col].iloc[0],
        "OlinkID": oid,
        "UniProt": group[uniprot_col].iloc[0],
        "Panel": group[panel_col].iloc[0],
    }


def _unpaired_row(
    group: pd.DataFrame,
    *,
    oid: object,
    level1: str,
    level2: str,
    variable: str,
    quant_col: str,
    assay_col: str,
    uniprot_col: str,
    panel_col: str,
    alternative: str,
    confidence_level: float,
) -> dict | None:
    g1 = group.loc[group[variable].astype(str) == str(level1), quant_col].dropna().to_numpy(dtype=float)
    g2 = group.loc[group[variable].astype(str) == str(level2), quant_col].dropna().to_numpy(dtype=float)
    if g1.size < 2 or g2.size < 2:
        return None
    test = stats.ttest_ind(g1, g2, equal_var=False, alternative=alternative)
    ci = test.confidence_interval(confidence_level=confidence_level)
    row = _meta(group, assay_col, uniprot_col, panel_col, oid)
    row.update(
        {
            "estimate": float(np.mean(g1) - np.mean(g2)),
            str(level1): float(np.mean(g1)),
            str(level2): float(np.mean(g2)),
            "statistic": float(test.statistic),
            "p.value": float(test.pvalue),
            "parameter": float(test.df),
            "conf.low": float(ci.low),
            "conf.high": float(ci.high),
            "method": "Welch Two Sample t-test",
            "alternative": alternative,
        }
    )
    return row


def _paired_row(
    group: pd.DataFrame,
    *,
    oid: object,
    level1: str,
    level2: str,
    variable: str,
    pair_id: str,
    quant_col: str,
    assay_col: str,
    uniprot_col: str,
    panel_col: str,
    alternative: str,
    confidence_level: float,
) -> dict | None:
    wide = (
        group.pivot_table(index=pair_id, columns=variable, values=quant_col, aggfunc="mean")
        .reindex(columns=[level1, level2])
        .dropna()
    )
    if len(wide) < 2:
        return None
    x = wide[level1].to_numpy(dtype=float)
    y = wide[level2].to_numpy(dtype=float)
    test = stats.ttest_rel(x, y, alternative=alternative)
    ci = test.confidence_interval(confidence_level=confidence_level)
    row = _meta(group, assay_col, uniprot_col, panel_col, oid)
    row.update(
        {
            "estimate": float(np.mean(x - y)),
            str(level1): float(np.mean(x)),
            str(level2): float(np.mean(y)),
            "statistic": float(test.statistic),
            "p.value": float(test.pvalue),
            "parameter": float(test.df),
            "conf.low": float(ci.low),
            "conf.high": float(ci.high),
            "method": "Paired t-test",
            "alternative": alternative,
        }
    )
    return row
