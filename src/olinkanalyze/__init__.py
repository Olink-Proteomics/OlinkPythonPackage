"""Olink Analyze for Python: NPX import, QC, normalization, and statistics."""

from olinkanalyze._utils import extract_check_log
from olinkanalyze.data import npx_data1, npx_data2
from olinkanalyze.io import read_NPX, read_npx
from olinkanalyze.normalization import olink_normalization, olink_normalization_bridge
from olinkanalyze.plots import (
    olink_dist_plot,
    olink_pal,
    olink_qc_plot,
    olink_volcano_plot,
    set_plot_theme,
)
from olinkanalyze.qc import check_npx, clean_npx
from olinkanalyze.stats import olink_ttest

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "check_npx",
    "clean_npx",
    "extract_check_log",
    "npx_data1",
    "npx_data2",
    "olink_dist_plot",
    "olink_normalization",
    "olink_normalization_bridge",
    "olink_pal",
    "olink_qc_plot",
    "olink_ttest",
    "olink_volcano_plot",
    "read_NPX",
    "read_npx",
    "set_plot_theme",
]
