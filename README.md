
<!-- README.md is generated from README.Rmd. Please edit that file -->

<!-- badges: start -->

[![R-CMD-check](https://github.com/Olink-Proteomics/OlinkRPackage/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/Olink-Proteomics/OlinkRPackage/actions/workflows/R-CMD-check.yaml)
[![Python CI](https://github.com/Olink-Proteomics/OlinkRPackage/actions/workflows/python-ci.yaml/badge.svg)](https://github.com/Olink-Proteomics/OlinkRPackage/actions/workflows/python-ci.yaml)
[![lint](https://github.com/Olink-Proteomics/OlinkRPackage/actions/workflows/lint.yaml/badge.svg)](https://github.com/Olink-Proteomics/OlinkRPackage/actions/workflows/lint.yaml)
[![Coverage](https://img.shields.io/badge/Coverage-92%25-green)](https://img.shields.io/badge/Coverage-92%25-green)
[![CRAN
status](https://www.r-pkg.org/badges/version/OlinkAnalyze)](https://CRAN.R-project.org/package=OlinkAnalyze)
[![](https://img.shields.io/badge/DOI-10.32614/CRAN.package.OlinkAnalyze-1f57b6?style=flat&link=https://doi.org/10.32614/CRAN.package.OlinkAnalyze)](https://doi.org/10.32614/CRAN.package.OlinkAnalyze)
<!-- badges: end -->

# Olink® Analyze

The goal of Olink® Analyze is to provide a versatile toolbox to enable
easy and smooth handling of Olink NPX data to speed up your proteomic
research. Olink® Analyze provides functions ranging from reading Olink
NPX data as exported by NPX Manager to various statistical tests and
modelling, via different QC plot functions. Thereby providing a
convenient pipeline for your Olink NPX data analysis.

This repository now also ships a native **Python** package (`olinkanalyze`)
that covers the README analysis pipeline: `read_npx`, `check_npx`,
`clean_npx`, QC plots, bridge normalization, t-test, and volcano plots.
The original R package remains in [`OlinkAnalyze/`](OlinkAnalyze/).

## Python package

Create and activate a named conda/mamba environment. Conda writes `pip:` entries to a requirements file, so `--no-deps` cannot live in `environment.yml`; install the local package after the env exists:

```bash
mamba env create -f environment.yml
mamba activate olinkanalyze
pip install -e . --no-deps
pytest
```

`conda env create -f environment.yml` works the same if you use conda instead of mamba. To place the env in `./.venv` instead of a named env:

```bash
mamba env create -f environment.yml -p ./.venv
mamba activate ./.venv
pip install -e . --no-deps
```

Alternatively, install into an existing Python environment:

```bash
pip install -e ".[dev]"
```

```python
import olinkanalyze as oa

my_npx_data = oa.read_npx("path/to/my_NPX_data.csv")
check_log = oa.check_npx(my_npx_data)
npx_clean = oa.clean_npx(my_npx_data, check_log=check_log)

oa.olink_dist_plot(npx_clean)
oa.olink_qc_plot(npx_clean)

bridge_samples = sorted(
    set(oa.npx_data1["SampleID"]).intersection(oa.npx_data2["SampleID"])
)
bridge_samples = [s for s in bridge_samples if "CONTROL" not in s]

normalized = oa.olink_normalization(
    df1=oa.npx_data1,
    df2=oa.npx_data2,
    overlapping_samples_df1=bridge_samples,
    df1_project_nr="20200001",
    df2_project_nr="20200002",
    reference_project="20200001",
)

ttest_results = oa.olink_ttest(npx_clean, variable="Treatment")
oa.olink_volcano_plot(
    ttest_results,
    olinkid_list=ttest_results["OlinkID"].head(10),
)
```

Long-format CSV/TSV, Excel, and Parquet files are supported. Wide qPCR
exports, ZIP archives, mixed-model statistics, and cross-product
normalization remain R-only in this first Python slice.

## Installation


Olink® Analyze is available on CRAN:
<https://cran.r-project.org/package=OlinkAnalyze>.

``` r
install.packages("OlinkAnalyze")
```

## Vignette

``` r
browseVignettes("OlinkAnalyze")
```

**Note:** Process-specific, practical vignettes for analysis of Olink®
proteomics data hosted on OA have been moved to a newer package Olink®
Analyze Vignettes, that is hosted on CRAN:
<https://cran.r-project.org/package=OlinkAnalyzeVignettes>. Olink®
Analyze includes a master vignette and the core analysis functionality,
and Olink® Analyze Vignettes serves to complement this package.

``` r
install.packages("OlinkAnalyzeVignettes")
```

## Usage

### Reading Olink NPX data

``` r
# open package
library(OlinkAnalyze)

# reading Olink NPX data
my_npx_data <- OlinkAnalyze::read_NPX(
  filename = "path/to/my_NPX_data.xlsx"
)
# OR
my_npx_data <- OlinkAnalyze::read_npx(
  filename = "path/to/my_NPX_data.xlsx"
)
```

### Check and clean NPX data

Olink® Analyze provides several functions to check and clean your NPX
data. Below follows an example of how to check and clean the NPX data
using the package provided `npx_data1` dataset:

``` r
# check NPX data
check_npx_data1 <- OlinkAnalyze::check_npx(
  df = OlinkAnalyze::npx_data1
)

# clean NPX data
npx_data1_clean <- OlinkAnalyze::clean_npx(
  df = OlinkAnalyze::npx_data1,
  check_log = check_npx_data1
)

# re-check cleaned NPX data
check_npx_data1_clean <- OlinkAnalyze::check_npx(
  df = npx_data1_clean
)
```

### QC plot functions

There are several plot functions, below follow two examples using the
package provided `npx_data1` dataset:

``` r
# visualize the NPX distribution per sample per panel, example for one panel
npx_data1_clean |>
  dplyr::filter(
    .data[["Panel"]] == "Olink Cardiometabolic"
  ) |>
  OlinkAnalyze::olink_dist_plot(
    check_log = check_npx_data1_clean
  ) +
  ggplot2::theme(
    axis.text.x = ggplot2::element_blank(),
    axis.ticks.x = ggplot2::element_blank()
  ) +
  ggplot2::scale_fill_manual(
    values = c("turquoise3", "red")
  )
```

<img src="figures/example_distplot.png" alt="" width="100%" />

``` r
# visualize potential outliers by IQR vs. sample median per panel
# example for one panel
npx_data1_clean |>
  dplyr::filter(
    .data[["Panel"]] == "Olink Cardiometabolic"
  ) |>
  OlinkAnalyze::olink_qc_plot(
    check_log = check_npx_data1_clean
  ) +
  ggplot2::scale_color_manual(
    values = c("turquoise3", "red")
  )
```

<img src="figures/example_qcplot.png" alt="" width="100%" />

### Normalization

Olink® Analyze provides several means of normalization when analyzing
multiple datasets. Below follows an example of reference sample (aka
bridge) normalization using the two package provided `npx_data1` and
`npx_data2` datasets:

``` r
# identify bridge samples
bridge_samples <- intersect(
  x = npx_data1[["SampleID"]],
  y = npx_data2[["SampleID"]]
)
# remove control samples
bridge_samples <- bridge_samples[!grepl(
  pattern = "CONTROL",
  x = bridge_samples
)]

# npx_data1 was checked earlier
# we will check only npx_data2 before normalization

check_npx_data2 <- OlinkAnalyze::check_npx(
  df = npx_data2
)

# bridge normalize
bridge_normalized_data <- OlinkAnalyze::olink_normalization(
  df1 = npx_data1,
  df2 = npx_data2,
  overlapping_samples_df1 = bridge_samples,
  df1_project_nr = "20200001",
  df2_project_nr = "20200002",
  reference_project = "20200001",
  df1_check_log = check_npx_data1,
  df2_check_log = check_npx_data2
)
```

### Statistical tests and models

Olink® Analyze provides several statistical tests and model tools. Below
follows an example of how to perform a t-test and how to visualize the
t-test output in a volcano plot using the `npx_data1`:

``` r
# t-test npx_data1
ttest_results_npx1 <- OlinkAnalyze::olink_ttest(
  df = npx_data1_clean,
  check_log = check_npx_data1_clean,
  variable = "Treatment"
)

# select names of the top #10 most significant proteins
ttest_sign_npx1 <- ttest_results_npx1 |>
  dplyr::slice_head(
    n = 10L
  ) |>
  dplyr::pull(
    .data[["OlinkID"]]
  )

# volcano plot with annotated top #10 most significant proteins
OlinkAnalyze::olink_volcano_plot(
  p.val_tbl = ttest_results_npx1,
  olinkid_list = ttest_sign_npx1
) +
  ggplot2::scale_color_manual(
    values = c("turquoise3", "red")
  )
```

<img src="figures/example_volcanoplot.png" alt="" width="100%" />

## Learn more

Please see the function specific help pages. Moreover, Olink® Analyze
includes two simulated NPX datasets for your convenience to help you
explore the package and its functions.

## Issues

Please report any issues (good or bad) to \<biostattools\[a\]olink.com\>
or use the github [issue
function](https://github.com/Olink-Proteomics/OlinkRPackage/issues).

## Alternative install methods

To install directly from the github repository:

``` r
# Package remotes is required: install.packages("remotes")
remotes::install_github(
  repo = "Olink-Proteomics/OlinkRPackage/OlinkAnalyze",
  ref = "main",
  build_vignettes = TRUE
)
```

To install Olink Analyze into a new
[conda](https://docs.conda.io/en/latest/) environment:

``` bash
conda create -n OlinkAnalyze -c conda-forge r-olinkanalyze
```

## Credits

Olink® Analyze is developed and maintained by the Olink Proteomics Data
Science Team.
