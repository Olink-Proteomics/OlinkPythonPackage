<!-- badges: start -->

[![Python CI](https://github.com/Olink-Proteomics/OlinkRPackage/actions/workflows/python-ci.yaml/badge.svg)](https://github.com/Olink-Proteomics/OlinkRPackage/actions/workflows/python-ci.yaml)
<!-- badges: end -->

# Olink® Analyze (Python)

A Python toolbox for Olink NPX proteomics data: import, QC, bridge
normalization, t-tests, and plots.

## Installation

Create and activate a conda/mamba environment. Install the local package
after the env exists (`--no-deps` cannot go in `environment.yml` because
conda writes `pip:` entries to a requirements file):

```bash
mamba env create -f environment.yml
mamba activate olinkanalyze
pip install -e . --no-deps
pytest
```

`conda env create -f environment.yml` works the same if you use conda
instead of mamba. To place the env in `./.venv`:

```bash
mamba env create -f environment.yml -p ./.venv
mamba activate ./.venv
pip install -e . --no-deps
```

Alternatively, install into an existing Python 3.10+ environment:

```bash
pip install -e ".[dev]"
```

## Usage

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

<img src="figures/example_distplot.png" alt="NPX distribution plot" width="100%" />
<img src="figures/example_qcplot.png" alt="QC IQR vs median plot" width="100%" />
<img src="figures/example_volcanoplot.png" alt="Volcano plot" width="100%" />

`read_npx` supports long-format CSV/TSV, Excel, and Parquet. Wide qPCR
exports, ZIP archives, mixed models, and cross-product normalization are
not implemented yet.

The package includes two example datasets, `npx_data1` and `npx_data2`,
for exploring the API.

## Issues

Please report issues to biostattools@olink.com or use GitHub issues.

## Credits

Olink® Analyze is developed and maintained by the Olink Proteomics Data
Science Team.
