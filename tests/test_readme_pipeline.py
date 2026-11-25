from olinkanalyze import (
    check_npx,
    clean_npx,
    npx_data1,
    npx_data2,
    olink_dist_plot,
    olink_normalization,
    olink_qc_plot,
    olink_ttest,
    olink_volcano_plot,
    read_npx,
)
from olinkanalyze._constants import BRIDGE_SAMPLE_IDS
from olinkanalyze.data import load_example_csv


def test_readme_pipeline_end_to_end():
    raw = read_npx(load_example_csv())
    log = check_npx(raw)
    # The bundled long CSV fixture labels every row as CONTROL; keep those rows
    # so the import/check/clean path can be exercised on real Olink output.
    csv_cleaned = clean_npx(raw, check_log=log, remove_control_sample=False)
    assert not csv_cleaned.empty

    cleaned = clean_npx(npx_data1)
    dist = olink_dist_plot(cleaned)
    qc = olink_qc_plot(cleaned)
    assert dist.axes
    assert qc.axes

    check1 = check_npx(npx_data1)
    check2 = check_npx(npx_data2)
    bridge = [
        sample
        for sample in BRIDGE_SAMPLE_IDS
        if sample in set(npx_data1["SampleID"]) and sample in set(npx_data2["SampleID"])
    ]
    normalized = olink_normalization(
        df1=npx_data1,
        df2=npx_data2,
        overlapping_samples_df1=bridge,
        df1_project_nr="20200001",
        df2_project_nr="20200002",
        reference_project="20200001",
        df1_check_log=check1,
        df2_check_log=check2,
    )
    assert {"Project", "Adj_factor"} <= set(normalized.columns)

    ttest = olink_ttest(clean_npx(npx_data1), variable="Treatment")
    volcano = olink_volcano_plot(ttest, olinkid_list=ttest["OlinkID"].head(8).tolist())
    assert not ttest.empty
    assert volcano.axes
