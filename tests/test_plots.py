from matplotlib.figure import Figure

from olinkanalyze import (
    check_npx,
    clean_npx,
    olink_dist_plot,
    olink_qc_plot,
    olink_ttest,
    olink_volcano_plot,
)


def test_dist_and_qc_plots_return_figures(data1):
    log = check_npx(data1)
    cleaned = clean_npx(data1, check_log=log)
    dist = olink_dist_plot(cleaned)
    qc = olink_qc_plot(cleaned, label_outliers=True)
    assert isinstance(dist, Figure)
    assert isinstance(qc, Figure)
    assert hasattr(qc, "_olink_qc_data")


def test_volcano_plot_from_ttest(data1):
    cleaned = clean_npx(data1)
    ttest = olink_ttest(cleaned, variable="Treatment")
    fig = olink_volcano_plot(ttest, olinkid_list=ttest["OlinkID"].head(5))
    assert isinstance(fig, Figure)
