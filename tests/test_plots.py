from matplotlib.figure import Figure

from olinkanalyze import (
    check_npx,
    clean_npx,
    olink_dist_plot,
    olink_qc_plot,
    olink_ttest,
    olink_volcano_plot,
)


def test_dist_and_qc_plots_on_uncleaned_data(data1):
    dist = olink_dist_plot(data1)
    qc = olink_qc_plot(data1)
    assert isinstance(dist, Figure)
    assert isinstance(qc, Figure)


def test_dist_and_qc_plots_return_figures(data1):
    log = check_npx(data1)
    cleaned = clean_npx(data1, check_log=log)
    dist = olink_dist_plot(cleaned)
    qc = olink_qc_plot(cleaned, label_outliers=True)
    assert isinstance(dist, Figure)
    assert isinstance(qc, Figure)
    assert hasattr(qc, "_olink_qc_data")


def test_dist_plot_one_box_per_sample(data1):
    cleaned = clean_npx(data1)
    fig = olink_dist_plot(cleaned)
    panel = str(cleaned["Panel"].dropna().iloc[0]).replace("Olink ", "")
    in_panel = cleaned["Panel"].astype(str).str.replace("Olink ", "", regex=False) == panel
    n_samples = cleaned.loc[in_panel, "SampleID"].nunique()
    assert len(fig.axes[0].patches) == n_samples


def test_dist_plot_legend_is_outside_axes(data1):
    fig = olink_dist_plot(clean_npx(data1))
    assert fig.axes[0].get_legend() is None
    assert len(fig.legends) == 1
    fig.canvas.draw()
    axes_right = max(ax.get_window_extent().x1 for ax in fig.axes)
    legend_box = fig.legends[0].get_window_extent()
    assert legend_box.x0 >= axes_right - 1


def test_volcano_plot_from_ttest(data1):
    cleaned = clean_npx(data1)
    ttest = olink_ttest(cleaned, variable="Treatment")
    fig = olink_volcano_plot(ttest, olinkid_list=ttest["OlinkID"].head(5))
    assert isinstance(fig, Figure)
