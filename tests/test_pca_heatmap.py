from matplotlib.figure import Figure

from olinkanalyze import clean_npx, olink_heatmap_plot, olink_pca_plot


def test_olink_pca_plot_returns_scores(data1):
    fig = olink_pca_plot(clean_npx(data1), color_g="Treatment")
    assert isinstance(fig, Figure)
    assert hasattr(fig, "_olink_pca_scores")
    assert {"PC1", "PC2"} <= set(fig._olink_pca_scores.columns)


def test_olink_pca_plot_legend_is_outside_axes(data1):
    fig = olink_pca_plot(clean_npx(data1), color_g="Treatment")
    assert fig.axes[0].get_legend() is None
    assert len(fig.legends) == 1
    fig.canvas.draw()
    ax_box = fig.axes[0].get_window_extent()
    legend_box = fig.legends[0].get_window_extent()
    assert legend_box.x0 >= ax_box.x1 - 1


def test_olink_heatmap_plot_returns_figure(data1):
    fig = olink_heatmap_plot(clean_npx(data1), n_assays=8)
    assert isinstance(fig, Figure)
