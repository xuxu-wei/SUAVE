import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from suave.plots import (
    TrainingPlotMonitor,
    compute_feature_latent_correlation,
    plot_bubble_matrix,
    plot_feature_latent_correlation_bubble,
    plot_feature_latent_correlation_heatmap,
    plot_matrix_heatmap,
    plot_multilayer_path_graph,
    _adjust_p_values,
)


def test_training_plot_monitor_flips_reconstruction_sign():
    monitor = TrainingPlotMonitor("supervised")

    monitor.update(
        epoch=0,
        train_metrics={"reconstruction": -3.5},
        val_metrics={"reconstruction": -4.25},
    )

    train_history = monitor._history["reconstruction"]["train"]
    val_history = monitor._history["reconstruction"]["val"]

    assert train_history[-1] == pytest.approx(3.5)
    assert val_history[-1] == pytest.approx(4.25)

    plt.close(monitor._figure)


def test_plot_matrix_heatmap_returns_axes():
    matrix = np.array([[1.0, -0.5], [0.25, 0.75]])
    ax = plot_matrix_heatmap(matrix, annotate=True)
    assert ax.figure is not None
    plt.close(ax.figure)


def test_plot_bubble_matrix_supports_dataframe_input():
    matrix = pd.DataFrame(
        [[0.1, 0.6], [0.3, 0.9]], index=["a", "b"], columns=["x", "y"]
    )
    ax = plot_bubble_matrix(matrix, text_matrix=matrix)
    assert len(ax.collections) > 0
    plt.close(ax.figure)


def test_compute_feature_latent_correlation_and_bubble(tmp_path):
    class DummyModel:
        def encode(self, frame: pd.DataFrame) -> np.ndarray:
            return np.column_stack([frame["x"], frame["y"]])

    X = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [3.0, 2.0, 1.0, 0.0],
            "cat": ["a", "b", "a", "b"],
        }
    )
    targets = pd.Series([0, 1, 1, 0], name="label")

    output_base = tmp_path / "correlation" / "latent_feature"
    corr, pvals = compute_feature_latent_correlation(
        DummyModel(),
        X,
        targets=targets,
        latent_indices=[0, 1],
        method="pearson",
        p_adjust="bonferroni",
        max_dimension=50,
    )

    assert corr.loc["x", "z0"] == pytest.approx(1.0)
    assert corr.loc["y", "z1"] == pytest.approx(1.0)
    assert "cat" in corr.index
    assert pvals.shape == corr.shape

    fig, ax = plot_feature_latent_correlation_bubble(
        DummyModel(),
        X,
        targets=targets,
        latent_indices=[0, 1],
        method="pearson",
        p_adjust="bonferroni",
        title="Example",
        output_path=output_base,
        output_formats=["png", "pdf"],
        correlations=corr,
        p_values=pvals,
    )

    assert ax.get_title() == "Example"
    assert fig.axes[0] is ax

    for extension in (".png", ".pdf"):
        assert output_base.with_suffix(extension).exists()

    plt.close(fig)


def test_compute_feature_latent_correlation_guardrail_truncates():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.standard_normal((8, 60)),
        columns=[f"f{i}" for i in range(60)],
    )
    latents = rng.standard_normal((8, 60))

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        corr, pvals = compute_feature_latent_correlation(
            None,
            X,
            latents=latents,
        )

    assert len(captured) == 2
    assert "Latent dimensionality exceeds" in str(captured[0].message)
    assert corr.shape == (50, 50)
    assert pvals.shape == (50, 50)


def test_plot_feature_latent_correlation_bubble_default_title():
    class DummyModel:
        def encode(self, frame: pd.DataFrame) -> np.ndarray:
            return np.column_stack([frame["x"], frame["y"]])

    X = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})

    fig, ax = plot_feature_latent_correlation_bubble(DummyModel(), X)

    assert ax.get_title() == "Spearman correlation vs. adjusted p-values"
    assert fig.axes[0] is ax

    plt.close(fig)


def test_plot_feature_latent_correlation_heatmap_switch():
    class DummyModel:
        def encode(self, frame: pd.DataFrame) -> np.ndarray:
            return np.column_stack([frame["x"], frame["y"]])

    X = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})

    corr, pvals = compute_feature_latent_correlation(DummyModel(), X)

    fig_corr, ax_corr = plot_feature_latent_correlation_heatmap(
        DummyModel(),
        X,
        value="correlation",
        correlations=corr,
        p_values=pvals,
    )
    fig_p, ax_p = plot_feature_latent_correlation_heatmap(
        DummyModel(),
        X,
        value="pvalue",
        p_adjust=None,
        correlations=corr,
        p_values=pvals,
    )

    assert "correlation" in ax_corr.get_title().lower()
    assert "p-value" in ax_p.get_title().lower()

    plt.close(fig_corr)
    plt.close(fig_p)


def test_plot_feature_latent_correlation_requires_pair():
    class DummyModel:
        def encode(self, frame: pd.DataFrame) -> np.ndarray:
            return np.column_stack([frame["x"], frame["y"]])

    X = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    corr, pvals = compute_feature_latent_correlation(DummyModel(), X)

    with pytest.raises(ValueError):
        plot_feature_latent_correlation_bubble(
            DummyModel(), X, correlations=corr, p_values=None
        )
    with pytest.raises(ValueError):
        plot_feature_latent_correlation_heatmap(
            DummyModel(), X, correlations=None, p_values=pvals
        )


def test_plot_multilayer_path_graph_warns_and_renders():
    edges = pd.DataFrame(
        {
            "source": ["a", "a", "b"],
            "target": ["b", "b", "c"],
            "weight_edge_size": [0.2, 0.4, 1.2],
            "label": ["beta", "beta", "sig"],
        }
    )
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "label": ["A", "B", "C", "D"],
            "layer": [0, 1, 2, 0],
            "group": ["g1", "g2", "g3", "g1"],
            "color": ["#ff0000", None, None, None],
        }
    )

    with pytest.warns(UserWarning) as captured:
        fig, ax = plot_multilayer_path_graph(
            edges,
            nodes,
            duplicate_edge_action="warn",
            isolated_node_action="warn",
            layer_color_mapping={0: "#aaaaaa", 1: "#bbbbbb", 2: "#cccccc"},
            group_color_mapping={"g1": "#00ffff", "g2": "#004488"},
            edge_size_legend_values=[0.2, 1.2],
        )

    messages = {str(record.message) for record in captured}
    assert any("duplicate edge" in message for message in messages)
    assert any("isolated node" in message for message in messages)
    assert any("weight_edge_color" in message for message in messages)

    arrow_patches = [patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]
    assert len(arrow_patches) == 2

    scatter = ax.collections[0]
    facecolors = scatter.get_facecolors()
    expected = [
        to_rgba("#ff0000"),
        to_rgba("#004488"),
        to_rgba("#cccccc"),
        to_rgba("#00ffff"),
    ]
    np.testing.assert_allclose(facecolors, expected, atol=1e-6)

    plt.close(fig)


def test_plot_multilayer_path_graph_raises_on_isolated_error():
    edges = pd.DataFrame(
        {
            "source": ["x"],
            "target": ["y"],
            "weight_edge_size": [1.0],
            "weight_edge_color": [0.5],
        }
    )
    nodes = pd.DataFrame(
        {
            "id": ["x", "y", "z"],
            "layer": [0, 1, 2],
        }
    )

    with pytest.raises(ValueError):
        plot_multilayer_path_graph(
            edges,
            nodes,
            isolated_node_action="error",
        )



def test_adjust_p_values_matches_statsmodels():
    matrix = pd.DataFrame(
        [[0.001, 0.2, np.nan], [0.04, 0.8, 0.5]],
        columns=["a", "b", "c"],
    )

    for method in ("fdr_bh", "bonferroni", "holm"):
        adjusted = _adjust_p_values(matrix, method=method)
        mask = np.isfinite(matrix.to_numpy())
        reference = np.full_like(matrix.to_numpy(), np.nan, dtype=float)
        _, corrected, _, _ = multipletests(matrix.to_numpy()[mask], method=method)
        reference[mask] = corrected
        np.testing.assert_allclose(adjusted.to_numpy(), reference, rtol=0, atol=1e-10)
