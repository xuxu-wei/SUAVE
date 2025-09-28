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
    plot_feature_latent_outcome_path_graph,
    plot_matrix_heatmap,
    plot_multilayer_path_graph,
    plot_multilayer_path_graph_from_graph,
    _adjust_p_values,
)
from suave.types import Schema


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

    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "$-\\log_{10}(p)$"

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

    fig, ax = plot_feature_latent_correlation_bubble(
        DummyModel(),
        X,
        variable_name={"x": "Feature X", "y": "Feature Y"},
    )

    assert ax.get_title() == ""
    assert ax.get_xticklabels()[0].get_text() == "$z_{1}$"
    assert ax.get_yticklabels()[0].get_text() == "Feature X"
    assert fig.axes[0] is ax

    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "$-\\log_{10}(FDR)$"

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
        title="Correlation",
    )
    fig_p, ax_p = plot_feature_latent_correlation_heatmap(
        DummyModel(),
        X,
        value="pvalue",
        p_adjust=None,
        correlations=corr,
        p_values=pvals,
        title="P-values",
    )
    fig_fdr, ax_fdr = plot_feature_latent_correlation_heatmap(
        DummyModel(),
        X,
        value="pvalue",
        p_adjust="fdr_bh",
        correlations=corr,
        p_values=pvals,
        title="FDR",
    )

    assert ax_corr.get_title() == "Correlation"
    assert ax_p.get_title() == "P-values"
    assert ax_corr.get_xticklabels()[0].get_text() == "$z_{1}$"

    assert fig_corr.axes[-1].get_xlabel() == "Spearman correlation"
    assert fig_p.axes[-1].get_xlabel() == "$-\\log_{10}(p)$"
    assert fig_fdr.axes[-1].get_xlabel() == "$-\\log_{10}(FDR)$"

    plt.close(fig_corr)
    plt.close(fig_p)
    plt.close(fig_fdr)


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



def test_plot_feature_latent_outcome_path_graph_labels_and_edges():
    class DemoModel:
        def __init__(self) -> None:
            self.schema = Schema({"age": {"type": "real"}, "sofa": {"type": "real"}})

        def encode(self, frame: pd.DataFrame) -> np.ndarray:
            return np.column_stack([frame["age"], frame["sofa"]])

    X = pd.DataFrame({"sofa": [4.0, 5.0, 6.0, 7.0], "age": [60, 70, 80, 90]})
    y = pd.Series([0, 1, 1, 0], name="mortality")

    fig, ax = plot_feature_latent_outcome_path_graph(
        DemoModel(),
        X,
        y=y,
        significance_level=0.99,
        edge_label_top_k=2,
    )

    arrow_count = sum(isinstance(patch, FancyArrowPatch) for patch in ax.patches)
    assert arrow_count == 6

    rho_labels = [text.get_text() for text in ax.texts if text.get_text().startswith("$\\rho=")]
    assert len(rho_labels) == 2
    assert "$z_{0}$" in {text.get_text() for text in ax.texts}

    plt.close(fig)


def test_plot_feature_latent_outcome_path_graph_respects_custom_mappings():
    class DemoModel:
        def __init__(self) -> None:
            self.schema = Schema({"f1": {"type": "real"}, "f2": {"type": "real"}})

        def encode(self, frame: pd.DataFrame) -> np.ndarray:
            return np.column_stack([frame["f1"], frame["f2"]])

    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [3.0, 2.0, 1.0, 0.0]})
    y = pd.Series([0, 1, 0, 1], name="outcome")

    node_label_mapping = {
        "f1": "Feature one",
        "f2": "Feature two",
        "outcome": "Outcome label",
    }
    node_color_mapping = {
        "f1": "#112233",
        "f2": "#445566",
        "outcome": "#778899",
    }
    node_group_mapping = {"f1": "Vitals", "f2": "Labs", "outcome": "Outcome"}
    group_color_mapping = {
        "Vitals": "#123456",
        "Labs": "#654321",
        "Latent": "#abcdef",
        "Outcome": "#778899",
    }

    fig, ax = plot_feature_latent_outcome_path_graph(
        DemoModel(),
        X,
        y=y,
        node_label_mapping=node_label_mapping,
        node_color_mapping=node_color_mapping,
        node_group_mapping=node_group_mapping,
        group_color_mapping=group_color_mapping,
        edge_label_top_k=0,
    )

    labels = {text.get_text() for text in ax.texts}
    assert {"Feature one", "Feature two", "Outcome label"}.issubset(labels)

    scatter = next(collection for collection in ax.collections if collection.get_offsets().size)
    facecolors = scatter.get_facecolors()
    np.testing.assert_allclose(facecolors[0], to_rgba("#112233"), atol=1e-6)
    np.testing.assert_allclose(facecolors[1], to_rgba("#445566"), atol=1e-6)
    np.testing.assert_allclose(facecolors[-1], to_rgba("#778899"), atol=1e-6)

    plt.close(fig)


def test_plot_feature_latent_outcome_path_graph_warns_without_model_schema():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([1, 0, 1, 0], name="outcome")
    latents = np.column_stack([X["x"], X["x"]])

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fig, ax = plot_feature_latent_outcome_path_graph(
            None,
            X,
            y=y,
            latents=latents,
            significance_level=0.5,
            edge_label_top_k=None,
        )

    assert any("model is None" in str(w.message) for w in captured)
    plt.close(fig)


def test_plot_multilayer_path_graph_from_graph_roundtrip():
    nx = pytest.importorskip("networkx")

    graph = nx.DiGraph()
    graph.add_node("f", layer=0, label="F", group="Feature")
    graph.add_node("z0", layer=1, label="$z_0$", group="Latent")
    graph.add_node("y", layer=2, label="Y", group="Outcome")
    graph.add_edge("f", "z0", weight_edge_size=0.4, weight_edge_color=0.4)
    graph.add_edge("z0", "y", weight_edge_size=0.5, weight_edge_color=0.5)

    fig, ax = plot_multilayer_path_graph_from_graph(graph)
    arrow_count = sum(isinstance(patch, FancyArrowPatch) for patch in ax.patches)
    assert arrow_count == 2

    bad_graph = nx.DiGraph()
    bad_graph.add_node("a", layer=0)
    bad_graph.add_node("b", layer=1)
    bad_graph.add_edge("a", "b")
    with pytest.raises(ValueError):
        plot_multilayer_path_graph_from_graph(bad_graph)

    plt.close(fig)




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
