import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from suave.plots import (
    TrainingPlotMonitor,
    plot_bubble_matrix,
    plot_feature_latent_correlation,
    plot_matrix_heatmap,
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


def test_plot_feature_latent_correlation_outputs(tmp_path):
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
    fig, axes, corr, pvals = plot_feature_latent_correlation(
        DummyModel(),
        X,
        targets=targets,
        latent_indices=[0, 1],
        method="pearson",
        p_adjust="bonferroni",
        title="Example",
        output_path=output_base,
        output_formats=["png", "pdf"],
        include_corr_heatmap=True,
        include_pvalue_heatmap=True,
    )

    assert corr.loc["x", "z0"] == pytest.approx(1.0)
    assert corr.loc["y", "z1"] == pytest.approx(1.0)
    assert "cat" in corr.index
    assert pvals.shape == corr.shape
    assert len(axes) == 3

    for extension in (".png", ".pdf"):
        assert output_base.with_suffix(extension).exists()

    plt.close(fig)


def test_plot_feature_latent_correlation_guardrail_truncates():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.standard_normal((8, 60)),
        columns=[f"f{i}" for i in range(60)],
    )
    latents = rng.standard_normal((8, 60))

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fig, axes, corr, pvals = plot_feature_latent_correlation(
            None,
            X,
            latents=latents,
        )

    assert len(captured) == 2
    assert "Latent dimensionality exceeds" in str(captured[0].message)
    assert corr.shape == (50, 50)
    assert pvals.shape == (50, 50)
    assert len(axes) == 1

    plt.close(fig)


def test_plot_feature_latent_correlation_default_single_panel():
    class DummyModel:
        def encode(self, frame: pd.DataFrame) -> np.ndarray:
            return np.column_stack([frame["x"], frame["y"]])

    X = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})

    fig, axes, corr, pvals = plot_feature_latent_correlation(DummyModel(), X)

    assert corr.shape[1] == 2
    assert pvals.shape == corr.shape
    assert len(axes) == 1

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
