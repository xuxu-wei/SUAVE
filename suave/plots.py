"""Plotting helpers for SUAVE."""

from __future__ import annotations

import itertools
import math
import warnings
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _in_notebook() -> bool:
    """Return ``True`` when running inside a Jupyter notebook shell."""

    try:  # pragma: no cover - IPython may be unavailable during tests
        from IPython import get_ipython  # type: ignore
    except Exception:  # pragma: no cover - best-effort detection
        return False

    shell = get_ipython()
    if shell is None:
        return False
    shell_name = shell.__class__.__name__
    return shell_name == "ZMQInteractiveShell"


def _coerce_float(value: float | int | None) -> float:
    """Convert ``value`` to ``float`` while preserving missing entries."""

    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive branch
        return float("nan")


def _display_figure(figure: plt.Figure) -> object | None:
    """Display ``figure`` in a notebook and return a handle for refreshing."""

    try:  # pragma: no cover - IPython optional during tests
        from IPython.display import display
    except Exception:  # pragma: no cover - graceful fallback
        display = None  # type: ignore

    if display is None:
        return None

    try:  # pragma: no cover - depends on notebook backend
        return display(figure, display_id=True)
    except Exception:
        display(figure)
        return None


class TrainingPlotMonitor:
    """Visualise training and validation metrics during ``fit``."""

    def __init__(self, behaviour: str) -> None:
        self._behaviour = behaviour
        self._is_notebook = _in_notebook()
        rows, cols = (2, 3) if behaviour == "supervised" else (1, 3)
        figsize = (cols * 4, rows * 3)
        self._figure, axes = plt.subplots(rows, cols, figsize=figsize)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        self._metrics = self._metric_configuration()
        self._config_by_name = {metric["name"]: metric for metric in self._metrics}
        self._coefficients: dict[str, float | None] = {
            "beta": None,
            "classification_loss_weight": None,
        }
        self._axes: dict[str, Axes] = {}
        self._lines: dict[str, dict[str, Line2D | None]] = {}
        self._history: dict[str, dict[str, list[float]]] = {}

        for axis, metric in zip(axes, self._metrics):
            axis.set_title(self._format_metric_title(metric["name"]))
            axis.set_xlabel("Epoch")
            axis.set_ylabel(metric["ylabel"])

            train_line = None
            val_line = None
            handles = []
            labels = []
            if metric["plot_train"]:
                (train_line,) = axis.plot([], [], label="Train", color="tab:blue")
                handles.append(train_line)
                labels.append("Train")
            if metric["plot_val"]:
                (val_line,) = axis.plot([], [], label="Val", color="tab:orange")
                handles.append(val_line)
                labels.append("Val")
            if handles:
                axis.legend(handles, labels, loc="best")
            else:  # pragma: no cover - defensive fallback
                axis.legend().set_visible(False)

            self._axes[metric["name"]] = axis
            self._lines[metric["name"]] = {"train": train_line, "val": val_line}
            self._history[metric["name"]] = {
                "epoch": [],
                "train": [],
                "val": [],
            }

        for axis in axes[len(self._metrics) :]:
            axis.axis("off")

        self._figure.tight_layout()
        self._display_handle = (
            _display_figure(self._figure) if self._is_notebook else None
        )
        if not self._is_notebook:
            plt.ion()
            self._figure.show()

    def update(
        self,
        *,
        epoch: int,
        train_metrics: Mapping[str, float | int | None] | None = None,
        val_metrics: Mapping[str, float | int | None] | None = None,
        beta: float | None = None,
        classification_loss_weight: float | None = None,
    ) -> None:
        """Append metrics for ``epoch`` and refresh the visualisation."""

        train_metrics = train_metrics or {}
        val_metrics = val_metrics or {}

        if beta is not None and math.isfinite(float(beta)):
            self._coefficients["beta"] = float(beta)
        if classification_loss_weight is not None and math.isfinite(
            float(classification_loss_weight)
        ):
            self._coefficients["classification_loss_weight"] = float(
                classification_loss_weight
            )

        for metric in self._metrics:
            name = metric["name"]
            history = self._history[name]
            history["epoch"].append(float(epoch))
            train_value = self._transform_metric_value(
                name, train_metrics.get(name)
            )
            val_value = self._transform_metric_value(name, val_metrics.get(name))
            history["train"].append(_coerce_float(train_value))
            history["val"].append(_coerce_float(val_value))

            lines = self._lines[name]
            axis = self._axes[name]

            if lines["train"] is not None:
                lines["train"].set_data(history["epoch"], history["train"])
            if lines["val"] is not None:
                lines["val"].set_data(history["epoch"], history["val"])

            axis.relim()
            axis.autoscale_view()

            axis.set_title(self._format_metric_title(name))

        self._figure.tight_layout()
        self._refresh()

    def _refresh(self) -> None:
        """Redraw the figure to reflect the latest metric values."""

        if self._is_notebook and self._display_handle is not None:
            try:  # pragma: no cover - display handle is notebook-specific
                self._display_handle.update(self._figure)
                return
            except Exception:
                pass  # Fall back to clearing the output below

        if self._is_notebook:
            try:  # pragma: no cover - notebook specific
                from IPython.display import clear_output, display

                clear_output(wait=True)
                display(self._figure)
            except Exception:
                self._figure.canvas.draw_idle()
                self._figure.canvas.flush_events()
        else:
            self._figure.canvas.draw_idle()
            self._figure.canvas.flush_events()
            plt.pause(0.001)

    def _metric_configuration(self) -> list[dict[str, object]]:
        """Return plot configuration based on the model behaviour."""

        if self._behaviour == "supervised":
            return [
                {
                    "name": "reconstruction",
                    "title": "Reconstruction",
                    "ylabel": "Value",
                    "plot_train": True,
                    "plot_val": True,
                },
                {
                    "name": "kl",
                    "title": "KL Divergence",
                    "ylabel": "Value",
                    "plot_train": True,
                    "plot_val": True,
                },
                {
                    "name": "total_loss",
                    "title": "ELBO",
                    "ylabel": "Loss",
                    "plot_train": True,
                    "plot_val": True,
                },
                {
                    "name": "auroc",
                    "title": "AUROC",
                    "ylabel": "AUROC",
                    "plot_train": True,
                    "plot_val": True,
                },
                {
                    "name": "classification_loss",
                    "title": "Classification Loss",
                    "ylabel": "Loss",
                    "plot_train": True,
                    "plot_val": True,
                },
                {
                    "name": "joint_objective",
                    "title": "Joint Objective",
                    "ylabel": "Loss",
                    "plot_train": True,
                    "plot_val": True,
                },
            ]

        return [
            {
                "name": "reconstruction",
                "title": "Reconstruction",
                "ylabel": "Value",
                "plot_train": True,
                "plot_val": True,
            },
            {
                "name": "kl",
                "title": "KL Divergence",
                "ylabel": "Value",
                "plot_train": True,
                "plot_val": True,
            },
            {
                "name": "total_loss",
                "title": "ELBO",
                "ylabel": "Loss",
                "plot_train": True,
                "plot_val": True,
            },
        ]

    def _format_metric_title(self, metric_name: str) -> str:
        config = self._config_by_name.get(metric_name, {})
        base_title = str(config.get("title", metric_name))
        if metric_name == "total_loss":
            return self._format_elbo_title(base_title)
        if metric_name == "joint_objective":
            return self._format_joint_title(base_title)
        return base_title

    def _format_elbo_title(self, base_title: str) -> str:
        beta = self._coefficients.get("beta")
        if beta is None or not math.isfinite(beta):
            formula = "reconstruction + KL"
        else:
            formatted = self._format_coefficient(beta)
            if formatted is None:
                formula = "reconstruction + KL"
            else:
                formula = f"reconstruction + {formatted}×KL"
        return f"{base_title}\n({formula})"

    def _format_joint_title(self, base_title: str) -> str:
        weight = self._coefficients.get("classification_loss_weight")
        if weight is None or not math.isfinite(weight):
            formula = "ELBO + Classification Loss"
        else:
            formatted = self._format_coefficient(weight)
            if formatted is None:
                formula = "ELBO + Classification Loss"
            else:
                formula = f"ELBO + {formatted}×Classification Loss"
        return f"{base_title}\n({formula})"

    @staticmethod
    def _format_coefficient(value: float) -> str | None:
        rounded = round(float(value), 1)
        if math.isclose(rounded, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            return None
        if math.isclose(rounded, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            return "0"
        text = f"{rounded:.1f}".rstrip("0").rstrip(".")
        return text

    @staticmethod
    def _transform_metric_value(
        metric_name: str, value: float | int | None
    ) -> float | int | None:
        if value is None:
            return None
        if metric_name != "reconstruction":
            return value
        try:
            numeric = float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive conversion
            return value
        if math.isnan(numeric):
            return float("nan")
        return -numeric

      
def plot_reliability_curve(
    probabilities: Iterable[float],
    targets: Iterable[int],
    output_path: str | Path | None = None,
) -> Path | None:
    """Create an empty reliability curve as a placeholder."""

    fig, ax = plt.subplots()
    ax.set_title("Reliability Curve (placeholder)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.plot([0, 1], [0, 1], linestyle="--", color="tab:blue")

    if output_path is None:
        plt.close(fig)
        return None

    output_path = Path(output_path)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_matrix_heatmap(
    matrix: pd.DataFrame | np.ndarray,
    *,
    ax: Axes | None = None,
    cmap: str = "RdBu_r",
    annotate: bool = False,
    value_format: str = ".2f",
    colorbar: bool = True,
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    log_scale: bool = False,
) -> Axes:
    """Draw a heatmap for ``matrix`` with optional annotations.

    Parameters
    ----------
    matrix : pandas.DataFrame or numpy.ndarray
        Two-dimensional matrix whose values are rendered as colours.
        ``pandas.DataFrame`` inputs expose their index and columns as axis
        labels. ``numpy.ndarray`` inputs default to integer labels.
    ax : matplotlib.axes.Axes, optional
        Axis on which to render the heatmap. When omitted a new axis is
        created via :func:`matplotlib.pyplot.subplots`.
    cmap : str, default "RdBu_r"
        Name of the Matplotlib colormap used to convert values into colours.
    annotate : bool, default False
        When ``True`` the numeric value of each cell is written at the centre
        of the corresponding square.
    value_format : str, default ".2f"
        Format string applied when :paramref:`annotate` is ``True``.
    colorbar : bool, default True
        Whether to attach a colorbar to the provided axis.
    colorbar_label : str, optional
        Text displayed beneath the colorbar when :paramref:`colorbar` is
        enabled.
    vmin, vmax : float, optional
        Explicit colour scale bounds. When omitted the limits are inferred
        from the observed data.
    log_scale : bool, default False
        If ``True`` the colour scale is logarithmic using
        :class:`matplotlib.colors.LogNorm`.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the rendered heatmap. The returned axis is always
        valid regardless of whether :paramref:`ax` was provided.

    Examples
    --------
    >>> import numpy as np
    >>> from suave.plots import plot_matrix_heatmap
    >>> values = np.array([[1.0, -0.5], [0.25, 0.75]])
    >>> ax = plot_matrix_heatmap(values)
    >>> ax.figure is not None
    True
    """

    if not isinstance(matrix, pd.DataFrame):
        matrix = pd.DataFrame(matrix)

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    if ax is None:
        _, ax = plt.subplots(
            figsize=(max(4.0, matrix.shape[1] * 0.6), max(3.0, matrix.shape[0] * 0.6))
        )

    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.astype(str), rotation=45, ha="right")
    ax.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.astype(str))

    if annotate:
        array = matrix.to_numpy(dtype=float)
        for row_index, col_index in itertools.product(
            range(array.shape[0]), range(array.shape[1])
        ):
            value = array[row_index, col_index]
            if not np.isfinite(value):
                continue
            ax.text(
                col_index,
                row_index,
                format(value, value_format),
                ha="center",
                va="center",
                fontsize=9,
            )

    if colorbar:
        cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    return ax


def plot_bubble_matrix(
    size_matrix: pd.DataFrame | np.ndarray,
    *,
    color_matrix: pd.DataFrame | np.ndarray | None = None,
    text_matrix: pd.DataFrame | np.ndarray | None = None,
    ax: Axes | None = None,
    cmap: str = "viridis",
    colorbar_label: str | None = None,
    size_scale: tuple[float, float] = (30.0, 600.0),
    text_format: str = ".2f",
    size_label: str = "|value|",
) -> Axes:
    """Render a bubble chart representing two matrices simultaneously.

    Parameters
    ----------
    size_matrix : pandas.DataFrame or numpy.ndarray
        Matrix whose absolute values determine each bubble's diameter.
        ``pandas.DataFrame`` inputs expose their index and columns as axis
        labels. ``numpy.ndarray`` inputs default to integer labels.
    color_matrix : pandas.DataFrame or numpy.ndarray, optional
        Matrix used to colour the bubbles. When omitted ``size_matrix`` is
        reused. Shapes must match ``size_matrix``.
    text_matrix : pandas.DataFrame or numpy.ndarray, optional
        Matrix whose values are written at the centre of each bubble. When
        omitted no text annotations are drawn.
    ax : matplotlib.axes.Axes, optional
        Axis on which to draw the chart. When ``None`` a new axis is created.
    cmap : str, default "viridis"
        Name of the Matplotlib colormap used to colour the bubbles.
    colorbar_label : str, optional
        Text displayed next to the colour legend.
    size_scale : tuple of float, default (30.0, 600.0)
        Minimum and maximum area allocated to the bubbles. Values are passed
        directly to the ``s`` argument of :meth:`matplotlib.axes.Axes.scatter`.
    text_format : str, default ".2f"
        Format string applied to :paramref:`text_matrix` values.
    size_label : str, default "|value|"
        Legend title describing the meaning of the bubble sizes.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the rendered bubble chart.

    Examples
    --------
    >>> import numpy as np
    >>> from suave.plots import plot_bubble_matrix
    >>> values = np.array([[0.1, 0.8], [0.4, 0.6]])
    >>> ax = plot_bubble_matrix(values, text_matrix=values)
    >>> len(ax.collections) > 0
    True
    """

    if not isinstance(size_matrix, pd.DataFrame):
        size_matrix = pd.DataFrame(size_matrix)

    if size_matrix.ndim != 2:
        raise ValueError("size_matrix must be two-dimensional")

    if color_matrix is None:
        color_matrix = size_matrix
    elif not isinstance(color_matrix, pd.DataFrame):
        color_matrix = pd.DataFrame(
            color_matrix, index=size_matrix.index, columns=size_matrix.columns
        )
    else:
        color_matrix = color_matrix.reindex(
            index=size_matrix.index, columns=size_matrix.columns
        )

    if text_matrix is not None:
        if not isinstance(text_matrix, pd.DataFrame):
            text_matrix = pd.DataFrame(
                text_matrix, index=size_matrix.index, columns=size_matrix.columns
            )
        else:
            text_matrix = text_matrix.reindex(
                index=size_matrix.index, columns=size_matrix.columns
            )

    if ax is None:
        _, ax = plt.subplots(
            figsize=(max(4.0, size_matrix.shape[1] * 0.8), max(3.0, size_matrix.shape[0] * 0.8))
        )

    columns = size_matrix.columns.astype(str)
    rows = size_matrix.index.astype(str)
    x_positions = np.arange(len(columns))
    y_positions = np.arange(len(rows))
    grid_x, grid_y = np.meshgrid(x_positions, y_positions)

    raw_sizes = np.abs(size_matrix.to_numpy(dtype=float)).flatten()
    scaled_sizes = np.zeros_like(raw_sizes)
    finite_mask = np.isfinite(raw_sizes)
    size_min = float(np.nanmin(raw_sizes)) if finite_mask.any() else 0.0
    size_max = float(np.nanmax(raw_sizes)) if finite_mask.any() else 0.0

    def _scale(values: np.ndarray) -> np.ndarray:
        minimum, maximum = size_scale
        if size_max <= size_min:
            return np.full_like(values, maximum)
        normalised = (values - size_min) / (size_max - size_min)
        return normalised * (maximum - minimum) + minimum

    if finite_mask.any():
        scaled_sizes[finite_mask] = _scale(raw_sizes[finite_mask])

    color_values = color_matrix.to_numpy(dtype=float)
    flat_colors = color_values.flatten()
    color_mask = np.isfinite(flat_colors)
    if color_mask.any():
        vmin = float(np.nanmin(flat_colors))
        vmax = float(np.nanmax(flat_colors))
        if np.isclose(vmin, vmax):
            norm = None
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    scatter = ax.scatter(
        grid_x.flatten(),
        grid_y.flatten(),
        s=scaled_sizes,
        c=flat_colors,
        cmap=cmap,
        norm=norm,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(rows)
    ax.set_xlim(-0.5, len(columns) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_aspect("equal")

    if text_matrix is not None:
        values = text_matrix.to_numpy(dtype=float)
        colormap = plt.get_cmap(cmap)
        if norm is None:
            rgba = colormap(0.5)
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            colour = "white" if luminance < 0.5 else "black"
            for row_index, col_index in itertools.product(
                range(values.shape[0]), range(values.shape[1])
            ):
                value = values[row_index, col_index]
                if not np.isfinite(value):
                    continue
                ax.text(
                    col_index,
                    row_index,
                    format(value, text_format),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=colour,
                )
        else:
            face_colours = colormap(norm(color_values))
            luminance = (
                0.299 * face_colours[..., 0]
                + 0.587 * face_colours[..., 1]
                + 0.114 * face_colours[..., 2]
            )
            text_colours = np.where(luminance < 0.5, "white", "black")
            for row_index, col_index in itertools.product(
                range(values.shape[0]), range(values.shape[1])
            ):
                value = values[row_index, col_index]
                if not np.isfinite(value):
                    continue
                ax.text(
                    col_index,
                    row_index,
                    format(value, text_format),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_colours[row_index, col_index],
                )

    cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    legend_values = np.linspace(size_min, size_max, num=3)
    legend_sizes = _scale(legend_values)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="gray",
            markersize=math.sqrt(size / math.pi) if size > 0 else 0.0,
        )
        for size in legend_sizes
    ]
    labels = [f"{value:.2f}" for value in legend_values]
    ax.legend(handles, labels, title=size_label, loc="upper right", frameon=False)

    return ax


def create_centered_colormap(
    cmap_name: str = "RdYlBu_r",
    *,
    vmin: float = -1.0,
    vmax: float = 1.0,
    midpoint: float = 0.0,
) -> LinearSegmentedColormap:
    """Return a colormap centred around ``midpoint``.

    Parameters
    ----------
    cmap_name : str, default "RdYlBu_r"
        Name of the base Matplotlib colormap to transform.
    vmin, vmax : float, default -1.0 and 1.0
        Bounds of the numeric range the colormap will cover.
    midpoint : float, default 0.0
        Value that should be mapped to the central colour of ``cmap_name``.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Customised colormap spanning the requested numeric range.

    Examples
    --------
    >>> from suave.plots import create_centered_colormap
    >>> cmap = create_centered_colormap("coolwarm", vmin=-2, vmax=2, midpoint=0)
    >>> isinstance(cmap, LinearSegmentedColormap)
    True
    """

    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin")

    if midpoint <= vmin or midpoint >= vmax:
        return plt.get_cmap(cmap_name)

    base = plt.get_cmap(cmap_name)
    normaliser = TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=vmax)
    sample_values = np.linspace(vmin, vmax, 256)
    colours = base(normaliser(sample_values))
    return LinearSegmentedColormap.from_list(f"{cmap_name}_centred", colours)


def plot_feature_latent_correlation(
    model,
    X: pd.DataFrame,
    *,
    targets: pd.Series | pd.DataFrame | np.ndarray | Mapping[str, Sequence[object]] | None = None,
    target_name: str = "target",
    variables: Sequence[str] | None = None,
    latent_indices: Sequence[int] | None = None,
    method: str = "spearman",
    p_adjust: str | None = "fdr_bh",
    include_corr_heatmap: bool = False,
    include_pvalue_heatmap: bool = False,
    title: str | None = None,
    output_path: str | Path | None = None,
    output_formats: Sequence[str] | str = ("png",),
    max_dimension: int = 50,
    latents: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Visualise correlations between features (and targets) and latent codes.

    The helper encodes ``X`` with :meth:`suave.model.SUAVE.encode` to obtain the
    posterior mean of the latent representation. The expectation of the latent
    posterior is therefore used as the summary statistic when measuring
    correlation. Users requiring Monte Carlo summaries can supply their own
    ``latents`` array drawn from repeated calls to :meth:`~suave.model.SUAVE.sample`
    or custom sampling logic.

    Parameters
    ----------
    model : suave.model.SUAVE or compatible object
        Fitted estimator providing an :meth:`encode` method returning latent
        representations of ``X``.
    X : pandas.DataFrame
        Feature matrix aligned with the schema used during training.
    targets : pandas.Series, pandas.DataFrame, numpy.ndarray or mapping, optional
        Targets or predictions added as extra rows in the correlation matrix.
        When a one-dimensional structure is supplied its column label defaults
        to ``target_name``.
    target_name : str, default "target"
        Name assigned to :paramref:`targets` when it is a one-dimensional
        sequence without an explicit label.
    variables : sequence of str, optional
        Subset of feature/target columns to include. ``None`` keeps every
        column from ``X`` and :paramref:`targets`.
    latent_indices : sequence of int, optional
        Indices of the latent dimensions to analyse. ``None`` uses every
        dimension returned by :meth:`encode`.
    method : {"spearman", "pearson", "kendall"}, default "spearman"
        Correlation coefficient applied pairwise between features and latent
        variables.
    p_adjust : {"fdr_bh", "bonferroni", "holm", None}, default "fdr_bh"
        Multiplicity correction applied to the correlation p-values. ``None``
        disables any adjustment.
    include_corr_heatmap : bool, default False
        When ``True`` the function prepends a correlation coefficient heatmap to
        the returned figure.
    include_pvalue_heatmap : bool, default False
        When ``True`` the function adds a p-value heatmap before the bubble
        chart. Both flags can be combined to recreate the original
        three-panel layout.
    title : str, optional
        Figure title displayed above the selected panel layout.
    output_path : str or pathlib.Path, optional
        File path used to persist the resulting figure. When supplied the
        directory is created automatically.
    output_formats : sequence of str or str, default ("png",)
        File formats written when :paramref:`output_path` is provided. A single
        string is treated as a singleton list. Extensions must be supported by
        :meth:`matplotlib.figure.Figure.savefig`.
    max_dimension : int, default 50
        Guard rail applied when :paramref:`variables` and
        :paramref:`latent_indices` are left unspecified. Larger matrices are
        truncated to ``max_dimension`` in both directions with a warning.
    latents : numpy.ndarray, optional
        Pre-computed latent representations. When omitted ``model.encode`` is
        invoked which returns the posterior means of the latent distribution.

    Returns
    -------
    tuple
        ``(figure, axes, correlations, p_values)`` containing the Matplotlib
        figure, the array of axes for further customisation and the numeric
        correlation/p-value matrices. The bubble chart axis is always the last
        entry in ``axes``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> class DummyModel:
    ...     def encode(self, frame):
    ...         return np.column_stack([frame["x"], frame["y"]])
    >>> frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    >>> fig, axes, corr, pvals = plot_feature_latent_correlation(DummyModel(), frame)
    >>> corr.shape
    (2, 2)
    >>> fig, axes, corr, pvals = plot_feature_latent_correlation(
    ...     DummyModel(),
    ...     frame,
    ...     include_corr_heatmap=True,
    ...     include_pvalue_heatmap=True,
    ... )
    >>> len(axes)
    3

    See Also
    --------
    plot_matrix_heatmap : Low-level heatmap rendering utility.
    plot_bubble_matrix : Bubble-chart representation for matrices.
    create_centered_colormap : Helper to centre diverging colour maps.
    """

    if latents is None:
        if not hasattr(model, "encode"):
            raise AttributeError(
                "model must define an 'encode' method or latents must be provided"
            )
        latents = model.encode(X)
    latent_array = np.asarray(latents, dtype=float)
    if latent_array.ndim != 2:
        raise ValueError("latents must be a two-dimensional array")
    if latent_array.shape[0] != len(X):
        raise ValueError("Number of latent rows must match len(X)")

    if latent_indices is None:
        latent_labels = [f"z{i}" for i in range(latent_array.shape[1])]
        if latent_array.shape[1] > max_dimension:
            warnings.warn(
                "Latent dimensionality exceeds max_dimension; only the first "
                f"{max_dimension} dimensions will be displayed.",
                UserWarning,
                stacklevel=2,
            )
            latent_array = latent_array[:, :max_dimension]
            latent_labels = latent_labels[:max_dimension]
    else:
        latent_indices = [int(index) for index in latent_indices]
        if any(index < 0 or index >= latent_array.shape[1] for index in latent_indices):
            raise IndexError("latent_indices must reference valid latent dimensions")
        latent_array = latent_array[:, latent_indices]
        latent_labels = [f"z{i}" for i in latent_indices]

    feature_frame = X.copy()

    if targets is not None:
        target_frame: pd.DataFrame
        if isinstance(targets, pd.Series):
            name = targets.name or target_name
            target_frame = targets.to_frame(name=name)
        elif isinstance(targets, pd.DataFrame):
            target_frame = targets.copy()
        elif isinstance(targets, np.ndarray):
            if targets.ndim == 1:
                target_frame = pd.DataFrame({target_name: targets})
            elif targets.ndim == 2:
                columns = [f"{target_name}_{idx}" for idx in range(targets.shape[1])]
                target_frame = pd.DataFrame(targets, columns=columns)
            else:
                raise ValueError("targets array must be one- or two-dimensional")
        elif isinstance(targets, Mapping):
            target_frame = pd.DataFrame(targets)
        else:
            target_frame = pd.DataFrame({target_name: list(targets)})

        target_frame = target_frame.copy()
        if len(target_frame) != len(feature_frame):
            raise ValueError("targets must align with X by row")
        target_frame = target_frame.reset_index(drop=True)
        target_frame.index = feature_frame.index
        feature_frame = pd.concat([feature_frame, target_frame], axis=1)

    if variables is None:
        selected_columns = list(feature_frame.columns)
        if len(selected_columns) > max_dimension:
            warnings.warn(
                "Number of variables exceeds max_dimension; only the first "
                f"{max_dimension} variables will be displayed.",
                UserWarning,
                stacklevel=2,
            )
            selected_columns = selected_columns[:max_dimension]
    else:
        selected_columns = []
        for column in variables:
            if column not in feature_frame.columns:
                raise KeyError(f"Column '{column}' is not present in the supplied data")
            if column not in selected_columns:
                selected_columns.append(column)
    feature_frame = feature_frame.loc[:, selected_columns]

    numeric_frame = pd.DataFrame(index=feature_frame.index)
    for column in feature_frame.columns:
        series = feature_frame[column]
        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.isna().all():
            try:
                codes, _ = pd.factorize(series, na_sentinel=-1)
            except TypeError:  # pandas >= 2.1 uses use_na_sentinel instead
                codes, _ = pd.factorize(series, use_na_sentinel=True)
            numeric_series = pd.Series(codes.astype(float), index=series.index)
            numeric_series.replace(-1, np.nan, inplace=True)
        numeric_frame[column] = numeric_series.astype(float)

    feature_array = numeric_frame.to_numpy(dtype=float)
    if feature_array.shape[0] != latent_array.shape[0]:
        raise ValueError("Features and latents must share the same number of rows")

    n_features = feature_array.shape[1]
    n_latents = latent_array.shape[1]

    if method not in {"spearman", "pearson", "kendall"}:
        raise ValueError("method must be one of {'spearman', 'pearson', 'kendall'}")

    if n_features == 0 or n_latents == 0:
        raise ValueError("At least one feature and one latent dimension are required")

    if method == "spearman":
        corr_matrix, p_matrix = stats.spearmanr(
            feature_array, latent_array, axis=0, nan_policy="omit"
        )
        corr_matrix = np.asarray(corr_matrix)[:n_features, n_features : n_features + n_latents]
        p_matrix = np.asarray(p_matrix)[:n_features, n_features : n_features + n_latents]
    else:
        corr_matrix = np.full((n_features, n_latents), np.nan, dtype=float)
        p_matrix = np.full_like(corr_matrix, np.nan)
        for feat_index in range(n_features):
            feature_values = feature_array[:, feat_index]
            for latent_index in range(n_latents):
                latent_values = latent_array[:, latent_index]
                mask = np.isfinite(feature_values) & np.isfinite(latent_values)
                if mask.sum() < 3:
                    continue
                if method == "pearson":
                    corr, p_value = stats.pearsonr(
                        feature_values[mask], latent_values[mask]
                    )
                else:
                    corr, p_value = stats.kendalltau(
                        feature_values[mask], latent_values[mask]
                    )
                corr_matrix[feat_index, latent_index] = corr
                p_matrix[feat_index, latent_index] = p_value

    corr_df = pd.DataFrame(corr_matrix, index=selected_columns, columns=latent_labels)
    pval_df = pd.DataFrame(p_matrix, index=selected_columns, columns=latent_labels)

    if p_adjust is not None:
        method_lower = p_adjust.lower()
        if method_lower not in {"fdr_bh", "bonferroni", "holm"}:
            raise ValueError(
                "p_adjust must be one of {'fdr_bh', 'bonferroni', 'holm', None}"
            )
        pval_df = _adjust_p_values(pval_df, method=method_lower)

    panel_order: list[str] = []
    if include_corr_heatmap:
        panel_order.append("corr")
    if include_pvalue_heatmap:
        panel_order.append("pval")
    panel_order.append("bubble")

    n_panels = len(panel_order)
    base_width = max(12.0, 3.5 * n_latents)
    width = max(4.0 * n_panels, base_width * (n_panels / 3))
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(width, max(5.0, 1.2 * n_features)),
        constrained_layout=True,
    )
    if isinstance(axes, Axes):
        axes_array = np.array([axes], dtype=object)
    else:
        axes_array = np.asarray(axes, dtype=object)

    heatmap_cmap = create_centered_colormap(
        "RdBu_r", vmin=-1.0, vmax=1.0, midpoint=0.0
    )
    pvalue_cmap = "magma_r"
    pval_values = pval_df.to_numpy(dtype=float)
    vmax_p = float(np.nanmax(pval_values)) if np.isfinite(pval_values).any() else 1.0

    for axis, panel in zip(axes_array, panel_order):
        if panel == "corr":
            plot_matrix_heatmap(
                corr_df,
                ax=axis,
                cmap=heatmap_cmap,
                annotate=False,
                colorbar_label=f"{method.title()} correlation",
                vmin=-1.0,
                vmax=1.0,
            )
            axis.set_title("Correlation coefficients")
        elif panel == "pval":
            plot_matrix_heatmap(
                pval_df,
                ax=axis,
                cmap=pvalue_cmap,
                annotate=False,
                colorbar_label="Adjusted p-value" if p_adjust else "P-value",
                vmin=0.0,
                vmax=min(1.0, vmax_p),
            )
            axis.set_title("P-values")
        else:
            plot_bubble_matrix(
                corr_df.abs(),
                color_matrix=pval_df,
                text_matrix=corr_df,
                ax=axis,
                cmap=pvalue_cmap,
                colorbar_label="Adjusted p-value" if p_adjust else "P-value",
                size_label=f"|{method.title()}|",
            )
            axis.set_title("Correlation vs. significance")

    axes = axes_array

    if title:
        fig.suptitle(title)

    if output_path is not None:
        formats = [output_formats] if isinstance(output_formats, str) else list(output_formats)
        if not formats:
            formats = ["png"]
        base_path = Path(output_path)
        if base_path.suffix:
            base_stem = base_path.with_suffix("")
        else:
            base_stem = base_path
        base_stem.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            fig.savefig(base_stem.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")

    return fig, axes, corr_df, pval_df


def _adjust_p_values(p_values: pd.DataFrame, *, method: str) -> pd.DataFrame:
    matrix = p_values.to_numpy(dtype=float)
    adjusted = np.full_like(matrix, np.nan)
    mask = np.isfinite(matrix)
    values = matrix[mask]
    if values.size == 0:
        return p_values

    # ``multipletests`` preserves the order of the supplied p-values so we can
    # simply reshape the corrected results back into the original matrix.
    _, corrected, _, _ = multipletests(values, method=method)
    adjusted[mask] = corrected
    return pd.DataFrame(adjusted, index=p_values.index, columns=p_values.columns)
