"""Plotting helpers for SUAVE."""

from __future__ import annotations

import itertools
import math
import warnings
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    LogNorm,
    Normalize,
    TwoSlopeNorm,
    is_color_like,
)
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as BezierPath
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


_ISSUE_ACTIONS = {"ignore", "warn", "error"}

DEFAULT_PHASE_PALETTE: tuple[tuple[str, str], ...] = (
    ("KL annealing", "#e55109"),
    ("VAE warmup", "#ff980e"),
    ("Classification head", "#531f9c"),
    ("Joint fine-tuning", "#b89bd9"),
    ("Decoder refining", "#faddb2"),
)


def _is_fdr_adjustment(method: str | None) -> bool:
    """Return ``True`` when ``method`` corresponds to an FDR correction."""

    if method is None:
        return False
    return method.lower().startswith("fdr")


def _handle_data_issue(action: str, message: str) -> None:
    """Dispatch *message* according to the configured issue *action*."""

    if action not in _ISSUE_ACTIONS:
        raise ValueError(
            "issue action must be one of {'ignore', 'warn', 'error'}, "
            f"got {action!r}"
        )
    if action == "ignore":
        return
    if action == "warn":
        warnings.warn(message, UserWarning, stacklevel=3)
        return
    raise ValueError(message)


def _prepare_phase_palette(
    palette: Mapping[str, str] | Sequence[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    """Validate and normalise the shading palette configuration."""

    if palette is None:
        return []

    if isinstance(palette, Mapping):
        items = list(palette.items())
    else:
        items = list(palette)

    prepared: list[tuple[str, str]] = []
    for name, color in items:
        if color is None:
            continue
        if not is_color_like(color):
            warnings.warn(
                f"Ignoring invalid colour {color!r} for phase {name!r}",
                UserWarning,
                stacklevel=3,
            )
            continue
        prepared.append((str(name), str(color)))
    return prepared


def _resolve_variable_label(
    identifier: object,
    mapping: Mapping[str, object] | Callable[[str], object] | None,
) -> str:
    """Return the display label for ``identifier`` using ``mapping`` when available."""

    key = str(identifier)
    if mapping is None:
        label = key
    elif callable(mapping):
        label = mapping(key)
    else:
        label = mapping.get(key, key)
    if label is None:
        return ""
    return str(label)


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

    def __init__(
        self,
        behaviour: str,
        phase_palette: Mapping[str, str] | Sequence[tuple[str, str]] | None = DEFAULT_PHASE_PALETTE,
    ) -> None:
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
        self._phase_palette = _prepare_phase_palette(phase_palette)
        self._phase_colors = {name: color for name, color in self._phase_palette}
        self._phase_alpha = 0.2
        self._active_phase: str | None = None
        self._active_phase_start: float | None = None
        self._active_phase_last_epoch: float | None = None
        self._active_phase_patches: dict[str, object] = {}
        self._phase_legend = None
        self._tight_layout_rect: tuple[float, float, float, float] | None = None

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

        if self._phase_palette:
            handles = [
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=self._phase_colors[name],
                    edgecolor="none",
                    alpha=self._phase_alpha,
                )
                for name, _ in self._phase_palette
            ]
            labels = [name for name, _ in self._phase_palette]
            self._phase_legend = self._figure.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=max(1, len(handles)),
                frameon=False,
                title="Training phases",
            )
            self._tight_layout_rect = (0.0, 0.08, 1.0, 1.0)

        self._apply_layout()
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
        phase: str | None = None,
    ) -> None:
        """Append metrics for ``epoch`` and refresh the visualisation."""

        train_metrics = train_metrics or {}
        val_metrics = val_metrics or {}

        self._update_phase_shading(float(epoch), phase)

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

        self._apply_layout()
        self._refresh()

    def _apply_layout(self) -> None:
        """Apply tight layout while preserving space for the legend when needed."""

        if self._tight_layout_rect is not None:
            self._figure.tight_layout(rect=self._tight_layout_rect)
        else:
            self._figure.tight_layout()

    def _update_phase_shading(self, epoch: float, phase: str | None) -> None:
        """Synchronise phase shading across all axes."""

        if not self._phase_palette:
            return

        def _clear_active_phase() -> None:
            for patch in self._active_phase_patches.values():
                try:
                    patch.remove()
                except ValueError:
                    pass
            self._active_phase_patches = {}
            self._active_phase = None
            self._active_phase_start = None
            self._active_phase_last_epoch = None

        if phase is None or phase not in self._phase_colors:
            _clear_active_phase()
            return

        if self._active_phase != phase:
            _clear_active_phase()
            self._active_phase = phase
            self._active_phase_start = float(epoch)

        start = (
            self._active_phase_start if self._active_phase_start is not None else float(epoch)
        )
        last_epoch = self._active_phase_last_epoch

        color = self._phase_colors.get(phase)
        if color is None:
            return

        if last_epoch is None:
            span = 1.0
        else:
            span = float(epoch) - float(last_epoch)
            if span <= 0:
                span = 1.0

        start = float(start)
        end = float(epoch) + span
        if end < start:
            end = start

        for metric_name, axis in self._axes.items():
            previous = self._active_phase_patches.get(metric_name)
            if previous is not None:
                previous.remove()
            patch = axis.axvspan(start, end, facecolor=color, alpha=self._phase_alpha, zorder=0)
            self._active_phase_patches[metric_name] = patch
        self._active_phase_last_epoch = float(epoch)

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
    >>> import matplotlib.pyplot as plt
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


def compute_feature_latent_correlation(
    model,
    X: pd.DataFrame,
    *,
    targets: pd.Series | pd.DataFrame | np.ndarray | Mapping[str, Sequence[object]] | None = None,
    target_name: str = "target",
    variables: Sequence[str] | None = None,
    latent_indices: Sequence[int] | None = None,
    method: str = "spearman",
    p_adjust: str | None = "fdr_bh",
    max_dimension: int = 50,
    latents: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute correlations between clinical features and latent codes.

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
    variable_name : mapping or callable, optional
        Mapping (or callable) used to translate feature identifiers into
        display labels on the y-axis. ``None`` keeps the DataFrame index.
    method : {"spearman", "pearson", "kendall"}, default "spearman"
        Correlation coefficient applied pairwise between features and latent
        variables.
    p_adjust : {"fdr_bh", "bonferroni", "holm", None}, default "fdr_bh"
        Multiplicity correction applied to the correlation p-values. ``None``
        disables any adjustment.
    max_dimension : int, default 50
        Guard rail applied when :paramref:`variables` and
        :paramref:`latent_indices` are left unspecified. Larger matrices are
        truncated to ``max_dimension`` in both directions with a warning.
    latents : numpy.ndarray, optional
        Pre-computed latent representations. When omitted ``model.encode`` is
        invoked which returns the posterior means of the latent distribution.
    correlations, p_values : pandas.DataFrame, optional
        Pre-computed correlation and p-value matrices. When both are supplied
        the function skips recomputing statistics from :paramref:`model` and
        :paramref:`X` (which can therefore be left as ``None``), enabling reuse
        across multiple visualisations.

    Returns
    -------
    tuple of pandas.DataFrame
        Pair of ``(correlations, p_values)`` matrices indexed by the selected
        clinical variables and latent dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> class DummyModel:
    ...     def encode(self, frame):
    ...         return np.column_stack([frame["x"], frame["y"]])
    >>> frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    >>> corr, pvals = compute_feature_latent_correlation(DummyModel(), frame)
    >>> corr.loc["x", "z0"]
    1.0

    See Also
    --------
    plot_feature_latent_correlation_bubble : Visualise correlation magnitude and
        significance as a bubble chart.
    plot_feature_latent_correlation_heatmap : Render correlation or p-value
        heatmaps using the computed matrices.
    plot_feature_latent_outcome_path_graph : Construct a layered path graph from
        feature/latent/outcome correlations.
    plot_multilayer_path_graph : Draw layered directed graphs connecting features,
        latent representations and outcomes.
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

    return corr_df, pval_df


def plot_feature_latent_correlation_bubble(
    model,
    X: pd.DataFrame,
    *,
    targets: pd.Series | pd.DataFrame | np.ndarray | Mapping[str, Sequence[object]] | None = None,
    target_name: str = "target",
    variables: Sequence[str] | None = None,
    latent_indices: Sequence[int] | None = None,
    variable_name: Mapping[str, object] | Callable[[str], object] | None = None,
    method: str = "spearman",
    p_adjust: str | None = "fdr_bh",
    title: str | None = None,
    output_path: str | Path | None = None,
    output_formats: Sequence[str] | str = ("png",),
    max_dimension: int = 50,
    latents: np.ndarray | None = None,
    correlations: pd.DataFrame | None = None,
    p_values: pd.DataFrame | None = None,
) -> tuple[plt.Figure, Axes]:
    """Draw a bubble chart summarising feature/latent correlations.

    Each bubble encodes statistical significance through its radius while
    colours reflect the signed correlation coefficients. The underlying
    correlations are derived from :func:`compute_feature_latent_correlation`,
    which uses
    :meth:`suave.model.SUAVE.encode` to obtain posterior means unless custom
    latent samples are supplied.

    Parameters
    ----------
    model : suave.model.SUAVE or compatible object
        Fitted estimator providing an :meth:`encode` method returning latent
        representations of ``X``.
    X : pandas.DataFrame
        Feature matrix aligned with the schema used during training.
    targets : pandas.Series, pandas.DataFrame, numpy.ndarray or mapping, optional
        Targets or predictions appended as extra rows in the correlation matrix.
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
    variable_name : mapping or callable, optional
        Custom mapping from feature identifiers to human-readable labels. When
        omitted the DataFrame index is rendered verbatim.
    title : str, optional
        Axis title displayed above the bubble chart. Defaults to a descriptive
        label derived from :paramref:`method` and :paramref:`p_adjust`.
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
        Pair ``(figure, axis)`` exposing the Matplotlib handles for further
        customisation.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> class DummyModel:
    ...     def encode(self, frame):
    ...         return np.column_stack([frame["x"], frame["y"]])
    >>> frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    >>> fig, ax = plot_feature_latent_correlation_bubble(DummyModel(), frame)
    >>> ax.get_title()
    'Spearman correlation vs. adjusted p-values'

    See Also
    --------
    compute_feature_latent_correlation : Return the correlation and p-value
        matrices without rendering a figure.
    plot_feature_latent_correlation_heatmap : Render either the correlation or
        p-value heatmap.
    plot_feature_latent_outcome_path_graph : Construct a layered path diagram
        from correlations computed on the same data.
    plot_bubble_matrix : Low-level bubble chart helper used for rendering.
    plot_multilayer_path_graph : Visualise layered connections between features,
        latent representations and outcomes.
    """

    if correlations is None and p_values is None:
        corr_df, pval_df = compute_feature_latent_correlation(
            model,
            X,
            targets=targets,
            target_name=target_name,
            variables=variables,
            latent_indices=latent_indices,
            method=method,
            p_adjust=p_adjust,
            max_dimension=max_dimension,
            latents=latents,
        )
    elif correlations is not None and p_values is not None:
        corr_df = correlations.copy()
        pval_df = p_values.copy()
    else:
        raise ValueError(
            "correlations and p_values must either both be provided or both be omitted"
        )

    corr_matrix = corr_df.to_numpy(dtype=float)
    pval_matrix = pval_df.to_numpy(dtype=float)
    use_fdr_label = _is_fdr_adjustment(p_adjust)

    n_features, n_latents = corr_df.shape
    feature_labels = [
        _resolve_variable_label(name, variable_name) for name in corr_df.index
    ]
    latent_labels = [fr"$z_{{{idx + 1}}}$" for idx in range(n_latents)]

    clipped_pvals = np.clip(pval_matrix, np.finfo(float).tiny, 1.0)
    with np.errstate(divide="ignore"):
        neg_log_p = -np.log10(clipped_pvals)
    neg_log_p = np.where(np.isfinite(neg_log_p), neg_log_p, np.nan)
    significance_mask = neg_log_p >= 1.0
    valid_values = neg_log_p[significance_mask]

    if valid_values.size:
        min_value = float(np.nanmin(valid_values))
        max_value = float(np.nanmax(valid_values))
    else:
        min_value = 1.0
        max_value = 1.0

    formatted_corr = np.where(significance_mask, corr_matrix, np.nan)
    finite_corr = formatted_corr[np.isfinite(formatted_corr)]
    max_text_len = max((len(f"{value:.2f}") for value in finite_corr), default=4)
    min_area = max(900.0, 600.0 + 160.0 * max(0, max_text_len - 4))
    max_area = min_area * 4.0

    def _scale_significance(values: np.ndarray) -> np.ndarray:
        if not values.size:
            return np.array([], dtype=float)
        if not valid_values.size or math.isclose(max_value, min_value):
            return np.full(values.shape, max_area, dtype=float)
        normalised = (values - min_value) / (max_value - min_value)
        return normalised * (max_area - min_area) + min_area

    bubble_sizes = np.zeros_like(neg_log_p, dtype=float)
    if valid_values.size:
        bubble_sizes[significance_mask] = _scale_significance(valid_values)

    x_positions = np.arange(n_latents)
    y_positions = np.arange(n_features)
    grid_x, grid_y = np.meshgrid(x_positions, y_positions)
    flat_mask = significance_mask.flatten()

    fig_width = max(7.0, 1.4 * n_latents)
    longest_label = max((len(label) for label in feature_labels), default=1)
    fig_height = max(5.5, 0.45 * n_features, 0.28 * longest_label)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(bottom=0.28, right=0.78)

    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    scatter = ax.scatter(
        grid_x.flatten()[flat_mask],
        grid_y.flatten()[flat_mask],
        s=bubble_sizes.flatten()[flat_mask],
        c=corr_matrix.flatten()[flat_mask],
        cmap=cmap,
        norm=norm,
        edgecolors="black",
        linewidths=0.6,
        alpha=0.85,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(latent_labels)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(feature_labels)
    for label in ax.get_yticklabels():
        label.set_rotation(0)
        label.set_va("center")

    ax.set_xlim(-0.5, n_latents - 0.5)
    ax.set_ylim(n_features - 0.5, -0.5)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_index in range(n_features):
        for col_index in range(n_latents):
            if not significance_mask[row_index, col_index]:
                continue
            corr_value = corr_matrix[row_index, col_index]
            if not np.isfinite(corr_value):
                continue
            rgba = cmap(norm(corr_value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_colour = "white" if luminance < 0.5 else "black"
            ax.text(
                col_index,
                row_index,
                f"{corr_value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_colour,
                fontweight="semibold",
            )

    colorbar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        colorbar_mappable,
        ax=ax,
        orientation="horizontal",
        fraction=0.08,
        pad=0.12,
    )
    cbar.set_label(f"{method.title()} correlation", labelpad=10)
    cbar.ax.xaxis.set_label_position("bottom")

    if valid_values.size:
        unique_count = max(1, int(np.unique(valid_values).size))
        level_count = min(3, unique_count)
        legend_values = np.linspace(min_value, max_value, num=level_count)
        if np.isclose(max_value, min_value):
            legend_values = np.array([min_value])
        legend_sizes = _scale_significance(legend_values)
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#6c6c6c",
                markeredgecolor="#6c6c6c",
                alpha=0.8,
                markersize=math.sqrt(size / math.pi) if size > 0 else 0.0,
            )
            for size in legend_sizes
        ]
        legend_labels = [f"{value:.2f}" for value in legend_values]
        legend_title = (
            "$-\\log_{10}(FDR)$" if use_fdr_label else "$-\\log_{10}(p)$"
        )
        ax.legend(
            handles,
            legend_labels,
            title=legend_title,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )

    ax.set_title(title or "")

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

    return fig, ax


def plot_feature_latent_correlation_heatmap(
    model,
    X: pd.DataFrame,
    *,
    targets: pd.Series | pd.DataFrame | np.ndarray | Mapping[str, Sequence[object]] | None = None,
    target_name: str = "target",
    variables: Sequence[str] | None = None,
    latent_indices: Sequence[int] | None = None,
    variable_name: Mapping[str, object] | Callable[[str], object] | None = None,
    method: str = "spearman",
    p_adjust: str | None = "fdr_bh",
    value: str = "correlation",
    title: str | None = None,
    output_path: str | Path | None = None,
    output_formats: Sequence[str] | str = ("png",),
    max_dimension: int = 50,
    latents: np.ndarray | None = None,
    correlations: pd.DataFrame | None = None,
    p_values: pd.DataFrame | None = None,
) -> tuple[plt.Figure, Axes]:
    """Plot correlation heatmaps with optional p-value annotations.

    Parameters mirror :func:`compute_feature_latent_correlation`. Set
    :paramref:`value` to ``"correlation"`` (default) to annotate each cell with
    the coefficient, or ``"pvalue"`` to print formatted p-values while colours
    continue to reflect the signed correlations. Providing both
    :paramref:`correlations` and :paramref:`p_values` allows reusing
    pre-computed matrices without calling
    :func:`compute_feature_latent_correlation` again.

    Returns
    -------
    tuple
        Pair ``(figure, axis)`` exposing the Matplotlib handles for further
        customisation.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> class DummyModel:
    ...     def encode(self, frame):
    ...         return np.column_stack([frame["x"], frame["y"]])
    >>> frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    >>> fig, ax = plot_feature_latent_correlation_heatmap(DummyModel(), frame)
    >>> ax.get_title()
    'Spearman correlation coefficients'

    See Also
    --------
    compute_feature_latent_correlation : Return the correlation and p-value
        matrices for custom processing.
    plot_feature_latent_correlation_bubble : Bubble-chart overview of correlation
        magnitude and statistical significance.
    plot_feature_latent_outcome_path_graph : Construct a layered path graph from
        the same correlation inputs.
    plot_matrix_heatmap : Low-level heatmap rendering utility.
    plot_multilayer_path_graph : Layered path diagram connecting features,
        latent variables and outcomes.
    """

    if correlations is None and p_values is None:
        corr_df, pval_df = compute_feature_latent_correlation(
            model,
            X,
            targets=targets,
            target_name=target_name,
            variables=variables,
            latent_indices=latent_indices,
            method=method,
            p_adjust=p_adjust,
            max_dimension=max_dimension,
            latents=latents,
        )
    elif correlations is not None and p_values is not None:
        corr_df = correlations.copy()
        pval_df = p_values.copy()
    else:
        raise ValueError(
            "correlations and p_values must either both be provided or both be omitted"
        )

    value_lower = value.lower()
    if value_lower not in {"correlation", "pvalue"}:
        raise ValueError("value must be either 'correlation' or 'pvalue'")

    corr_matrix = corr_df.to_numpy(dtype=float)
    pval_matrix = pval_df.to_numpy(dtype=float)
    use_fdr_label = _is_fdr_adjustment(p_adjust)
    n_features, n_latents = corr_df.shape
    feature_labels = [
        _resolve_variable_label(name, variable_name) for name in corr_df.index
    ]
    latent_labels = [fr"$z_{{{idx + 1}}}$" for idx in range(n_latents)]

    fig_width = max(7.0, 1.2 * n_latents)
    longest_label = max((len(label) for label in feature_labels), default=1)
    fig_height = max(5.0, 0.38 * n_features, 0.26 * longest_label)
    left_margin = min(0.4, 0.18 + 0.012 * longest_label)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=left_margin, bottom=0.28, right=0.98)

    cmap = plt.cm.RdBu_r
    if value_lower == "correlation":
        color_matrix = corr_matrix
        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        colorbar_label = f"{method.title()} correlation"
    else:
        clipped = np.clip(pval_matrix, np.finfo(float).tiny, 1.0)
        with np.errstate(divide="ignore"):
            neg_log = -np.log10(clipped)
        color_matrix = np.where(np.isfinite(neg_log), neg_log, np.nan)
        finite_values = color_matrix[np.isfinite(color_matrix)]
        if finite_values.size:
            vmin = float(np.nanmin(finite_values))
            vmax = float(np.nanmax(finite_values))
        else:
            vmin = 0.0
            vmax = 0.0
        if not math.isfinite(vmin):
            vmin = 0.0
        if not math.isfinite(vmax):
            vmax = 0.0
        if math.isclose(vmin, vmax):
            if vmin <= 0.0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin = max(0.0, vmin - 0.5)
                vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        colorbar_label = "$-\\log_{10}(FDR)$" if use_fdr_label else "$-\\log_{10}(p)$"
    ax.imshow(color_matrix, cmap=cmap, norm=norm, aspect="equal")

    ax.set_xticks(np.arange(n_latents))
    ax.set_xticklabels(latent_labels)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_ha("center")

    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(feature_labels)
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)
        tick.set_va("center")

    ax.set_xlim(-0.5, n_latents - 0.5)
    ax.set_ylim(n_features - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    def _format_p_value(value: float) -> str:
        if not np.isfinite(value):
            return ""
        if value < 0.001:
            return "<0.001"
        if value > 0.99:
            return ">0.99"
        if 0.001 <= value < 0.01 or 0.049 <= value <= 0.051:
            return f"{value:.3f}"
        return f"{value:.2f}"

    for row_index in range(n_features):
        for col_index in range(n_latents):
            color_value = color_matrix[row_index, col_index]
            if not np.isfinite(color_value):
                continue
            if value_lower == "correlation":
                text = f"{corr_matrix[row_index, col_index]:.2f}"
            else:
                text = _format_p_value(pval_matrix[row_index, col_index])
                if not text:
                    continue
            rgba = cmap(norm(color_value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_colour = "white" if luminance < 0.5 else "black"
            ax.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color=text_colour,
                fontweight="semibold",
            )

    colorbar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        colorbar_mappable,
        ax=ax,
        orientation="horizontal",
        fraction=0.08,
        pad=0.12,
    )
    cbar.set_label(colorbar_label, labelpad=10)
    cbar.ax.xaxis.set_label_position("bottom")

    ax.set_title(title or "")

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

    return fig, ax


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


def plot_multilayer_path_graph(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    layer_spacing: float = 2.6,
    node_spacing: float = 1.4,
    node_size: float = 600.0,
    edge_cmap: str | Colormap = plt.cm.RdBu_r,
    edge_color_normalization: str = "minmax",
    edge_label_position: str = "center",
    edge_label_offset: float = 0.1,
    node_outline_color: str = "white",
    node_outline_width: float = 0.0,
    ax: Axes | None = None,
    figure_kwargs: Mapping[str, float] | None = None,
    duplicate_edge_action: str = "warn",
    self_loop_action: str = "warn",
    isolated_node_action: str = "warn",
    group_color_mapping: Mapping[str, str] | None = None,
    layer_color_mapping: Mapping[int, str] | None = None,
    default_layer_cmap: str = "tab10",
    edge_width_range: tuple[float, float] = (1.4, 6.0),
    colorbar_title: str = "Edge colour",
    edge_size_legend_title: str | None = "Edge weight",
    edge_size_legend_values: Sequence[float] | None = None,
    colorbar_kwargs: Mapping[str, float] | None = None,
) -> tuple[plt.Figure, Axes]:
    """Render a layered path diagram from tidy edge and node tables.

    The function organises nodes into vertical layers and draws directed edges
    using smooth curves whose colour, width and transparency are derived from
    the supplied edge attributes. Node colours follow the priority order
    ``node colour column > group colours > layer colours > automatic palette``.

    Parameters
    ----------
    edges : pandas.DataFrame
        Adjacency information with at least the ``source`` and ``target``
        columns. Optional columns include ``weight_edge_size`` (controls edge
        width), ``weight_edge_color`` (drives the colour map), ``alpha``
        (transparency) and ``label`` (text placed along the edge).
    nodes : pandas.DataFrame
        Table describing node attributes. Must contain an ``id`` column used as
        the unique key, alongside a required integer ``layer`` column. Optional
        columns include ``label`` (rendered text), ``color`` (explicit node
        colour) and ``group`` (used for the legend).
    layer_spacing : float, default 2.6
        Horizontal distance between consecutive layers.
    node_spacing : float, default 1.4
        Vertical spacing between nodes within the same layer.
    node_size : float, default 600.0
        Area passed to :func:`matplotlib.axes.Axes.scatter` when drawing nodes.
    edge_cmap : str or matplotlib.colors.Colormap, default ``plt.cm.RdBu_r``
        Colour map applied to ``weight_edge_color``.
    edge_color_normalization : {"minmax", "symmetric"}, default ``"minmax"``
        Strategy for normalising edge colours. ``"symmetric"`` centres the
        colour bar at zero using :class:`~matplotlib.colors.TwoSlopeNorm`.
    edge_label_position : {"source", "center", "target"}, default ``"center"``
        Location along each edge where labels are rendered.
    edge_label_offset : float, default 0.1
        Distance applied orthogonally to the edge direction when placing labels.
    node_outline_color : str, default ``"white"``
        Outline colour applied to all nodes for contrast.
    node_outline_width : float, default 0.0
        Line width of the node outline stroke.
    ax : matplotlib.axes.Axes, optional
        Existing axes used for drawing. When omitted, a new figure and axes are
        created.
    figure_kwargs : Mapping[str, float], optional
        Additional keyword arguments passed to :func:`matplotlib.pyplot.subplots`
        when ``ax`` is ``None``.
    duplicate_edge_action, self_loop_action, isolated_node_action : {"ignore", "warn", "error"}, default ``"warn"``
        Behaviour when encountering duplicate edges (same ``source`` and
        ``target``), self-loops or isolated nodes. The actions emit a warning,
        ignore the issue or raise a :class:`ValueError`.
    group_color_mapping : Mapping[str, str], optional
        Custom colours used for the node ``group`` column.
    layer_color_mapping : Mapping[int, str], optional
        Override colours for specific layers. Applies when nodes do not define
        an explicit ``color`` and no ``group_color_mapping`` is provided.
    default_layer_cmap : str, default ``"tab10"``
        Name of the Matplotlib colour map used when a layer colour needs to be
        assigned automatically.
    edge_width_range : tuple of float, default (1.4, 6.0)
        Range of line widths (in points) mapped from ``weight_edge_size``.
    colorbar_title : str, default ``"Edge colour"``
        Title displayed above the colour bar.
    edge_size_legend_title : str or None, default ``"Edge weight"``
        Title for the legend describing edge widths. Set to ``None`` to omit the
        legend entirely.
    edge_size_legend_values : sequence of float, optional
        Specific weight values illustrated in the edge-width legend. When not
        supplied, three representative values (minimum, median and maximum) are
        chosen automatically.
    colorbar_kwargs : Mapping[str, float], optional
        Additional keyword arguments forwarded to :func:`matplotlib.figure.Figure.colorbar`.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes containing the rendered diagram.

    Raises
    ------
    ValueError
        If required columns are missing, node identifiers are duplicated or the
        configured actions request an error for the detected data issues.

    Warns
    -----
    UserWarning
        Emitted when duplicates, self-loops or isolated nodes are found and the
        corresponding action is ``"warn"``. A warning is also raised if
        ``weight_edge_color`` is unavailable and ``weight_edge_size`` is reused
        for colouring.

    See Also
    --------
    compute_feature_latent_correlation : Compute correlations between features
        and latent codes without plotting.
    plot_feature_latent_correlation_bubble : Bubble-chart view of latent-feature correlations.
    plot_feature_latent_correlation_heatmap : Heatmap of latent-feature correlations or p-values.
    plot_feature_latent_outcome_path_graph : Build a layered path graph directly
        from model correlations and outcomes.
    plot_multilayer_path_graph_from_graph : Render layered diagrams from
        :class:`networkx.DiGraph` objects.

    Examples
    --------
    Build a three-layer graph from tidy tables::

        >>> import matplotlib.pyplot as plt
        >>> edges = pd.DataFrame(
        ...     {
        ...         "source": ["Age", "SOFA", "Phenotype A", "Phenotype B"],
        ...         "target": ["SOFA", "Phenotype A", "Mortality", "Mortality"],
        ...         "weight_edge_size": [0.4, 0.9, 1.2, 0.6],
        ...         "weight_edge_color": [0.1, 0.6, 0.3, -0.2],
        ...         "label": ["$\\beta=0.1$", "", "p<0.01", "p=0.05"],
        ...     }
        ... )
        >>> nodes = pd.DataFrame(
        ...     {
        ...         "id": ["Age", "SOFA", "Phenotype A", "Phenotype B", "Mortality"],
        ...         "label": ["Age", "SOFA", "Phenotype A", "Phenotype B", "28-day"],
        ...         "layer": [0, 1, 2, 2, 3],
        ...         "group": ["Clinical", "Clinical", "Latent", "Latent", "Outcome"],
        ...     }
        ... )
        >>> fig, ax = plot_multilayer_path_graph(edges, nodes)
        >>> plt.close(fig)

    Notes
    -----
    The colour priority is ``node['color']`` > ``group_color_mapping[group]`` >
    ``layer_color_mapping[layer]`` > automatically generated layer colours.
    """

    if not isinstance(edges, pd.DataFrame) or not isinstance(nodes, pd.DataFrame):
        raise TypeError("edges and nodes must both be pandas.DataFrame instances")

    if {"source", "target"} - set(edges.columns):
        missing = {"source", "target"} - set(edges.columns)
        raise ValueError(f"edges DataFrame is missing required columns: {sorted(missing)}")
    if "layer" not in nodes.columns or "id" not in nodes.columns:
        raise ValueError("nodes DataFrame must include 'id' and 'layer' columns")

    edges_proc = edges.copy()
    nodes_proc = nodes.copy()

    # Deduplicate edges using the mean for numeric attributes and first value otherwise.
    duplicates_mask = edges_proc.duplicated(subset=["source", "target"], keep=False)
    if duplicates_mask.any():
        duplicate_count = int(duplicates_mask.sum())
        _handle_data_issue(
            duplicate_edge_action,
            f"found {duplicate_count} duplicate edge rows; values will be aggregated",
        )
        aggregations: dict[str, str] = {}
        for column in edges_proc.columns:
            if column in {"source", "target"}:
                continue
            if pd.api.types.is_numeric_dtype(edges_proc[column]):
                aggregations[column] = "mean"
            else:
                aggregations[column] = "first"
        edges_proc = (
            edges_proc.groupby(["source", "target"], as_index=False).agg(aggregations)
            if aggregations
            else edges_proc.drop_duplicates(subset=["source", "target"])
        )

    # Remove self-loops after notifying the user.
    self_loop_mask = edges_proc["source"] == edges_proc["target"]
    if self_loop_mask.any():
        loop_count = int(self_loop_mask.sum())
        _handle_data_issue(
            self_loop_action,
            f"removed {loop_count} self-loop edge(s); please verify the adjacency table",
        )
        edges_proc = edges_proc.loc[~self_loop_mask].reset_index(drop=True)

    if edges_proc.empty:
        raise ValueError("no edges remain after preprocessing; unable to build the diagram")

    # Validate nodes.
    if nodes_proc["id"].duplicated().any():
        duplicated_ids = nodes_proc.loc[nodes_proc["id"].duplicated(), "id"].tolist()
        raise ValueError(f"node identifiers must be unique; duplicates: {duplicated_ids}")

    try:
        nodes_proc["layer"] = nodes_proc["layer"].astype(int)
    except ValueError as err:  # pragma: no cover - defensive
        raise ValueError("node 'layer' column must contain integers") from err

    node_lookup = nodes_proc.set_index("id", drop=False)
    missing_nodes = (
        (set(edges_proc["source"]) | set(edges_proc["target"]))
        - set(node_lookup.index)
    )
    if missing_nodes:
        raise ValueError(f"edges reference unknown node identifiers: {sorted(missing_nodes)}")

    connected_nodes = set(edges_proc["source"]) | set(edges_proc["target"])
    isolated_nodes = [node for node in node_lookup.index if node not in connected_nodes]
    if isolated_nodes:
        _handle_data_issue(
            isolated_node_action,
            f"found {len(isolated_nodes)} isolated node(s): {isolated_nodes}",
        )

    label_column = "label" if "label" in nodes_proc.columns else None
    if label_column:
        node_labels = nodes_proc[label_column].fillna(nodes_proc["id"])
    else:
        node_labels = nodes_proc["id"]
    node_labels.index = nodes_proc["id"].tolist()

    if isinstance(edge_cmap, str):
        cmap = plt.get_cmap(edge_cmap)
    else:
        cmap = edge_cmap

    if edge_color_normalization not in {"minmax", "symmetric"}:
        raise ValueError(
            "edge_color_normalization must be either 'minmax' or 'symmetric'"
        )

    size_series = pd.to_numeric(edges_proc.get("weight_edge_size"), errors="coerce")
    if size_series.isna().all():
        raise ValueError(
            "edges DataFrame must include a numeric 'weight_edge_size' column to control widths"
        )
    # Replace missing sizes with the column median to maintain continuity.
    if size_series.isna().any():
        median_size = float(size_series.median(skipna=True))
        size_series = size_series.fillna(median_size)

    if "weight_edge_color" in edges_proc.columns:
        color_series = pd.to_numeric(edges_proc["weight_edge_color"], errors="coerce")
    else:
        color_series = pd.Series(np.nan, index=edges_proc.index, dtype=float)
    if color_series.isna().all():
        _handle_data_issue(
            "warn",
            "weight_edge_color not provided; falling back to weight_edge_size for colouring",
        )
        color_series = size_series.copy()

    if "alpha" in edges_proc.columns:
        alpha_series = pd.to_numeric(edges_proc["alpha"], errors="coerce")
    else:
        alpha_series = pd.Series(np.nan, index=edges_proc.index, dtype=float)
    if alpha_series.isna().all():
        abs_size = size_series.abs()
        min_abs = float(abs_size.min())
        max_abs = float(abs_size.max())
        if max_abs - min_abs < 1e-12:
            alpha_values = np.full_like(abs_size.to_numpy(dtype=float), 0.8)
        else:
            norm = Normalize(vmin=min_abs, vmax=max_abs)
            alpha_values = 0.3 + 0.7 * norm(abs_size.to_numpy(dtype=float))
    else:
        alpha_values = np.clip(alpha_series.fillna(alpha_series.median()).to_numpy(dtype=float), 0.0, 1.0)

    color_values = color_series.to_numpy(dtype=float)
    finite_colors = color_values[np.isfinite(color_values)]
    if finite_colors.size == 0:
        finite_colors = np.array([0.0])
    color_min = float(np.nanmin(finite_colors))
    color_max = float(np.nanmax(finite_colors))
    if edge_color_normalization == "minmax":
        if abs(color_max - color_min) < 1e-12:
            color_min -= 1.0
            color_max += 1.0
        norm = Normalize(vmin=color_min, vmax=color_max)
    else:
        limit = max(abs(color_min), abs(color_max))
        if limit < 1e-12:
            limit = 1.0
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    abs_sizes = size_series.abs().to_numpy(dtype=float)
    min_width, max_width = edge_width_range
    min_size = float(abs_sizes.min())
    max_size = float(abs_sizes.max())
    if max_size - min_size < 1e-12:
        widths = np.full_like(abs_sizes, (min_width + max_width) / 2.0)
    else:
        widths = np.interp(abs_sizes, [min_size, max_size], [min_width, max_width])

    # Resolve node colours according to the priority rules.
    layer_values = np.sort(nodes_proc["layer"].unique())
    layer_counts = nodes_proc.groupby("layer").size()
    max_nodes_per_layer = int(layer_counts.max()) if not layer_counts.empty else 0

    # Adapt spacing and marker size heuristically to reduce overlap when the caller
    # relies on default parameters. The adjustments expand vertical distances for
    # dense layers and shrink marker areas when required while remaining
    # overrideable through the function arguments.
    if max_nodes_per_layer > 1:
        node_spacing = max(node_spacing, 1.0 + 0.25 * math.log1p(max_nodes_per_layer))
        if node_size > 0:
            scale = min(1.0, 6.0 / max_nodes_per_layer)
            node_size *= scale
    if layer_color_mapping is not None:
        layer_palette = dict(layer_color_mapping)
    else:
        cmap_layers = plt.get_cmap(default_layer_cmap)
        layer_palette = {
            layer: cmap_layers(idx / max(len(layer_values), 1))
            for idx, layer in enumerate(layer_values)
        }

    group_palette = dict(group_color_mapping) if group_color_mapping else {}

    resolved_colors: dict[str, str] = {}
    legend_groups: dict[str, str] = {}
    for node_id, node_row in node_lookup.iterrows():
        explicit_color = node_row.get("color")
        if is_color_like(explicit_color):
            resolved_color = explicit_color
        else:
            group_value = node_row.get("group")
            if group_value in group_palette and is_color_like(group_palette[group_value]):
                resolved_color = group_palette[group_value]
            else:
                layer_color = layer_palette.get(node_row["layer"])
                if layer_color is None or not is_color_like(layer_color):
                    # Fall back to a neutral grey.
                    resolved_color = "#6c757d"
                else:
                    resolved_color = layer_color
            if explicit_color not in (None, "") and not is_color_like(explicit_color):
                warnings.warn(
                    f"node {node_id!r} specifies colour {explicit_color!r} which is not recognised;"
                    " falling back to group/layer colours",
                    UserWarning,
                    stacklevel=3,
                )

        resolved_colors[node_id] = resolved_color

        group_value = node_row.get("group")
        if isinstance(group_value, str) and group_value not in legend_groups:
            if group_value in group_palette and is_color_like(group_palette[group_value]):
                legend_groups[group_value] = group_palette[group_value]
            else:
                legend_groups[group_value] = resolved_color

    # Compute node positions per layer.
    positions: dict[str, tuple[float, float]] = {}
    for layer_index, layer in enumerate(layer_values):
        layer_nodes = nodes_proc[nodes_proc["layer"] == layer].copy()
        layer_nodes.sort_values(by=["layer", "id"], inplace=True)
        count = len(layer_nodes)
        if count == 0:
            continue
        mid = (count - 1) / 2.0
        x_coord = layer_index * layer_spacing
        for order, (_, node_row) in enumerate(layer_nodes.iterrows()):
            y_coord = (mid - order) * node_spacing
            positions[node_row["id"]] = (x_coord, y_coord)

    if ax is None:
        layer_count = max(len(layer_values), 1)
        width = max(6.0, layer_spacing * (layer_count + 0.5))
        height = max(4.5, node_spacing * max(max_nodes_per_layer, 1) + 2.5)
        fig_kwargs = {"figsize": (width, height)}
        if figure_kwargs:
            fig_kwargs.update(figure_kwargs)
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
    else:
        fig = ax.figure

    scatter = ax.scatter(
        [positions[node_id][0] for node_id in node_lookup.index],
        [positions[node_id][1] for node_id in node_lookup.index],
        s=node_size,
        c=[resolved_colors[node_id] for node_id in node_lookup.index],
        edgecolor=node_outline_color,
        linewidth=node_outline_width,
        zorder=3,
    )

    # Annotate node labels.
    label_offset_x = max(layer_spacing * 0.12, 0.4)
    for node_id, (x_coord, y_coord) in positions.items():
        ax.text(
            x_coord - label_offset_x,
            y_coord,
            str(node_labels.loc[node_id]),
            ha="right",
            va="center",
            fontsize=11,
            zorder=4,
        )

    label_position = edge_label_position.lower()
    if label_position not in {"source", "center", "target"}:
        raise ValueError("edge_label_position must be one of {'source', 'center', 'target'}")
    position_fraction = {"source": 0.2, "center": 0.5, "target": 0.8}[label_position]

    def _bezier_point(
        t: float,
        p0: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
    ) -> tuple[float, float]:
        """Evaluate a cubic Bézier curve at fraction ``t``."""

        mt = 1.0 - t
        x_val = (
            (mt ** 3) * p0[0]
            + 3 * (mt ** 2) * t * p1[0]
            + 3 * mt * (t ** 2) * p2[0]
            + (t ** 3) * p3[0]
        )
        y_val = (
            (mt ** 3) * p0[1]
            + 3 * (mt ** 2) * t * p1[1]
            + 3 * mt * (t ** 2) * p2[1]
            + (t ** 3) * p3[1]
        )
        return (x_val, y_val)

    def _bezier_tangent(
        t: float,
        p0: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
    ) -> tuple[float, float]:
        """Return the tangent vector of a cubic Bézier curve at ``t``."""

        mt = 1.0 - t
        dx_dt = (
            3 * (mt ** 2) * (p1[0] - p0[0])
            + 6 * mt * t * (p2[0] - p1[0])
            + 3 * (t ** 2) * (p3[0] - p2[0])
        )
        dy_dt = (
            3 * (mt ** 2) * (p1[1] - p0[1])
            + 6 * mt * t * (p2[1] - p1[1])
            + 3 * (t ** 2) * (p3[1] - p2[1])
        )
        return (dx_dt, dy_dt)

    for index, edge_row in edges_proc.iterrows():
        start = positions[edge_row["source"]]
        end = positions[edge_row["target"]]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = float(np.hypot(dx, dy))
        if distance == 0.0:
            continue  # Already filtered self-loops; safeguard for degenerate positions.

        curvature = 0.25 if dx >= 0 else -0.25
        normal = np.array([-dy, dx], dtype=float)
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-9:
            normal = normal / normal_norm
        else:
            normal = np.array([0.0, 1.0])
        control_offset = curvature * distance
        control_shift = normal * control_offset
        control1 = (start[0] + dx * 0.25 + control_shift[0], start[1] + dy * 0.25 + control_shift[1])
        control2 = (start[0] + dx * 0.75 + control_shift[0], start[1] + dy * 0.75 + control_shift[1])

        bezier_path = BezierPath(
            [start, control1, control2, end],
            [BezierPath.MOVETO, BezierPath.CURVE4, BezierPath.CURVE4, BezierPath.CURVE4],
        )

        patch = FancyArrowPatch(
            path=bezier_path,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=widths[index],
            color=cmap(norm(color_values[index])),
            alpha=float(alpha_values[index]),
            zorder=2,
        )
        ax.add_patch(patch)

        label_value = edge_row.get("label")
        if isinstance(label_value, str) and label_value:
            t = position_fraction
            label_x, label_y = _bezier_point(t, start, control1, control2, end)
            tangent = _bezier_tangent(t, start, control1, control2, end)
            tangent_norm = float(np.hypot(*tangent))
            if tangent_norm > 0:
                nx, ny = -tangent[1] / tangent_norm, tangent[0] / tangent_norm
                label_x += edge_label_offset * nx
                label_y += edge_label_offset * ny
            ax.text(
                label_x,
                label_y,
                label_value,
                ha="center",
                va="center",
                fontsize=10,
                zorder=5,
            )

    ax.set_axis_off()
    ax.set_xlim(
        min(x for x, _ in positions.values()) - layer_spacing * 0.5,
        max(x for x, _ in positions.values()) + layer_spacing * 0.8,
    )
    y_values = [pos[1] for pos in positions.values()]
    ax.set_ylim(min(y_values) - node_spacing, max(y_values) + node_spacing)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb_kwargs = {"orientation": "horizontal", "fraction": 0.05, "pad": 0.12}
    if colorbar_kwargs:
        cb_kwargs.update(colorbar_kwargs)
    colorbar = fig.colorbar(sm, ax=ax, **cb_kwargs)
    colorbar.set_label(colorbar_title, fontsize=11)
    if colorbar.outline is not None:
        colorbar.outline.set_linewidth(0)
    for spine in colorbar.ax.spines.values():
        spine.set_visible(False)

    # Build group legend using the resolved colours.
    group_handles: list[Line2D] = []
    for group_name, color in legend_groups.items():
        handle = Line2D([0], [0], marker="o", color="none", markerfacecolor=color, label=group_name, markersize=10)
        group_handles.append(handle)

    if group_handles:
        fig_width, fig_height = fig.get_size_inches()
        axis_bbox = ax.get_position()
        axis_width = axis_bbox.width * fig_width
        axis_height = axis_bbox.height * fig_height
        horizontal_margin = fig_width - axis_width
        vertical_margin = fig_height - axis_height

        legend_side = "right" if horizontal_margin >= vertical_margin else "bottom"

        if legend_side == "right":
            approx_entry_height = 0.35
            max_rows = max(1, int(round(axis_height / approx_entry_height)))
            ncol = max(1, min(len(group_handles), math.ceil(len(group_handles) / max_rows)))
            legend_loc = "center left"
            legend_anchor = (1.02, 0.5)
        else:
            approx_entry_width = 1.0
            max_cols = max(1, int(round(axis_width / approx_entry_width)))
            ncol = max(1, min(len(group_handles), max_cols))
            row_count = math.ceil(len(group_handles) / ncol)
            while row_count > 3 and ncol < len(group_handles):
                ncol += 1
                row_count = math.ceil(len(group_handles) / ncol)
            legend_loc = "upper center"
            legend_anchor = (0.5, -0.14)

        group_legend = ax.legend(
            handles=group_handles,
            title="Groups",
            loc=legend_loc,
            bbox_to_anchor=legend_anchor,
            borderaxespad=0.0,
            frameon=False,
            ncol=ncol,
            columnspacing=0.8,
            handletextpad=0.8,
        )
        ax.add_artist(group_legend)

    if edge_size_legend_title is not None:
        weight_values = size_series.to_numpy(dtype=float)
        if edge_size_legend_values is None:
            unique_values = np.unique(np.round(weight_values, 6))
            if unique_values.size <= 3:
                legend_values = unique_values
            else:
                legend_values = np.array(
                    [np.min(weight_values), np.median(weight_values), np.max(weight_values)]
                )
        else:
            legend_values = np.asarray(edge_size_legend_values, dtype=float)

        legend_handles = []
        for value in legend_values:
            if max_size - min_size < 1e-12:
                lw = (min_width + max_width) / 2.0
            else:
                lw = np.interp(abs(value), [min_size, max_size], [min_width, max_width])
            legend_handles.append(
                Line2D([0], [0], color="#6c757d", linewidth=lw, label=f"{value:.2g}")
            )

        if legend_handles:
            edge_legend = ax.legend(
                handles=legend_handles,
                title=edge_size_legend_title,
                loc="lower left",
                bbox_to_anchor=(1.02, 0.0),
                borderaxespad=0.0,
            )
            ax.add_artist(edge_legend)

    fig.tight_layout()
    return fig, ax


def plot_feature_latent_outcome_path_graph(
    model,
    X: pd.DataFrame,
    *,
    y: pd.Series
    | pd.DataFrame
    | np.ndarray
    | Mapping[str, Sequence[object]]
    | None = None,
    target_name: str = "target",
    latents: np.ndarray | None = None,
    method: str = "spearman",
    p_adjust: str | None = "fdr_bh",
    significance_level: float = 0.05,
    significant_alpha: float = 0.7,
    insignificant_alpha: float = 0.0,
    edge_label_top_k: int | None = 5,
    edge_label_format: str = r"$\rho={value:.2f}$",
    node_label_mapping: Mapping[str, str] | None = None,
    node_color_mapping: Mapping[str, str] | None = None,
    node_group_mapping: Mapping[str, str] | None = None,
    group_color_mapping: Mapping[str, str] | None = None,
    layer_color_mapping: Mapping[int, str] | None = None,
    default_layer_cmap: str = "tab10",
    edge_width_range: tuple[float, float] = (1.4, 6.0),
    colorbar_title: str = "Spearman correlation",
    edge_size_legend_title: str | None = "Correlation magnitude",
    edge_size_legend_values: Sequence[float] | None = None,
    colorbar_kwargs: Mapping[str, float] | None = None,
    edge_cmap: str | Colormap = plt.cm.RdBu_r,
    **path_kwargs,
) -> tuple[plt.Figure, Axes]:
    """Render a feature → latent → outcome path graph from correlations.

    The helper aligns ``X`` to the model schema when available, encodes latent
    representations (or uses the supplied :paramref:`latents`), and computes
    feature/latent/outcome correlations using Spearman's coefficient by
    default. Edges are weighted and coloured by the correlation strength while
    their transparency reflects statistical significance after multiple-testing
    correction.

    Parameters
    ----------
    model : suave.model.SUAVE or compatible object, optional
        Fitted estimator exposing a :meth:`encode` method and an optional
        ``schema`` attribute. When ``None`` the caller must provide
        :paramref:`latents` and a warning reminds users to align ``X`` with the
        model schema used for training.
    X : pandas.DataFrame
        Feature matrix to analyse. When ``model.schema`` is defined the columns
        are reordered to match the schema feature order after verifying that no
        training features are missing.
    y : pandas.Series, pandas.DataFrame, numpy.ndarray or mapping, optional
        Outcome(s) correlated against the latent dimensions. Accepts the same
        structures as :func:`compute_feature_latent_correlation`.
    target_name : str, default "target"
        Label applied to one-dimensional :paramref:`y` inputs lacking a name.
    latents : numpy.ndarray, optional
        Pre-computed latent representations. When omitted the function calls
        ``model.encode``. ``model`` must not be ``None`` if latents are not
        supplied.
    method : {"spearman", "pearson", "kendall"}, default "spearman"
        Correlation coefficient forwarded to
        :func:`compute_feature_latent_correlation`.
    p_adjust : {"fdr_bh", "bonferroni", "holm", None}, default "fdr_bh"
        Multiplicity correction applied to the correlation p-values.
    significance_level : float, default 0.05
        Threshold applied to the (adjusted) p-values. Edges with p-values above
        the threshold use :paramref:`insignificant_alpha` while significant
        edges use :paramref:`significant_alpha`.
    significant_alpha : float, default 0.7
        Transparency assigned to statistically significant edges.
    insignificant_alpha : float, default 0.0
        Transparency assigned to non-significant edges.
    edge_label_top_k : int, optional
        Number of statistically significant edges (ranked by absolute
        correlation) annotated with :paramref:`edge_label_format`. ``None`` or
        values less than or equal to zero disable labelling.
    edge_label_format : str, default ``"$\\rho={value:.2f}$"``
        Template used when labelling the strongest edges. Available fields are
        ``value`` (signed correlation), ``abs_value`` (absolute correlation),
        ``source`` and ``target``, plus ``p_value`` when available.
    node_label_mapping, node_color_mapping, node_group_mapping : mapping, optional
        Dictionaries keyed by node identifier that override the automatic
        labels, colours or group assignments for specific nodes. Labels default
        to the node identifier for features/outcomes and ``$z_{k}$`` for latent
        variables. Groups default to ``"Feature"``, ``"Latent"`` and
        ``"Outcome"`` respectively.
    group_color_mapping : mapping, optional
        Colour dictionary passed to :func:`plot_multilayer_path_graph` that maps
        group names to colours. Overrides layer colours when present.
    layer_color_mapping : mapping, optional
        Optional override for per-layer colours used when explicit node colours
        or group colours are unavailable.
    default_layer_cmap : str, default "tab10"
        Matplotlib colormap name sampled for automatic layer colours.
    edge_width_range : tuple of float, default (1.4, 6.0)
        Range of edge widths forwarded to :func:`plot_multilayer_path_graph`.
    colorbar_title : str, default "Spearman correlation"
        Title shown above the correlation colour bar.
    edge_size_legend_title : str or None, default "Correlation magnitude"
        Legend title describing the mapping between edge width and correlation
        magnitude. Set to ``None`` to suppress the legend.
    edge_size_legend_values : sequence of float, optional
        Explicit correlation values highlighted in the edge-width legend.
    colorbar_kwargs : mapping, optional
        Extra keyword arguments for :meth:`matplotlib.figure.Figure.colorbar`.
    edge_cmap : str or matplotlib.colors.Colormap, default ``plt.cm.RdBu_r``
        Colormap applied to correlation values.
    **path_kwargs
        Additional keyword arguments forwarded to
        :func:`plot_multilayer_path_graph` (for example ``layer_spacing`` or
        ``edge_label_position``).

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes populated with the path diagram.

    Warns
    -----
    UserWarning
        Raised when ``model`` lacks a schema or is ``None`` and no automatic
        alignment is possible.

    See Also
    --------
    compute_feature_latent_correlation : Return the correlation and p-value
        matrices used to build the diagram.
    plot_feature_latent_correlation_bubble : Bubble-chart overview of the same
        statistics.
    plot_feature_latent_correlation_heatmap : Heatmap representation of the
        correlation or significance matrices.
    plot_multilayer_path_graph : Lower-level renderer accepting explicit edge
        and node tables.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> class DemoModel:
    ...     def __init__(self):
    ...         from suave.types import Schema
    ...         self.schema = Schema({"age": {"type": "real"}, "sofa": {"type": "real"}})
    ...     def encode(self, frame):
    ...         return np.column_stack([frame["age"], frame["sofa"]])
    >>> frame = pd.DataFrame({"age": [60, 70, 80], "sofa": [4, 6, 8]})
    >>> outcomes = pd.Series([0, 1, 1], name="mortality")
    >>> fig, ax = plot_feature_latent_outcome_path_graph(DemoModel(), frame, y=outcomes)
    >>> plt.close(fig)
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas.DataFrame")

    schema = getattr(model, "schema", None) if model is not None else None
    if schema is not None:
        schema.require_columns(X.columns)
        missing_in_X = [column for column in schema.feature_names if column not in X.columns]
        if missing_in_X:
            raise ValueError(
                "X is missing columns required by the model schema: " f"{missing_in_X}"
            )
        feature_ids = [column for column in schema.feature_names if column in X.columns]
        X_aligned = X.loc[:, feature_ids].copy()
    else:
        if model is None:
            warnings.warn(
                "model is None; ensure X columns follow the training schema order used for latents",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "model does not expose a schema; ensure X columns align with the training schema",
                UserWarning,
                stacklevel=2,
            )
        feature_ids = list(X.columns)
        X_aligned = X.copy()

    if not feature_ids:
        raise ValueError("X must contain at least one feature column")

    def _normalise_targets(
        targets_input: pd.Series
        | pd.DataFrame
        | np.ndarray
        | Mapping[str, Sequence[object]]
        | None,
    ) -> tuple[pd.DataFrame | None, list[str]]:
        if targets_input is None:
            return None, []
        if isinstance(targets_input, pd.Series):
            name = targets_input.name or target_name
            frame = targets_input.to_frame(name=name)
        elif isinstance(targets_input, pd.DataFrame):
            frame = targets_input.copy()
        elif isinstance(targets_input, np.ndarray):
            if targets_input.ndim == 1:
                frame = pd.DataFrame({target_name: targets_input})
            elif targets_input.ndim == 2:
                columns = [f"{target_name}_{idx}" for idx in range(targets_input.shape[1])]
                frame = pd.DataFrame(targets_input, columns=columns)
            else:
                raise ValueError("y array must be one- or two-dimensional")
        elif isinstance(targets_input, Mapping):
            frame = pd.DataFrame(targets_input)
        else:
            frame = pd.DataFrame({target_name: list(targets_input)})

        frame = frame.copy()
        if len(frame) != len(X_aligned):
            raise ValueError("y must align with X by row")
        frame = frame.reset_index(drop=True)
        frame.index = X_aligned.index
        return frame, list(frame.columns)

    targets_frame, target_ids = _normalise_targets(y)

    if latents is None:
        if model is None or not hasattr(model, "encode"):
            raise AttributeError(
                "model must define an 'encode' method or latents must be provided"
            )
        latents_matrix = model.encode(X_aligned)
    else:
        latents_matrix = latents

    latent_array = np.asarray(latents_matrix, dtype=float)
    if latent_array.ndim != 2:
        raise ValueError("latents must be a two-dimensional array")
    if latent_array.shape[0] != len(X_aligned):
        raise ValueError("latents must have the same number of rows as X")

    max_dimension = max(len(feature_ids) + len(target_ids), latent_array.shape[1])

    corr_df, pval_df = compute_feature_latent_correlation(
        model,
        X_aligned,
        targets=targets_frame,
        target_name=target_name,
        method=method,
        p_adjust=p_adjust,
        max_dimension=max_dimension,
        latents=latent_array,
    )

    latent_ids = list(corr_df.columns)
    if not latent_ids:
        raise ValueError("No latent dimensions available to build the path graph")

    target_ids = [name for name in target_ids if name in corr_df.index]

    if significance_level <= 0 or significance_level >= 1:
        raise ValueError("significance_level must lie in (0, 1)")
    if not (0.0 <= significant_alpha <= 1.0 and 0.0 <= insignificant_alpha <= 1.0):
        raise ValueError("alpha values must lie between 0 and 1")

    node_label_mapping = dict(node_label_mapping or {})
    node_color_mapping = dict(node_color_mapping or {})
    node_group_mapping = dict(node_group_mapping or {})

    def _resolve_label(node_id: str, default: str) -> str:
        return node_label_mapping.get(node_id, default or node_id)

    def _resolve_group(node_id: str, default: str) -> str:
        return node_group_mapping.get(node_id, default)

    def _resolve_color(node_id: str) -> str | None:
        return node_color_mapping.get(node_id)

    nodes_records: list[dict[str, object]] = []
    for feature_id in feature_ids:
        record: dict[str, object] = {
            "id": feature_id,
            "layer": 0,
            "label": _resolve_label(feature_id, feature_id),
            "group": _resolve_group(feature_id, "Feature"),
        }
        color = _resolve_color(feature_id)
        if color is not None:
            record["color"] = color
        nodes_records.append(record)

    for index, latent_id in enumerate(latent_ids):
        record = {
            "id": latent_id,
            "layer": 1,
            "label": _resolve_label(latent_id, f"$z_{{{index}}}$"),
            "group": _resolve_group(latent_id, "Latent"),
        }
        color = _resolve_color(latent_id)
        if color is not None:
            record["color"] = color
        nodes_records.append(record)

    for target_id in target_ids:
        record = {
            "id": target_id,
            "layer": 2,
            "label": _resolve_label(target_id, target_id),
            "group": _resolve_group(target_id, "Outcome"),
        }
        color = _resolve_color(target_id)
        if color is not None:
            record["color"] = color
        nodes_records.append(record)

    nodes_df = pd.DataFrame(nodes_records)
    if "color" in nodes_df.columns:
        nodes_df.loc[nodes_df["color"].isna(), "color"] = None

    def _edge_alpha(p_value: float) -> float:
        if not np.isfinite(p_value):
            return insignificant_alpha
        return significant_alpha if p_value <= significance_level else insignificant_alpha

    edge_records: list[dict[str, object]] = []

    for feature_id in feature_ids:
        for latent_id in latent_ids:
            corr_value = float(corr_df.loc[feature_id, latent_id])
            if not np.isfinite(corr_value):
                continue
            p_value = float(pval_df.loc[feature_id, latent_id])
            edge_records.append(
                {
                    "source": feature_id,
                    "target": latent_id,
                    "weight_edge_size": corr_value,
                    "weight_edge_color": corr_value,
                    "alpha": _edge_alpha(p_value),
                    "p_value": p_value if np.isfinite(p_value) else float("nan"),
                }
            )

    for target_id in target_ids:
        for latent_id in latent_ids:
            corr_value = float(corr_df.loc[target_id, latent_id])
            if not np.isfinite(corr_value):
                continue
            p_value = float(pval_df.loc[target_id, latent_id])
            edge_records.append(
                {
                    "source": latent_id,
                    "target": target_id,
                    "weight_edge_size": corr_value,
                    "weight_edge_color": corr_value,
                    "alpha": _edge_alpha(p_value),
                    "p_value": p_value if np.isfinite(p_value) else float("nan"),
                }
            )

    if not edge_records:
        raise ValueError("No finite correlations available to build the path graph")

    edges_df = pd.DataFrame(edge_records)

    if edge_label_top_k is not None and edge_label_top_k > 0:
        significant_edges = edges_df[edges_df["alpha"] > insignificant_alpha]
        if not significant_edges.empty:
            ranked = (
                significant_edges["weight_edge_size"].abs().sort_values(ascending=False).index
            )
            for index in ranked[: int(edge_label_top_k)]:
                value = float(edges_df.loc[index, "weight_edge_size"])
                p_value = edges_df.loc[index, "p_value"]
                edges_df.loc[index, "label"] = edge_label_format.format(
                    value=value,
                    abs_value=abs(value),
                    source=edges_df.loc[index, "source"],
                    target=edges_df.loc[index, "target"],
                    p_value=p_value,
                )

    call_kwargs = dict(path_kwargs)
    call_kwargs.setdefault("edge_color_normalization", "symmetric")
    call_kwargs.setdefault("edge_cmap", edge_cmap)
    call_kwargs.setdefault("edge_width_range", edge_width_range)
    call_kwargs.setdefault("colorbar_title", colorbar_title)
    call_kwargs.setdefault("edge_size_legend_title", edge_size_legend_title)
    call_kwargs.setdefault("edge_size_legend_values", edge_size_legend_values)
    call_kwargs.setdefault("colorbar_kwargs", colorbar_kwargs)
    if group_color_mapping is not None:
        call_kwargs.setdefault("group_color_mapping", group_color_mapping)
    if layer_color_mapping is not None:
        call_kwargs.setdefault("layer_color_mapping", layer_color_mapping)
    call_kwargs.setdefault("default_layer_cmap", default_layer_cmap)

    return plot_multilayer_path_graph(edges_df, nodes_df, **call_kwargs)


def plot_multilayer_path_graph_from_graph(
    graph,
    *,
    edge_size_attr: str = "weight_edge_size",
    edge_color_attr: str | None = "weight_edge_color",
    edge_alpha_attr: str | None = "alpha",
    edge_label_attr: str | None = "label",
    node_layer_attr: str = "layer",
    node_label_attr: str = "label",
    node_color_attr: str = "color",
    node_group_attr: str = "group",
    **path_kwargs,
) -> tuple[plt.Figure, Axes]:
    """Render a layered path graph directly from a NetworkX directed graph.

    Parameters
    ----------
    graph : networkx.DiGraph or compatible directed graph
        Graph whose nodes declare the ``layer`` attribute and optional ``label``,
        ``color`` and ``group`` metadata. Edges must provide
        :paramref:`edge_size_attr` and may include colour, alpha and label
        attributes.
    edge_size_attr : str, default "weight_edge_size"
        Name of the edge attribute controlling the edge width (and by default
        the colour scale).
    edge_color_attr : str or None, default "weight_edge_color"
        Edge attribute mapped to colours. ``None`` omits the column, causing the
        width attribute to be reused for colouring.
    edge_alpha_attr : str or None, default "alpha"
        Edge attribute supplying transparency values.
    edge_label_attr : str or None, default "label"
        Edge attribute rendered as text along the edge when present.
    node_layer_attr : str, default "layer"
        Node attribute indicating the layer index (integer required).
    node_label_attr : str, default "label"
        Node attribute supplying the rendered text label.
    node_color_attr : str, default "color"
        Node attribute specifying an explicit colour.
    node_group_attr : str, default "group"
        Node attribute used to populate the legend grouping.
    **path_kwargs
        Additional keyword arguments forwarded to
        :func:`plot_multilayer_path_graph`.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Matplotlib handles for the rendered diagram.

    Raises
    ------
    ImportError
        If :mod:`networkx` is not installed.
    TypeError
        If ``graph`` is not a directed NetworkX graph.
    ValueError
        If required attributes are missing from nodes or edges.

    See Also
    --------
    plot_multilayer_path_graph : Low-level renderer accepting tabular inputs.
    plot_feature_latent_outcome_path_graph : Automatically build graphs from
        feature/latent/outcome correlations.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.DiGraph()
    >>> G.add_node("x", layer=0, label="X", group="Feature")
    >>> G.add_node("z0", layer=1, label="$z_0$", group="Latent")
    >>> G.add_node("y", layer=2, label="Y", group="Outcome")
    >>> G.add_edge("x", "z0", weight_edge_size=0.6, weight_edge_color=0.6)
    >>> G.add_edge("z0", "y", weight_edge_size=0.6, weight_edge_color=0.6)
    >>> fig, ax = plot_multilayer_path_graph_from_graph(G)
    >>> plt.close(fig)
    """

    try:  # pragma: no cover - optional dependency guard
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - executed only without networkx
        raise ImportError(
            "plot_multilayer_path_graph_from_graph requires the 'networkx' package"
        ) from exc

    if not nx.is_directed(graph):
        raise TypeError("graph must be a directed NetworkX graph")

    nodes_records: list[dict[str, object]] = []
    for node_id, data in graph.nodes(data=True):
        if node_layer_attr not in data:
            raise ValueError(
                f"node {node_id!r} is missing required '{node_layer_attr}' attribute"
            )
        record: dict[str, object] = {"id": node_id, "layer": data[node_layer_attr]}
        if node_label_attr in data:
            record["label"] = data[node_label_attr]
        if node_color_attr in data:
            record["color"] = data[node_color_attr]
        if node_group_attr in data:
            record["group"] = data[node_group_attr]
        nodes_records.append(record)

    if not nodes_records:
        raise ValueError("graph must contain at least one node")

    edges_records: list[dict[str, object]] = []
    for source, target, data in graph.edges(data=True):
        if edge_size_attr not in data:
            raise ValueError(
                f"edge ({source!r}, {target!r}) is missing '{edge_size_attr}' attribute"
            )
        record = {
            "source": source,
            "target": target,
            "weight_edge_size": data[edge_size_attr],
        }
        if edge_color_attr and edge_color_attr in data:
            record["weight_edge_color"] = data[edge_color_attr]
        if edge_alpha_attr and edge_alpha_attr in data:
            record["alpha"] = data[edge_alpha_attr]
        if edge_label_attr and edge_label_attr in data:
            record["label"] = data[edge_label_attr]
        edges_records.append(record)

    if not edges_records:
        raise ValueError("graph must contain at least one edge")

    nodes_df = pd.DataFrame(nodes_records)
    edges_df = pd.DataFrame(edges_records)

    return plot_multilayer_path_graph(edges_df, nodes_df, **path_kwargs)
