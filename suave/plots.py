"""Plotting helpers for SUAVE."""

from __future__ import annotations

import itertools
import math
import warnings
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, Literal, Any

import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib.axes import Axes
from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    LogNorm,
    Normalize,
    TwoSlopeNorm,
    is_color_like,
)
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import (
    FuncFormatter, StrMethodFormatter, LogFormatter, LogLocator, FixedLocator
)
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as BezierPath

import seaborn as sns
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
    ("Decoder refining", "#8D6E63"),
)



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


def _format_latent_label_for_display(label: object) -> str:
    """Render latent dimension names such as ``z0`` as LaTeX math text."""

    text = str(label)
    if text.startswith("$"):
        return text
    if text.startswith("z") and text[1:].isdigit():
        index = int(text[1:])
        return fr"$z_{{{index + 1}}}$"
    return text


def _apply_axis_font(source_ax: Axes, target_ax: Axes) -> None:
    """Align font properties on ``target_ax`` with those from ``source_ax``."""

    font_props = source_ax.xaxis.label.get_fontproperties()
    if font_props is None:
        font_props = source_ax.yaxis.label.get_fontproperties()
    if font_props is None:
        for tick in itertools.chain(source_ax.get_xticklabels(), source_ax.get_yticklabels()):
            props = tick.get_fontproperties()
            if props is not None:
                font_props = props
                break
    if font_props is None:
        return

    math_fontfamily = plt.rcParams.get("mathtext.fontset")

    def _apply(text) -> None:
        if text is None:
            return
        text.set_fontproperties(font_props)
        setter = getattr(text, "set_math_fontfamily", None)
        if setter is not None and math_fontfamily is not None:
            setter(math_fontfamily)

    _apply(target_ax.xaxis.label)
    _apply(target_ax.yaxis.label)
    _apply(target_ax.title)
    for tick in itertools.chain(target_ax.get_xticklabels(), target_ax.get_yticklabels()):
        _apply(tick)


def _coerce_float(value: float | int | None) -> float:
    """Convert ``value`` to ``float`` while preserving missing entries."""

    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive branch
        return float("nan")


def _format_p_value(value: float, *, style: str = "medical") -> str:
    """Format *value* according to the requested p-value display *style*."""

    if not np.isfinite(value):
        return ""

    style_lower = style.lower()
    if style_lower == "scientific":
        return f"{value:.1e}"
    if style_lower != "medical":
        raise ValueError("style must be either 'medical' or 'scientific'")

    if value < 0.001:
        return "<0.001"
    if value > 0.99:
        return ">0.99"
    if 0.001 <= value < 0.01 or 0.045 <= value <= 0.054:
        return f"{value:.3f}"
    return f"{value:.2f}"

def apply_cbar_bound_labels(
    cbar,
    *,
    bound_ticks: Literal["both", "lower", "upper", None] | None = None,
    ):
    axis = cbar.ax.xaxis if cbar.orientation == "horizontal" else cbar.ax.yaxis
    cbar_ticks = np.asarray(cbar.get_ticks())
    if bound_ticks in {"both", "lower", "upper"} and cbar_ticks.size:
        formatter = axis.get_major_formatter()
        try:
            base_labels = formatter.format_ticks(cbar_ticks)
        except AttributeError:  # pragma: no cover - fallback for custom formatters
            base_labels = [formatter(tick, index) for index, tick in enumerate(cbar_ticks)]

        def _format_bound_label(label: str, inequality: str) -> str:
            text = label.strip()
            if text.startswith("$") and text.endswith("$"):
                inner = text[1:-1].strip()
                return f"$\\{inequality} {inner}$"
            return f"$\\{inequality} {text}$"

        formatted_labels = list(base_labels)
        if bound_ticks in {"both", "lower"}:
            formatted_labels[0] = _format_bound_label(formatted_labels[0], "leq")
        if bound_ticks in {"both", "upper"} and cbar_ticks.size > 1:
            formatted_labels[-1] = _format_bound_label(formatted_labels[-1], "geq")
        elif bound_ticks == "upper":
            formatted_labels[0] = _format_bound_label(formatted_labels[0], "geq")

        axis.set_major_locator(ticker.FixedLocator(cbar_ticks))
        axis.set_ticklabels(formatted_labels)

def _prepare_targets_frame(
    targets_input: pd.Series
    | pd.DataFrame
    | np.ndarray
    | Mapping[str, Sequence[object]]
    | Sequence[object]
    | None,
    target_name: str,
    *,
    expected_length: int,
    index: pd.Index | None,
    align_label: str,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Normalise *targets_input* into a DataFrame aligned with ``index``."""

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
            raise ValueError(f"{align_label} array must be one- or two-dimensional")
    elif isinstance(targets_input, Mapping):
        frame = pd.DataFrame(targets_input)
    else:
        frame = pd.DataFrame({target_name: list(targets_input)})

    frame = frame.copy()
    if len(frame) != expected_length:
        raise ValueError(f"{align_label} must align with X by row")
    frame = frame.reset_index(drop=True)
    if index is not None:
        frame.index = index
    return frame, list(frame.columns)


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
    """Visualise training and validation metrics during ``fit``.

    Parameters
    ----------
    behaviour : str
        Operating mode of the associated :class:`~suave.model.SUAVE` instance.
        Determines which panels are rendered (supervised vs. unsupervised).
    phase_palette : Mapping[str, str] or Sequence[tuple[str, str]], optional
        Palette used to shade optimisation phases. ``None`` disables phase
        shading.
    save_path : str or pathlib.Path, optional
        Directory that receives ``training_monitor_epoch``
        snapshots for each epoch. Snapshots are only written when the
        directory already exists.
    save_formats : str or Iterable[str], default "jpg"
        One or multiple image extensions (without a leading ``.``) that control
        the exported snapshot formats.
    save_dpi : int, default 360
        Resolution applied to persisted figures.
    """

    def __init__(
        self,
        behaviour: str,
        phase_palette: Mapping[str, str] | Sequence[tuple[str, str]] | None = DEFAULT_PHASE_PALETTE,
        *,
        save_path: str | Path | None = None,
        save_formats: str | Iterable[str] = "jpg",
        save_dpi: int = 360,
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
        self._save_path: Path | None = None
        self._save_formats: tuple[str, ...] = ()
        self._save_dpi: int = int(save_dpi)

        if self._save_dpi <= 0:
            raise ValueError("save_dpi must be a positive integer")

        if save_path is not None:
            candidate = Path(save_path)
            if candidate.exists() and candidate.is_dir():
                self._save_path = candidate
                self._save_formats = self._normalise_save_formats(save_formats)
                if not self._save_formats:
                    warnings.warn(
                        "No valid formats provided; monitor snapshots will not be saved.",
                        UserWarning,
                        stacklevel=3,
                    )
            else:
                warnings.warn(
                    "Training monitor save_path must reference an existing directory;"
                    " disabling figure persistence.",
                    UserWarning,
                    stacklevel=3,
                )
        else:
            self._save_formats = self._normalise_save_formats(save_formats)

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
                frameon=True,
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
        self._save_snapshot(epoch)
        self._refresh()

    def _apply_layout(self) -> None:
        """Apply tight layout while preserving space for the legend when needed."""

        if self._tight_layout_rect is not None:
            self._figure.tight_layout(rect=self._tight_layout_rect)
        else:
            self._figure.tight_layout()

    def _save_snapshot(self, epoch: int) -> None:
        """Persist the current figure state for the requested ``epoch``."""

        if self._save_path is None or not self._save_formats:
            return

        epoch_index = max(int(epoch), 0)
        for extension in self._save_formats:
            suffix = extension.lower().lstrip(".")
            if not suffix:
                continue
            output_path = self._save_path / f"training_monitor_epoch.{suffix}"
            self._figure.savefig(output_path, dpi=self._save_dpi, bbox_inches="tight")

    def _normalise_save_formats(
        self, value: str | Iterable[str]
    ) -> tuple[str, ...]:
        """Return a tuple of unique lowercase extensions without leading dots."""

        if isinstance(value, str):
            candidates = [value]
        else:
            candidates = list(value)

        normalised: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            text = str(candidate).strip().lower().lstrip(".")
            if not text or text in seen:
                continue
            seen.add(text)
            normalised.append(text)
        return tuple(normalised)

    def _update_phase_shading(self, epoch: float, phase: str | None) -> None:
        """Synchronise phase shading across all axes."""

        if not self._phase_palette:
            return

        def _clear_active_phase() -> None:
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


def plot_matrix_heatmap(
    data: pd.DataFrame | np.ndarray,
    *,
    cmap: str | Colormap = "viridis",
    cbar_label: str = "",
    annot: bool | np.ndarray | pd.DataFrame = True,
    fmt: str = ".2f",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    bound_ticks: str | None = "both",
    log_scale_cbar: bool = False,
    row_height: float | None = None,
    output_path: str | Path | None = None,
    output_formats: Sequence[str] | str = ("png",),
    annot_font: Mapping[str, object] | None = None,
    label_font: Mapping[str, object] | None = None,
    cbar_kws: Mapping[str, object] | None = None,
) -> tuple[plt.Figure, Axes]:
    """Plot a rectangular matrix as a heatmap.

    The helper adds a configurable colour bar and forwards annotation styling to
    :func:`seaborn.heatmap`. A logarithmic colour scale is supported when
    :paramref:`log_scale_cbar` is enabled.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Tabular matrix whose index/columns will be used for axis labels.
    cmap : str or matplotlib.colors.Colormap, default "viridis"
        Colormap used to render the heatmap cells.
    cbar_label : str, default "set label"
        Title applied to the colour bar.
    annot : bool or array-like of bool/str, default True
        Overlays textual annotations on each heatmap cell. ``True`` renders the
        numeric values from :paramref:`data`, ``False`` omits text entirely, and
        supplying a DataFrame/ndarray provides custom labels that must match the
        shape of :paramref:`data`.
    fmt : str, default ".2f"
        String formatter applied to numeric annotations.
    vmin, vmax : float, optional
        Lower and upper bounds supplied to :func:`seaborn.heatmap`.
    norm : matplotlib.colors.Normalize, optional
        Custom colour normalisation overriding :paramref:`vmin` and
        :paramref:`vmax` when provided.
    bound_ticks : {"both", "lower", "upper", None}, default "both"
        Controls whether the minimum/maximum colour bar ticks are relabelled
        with inequality symbols. ``"both"`` displays ``"≤"`` at the lower
        bound and ``"≥"`` at the upper bound, ``"lower"``/``"upper"`` affect a
        single extreme, and ``None`` keeps the original numeric tick labels.
    log_scale_cbar : bool, default False
        When ``True`` the colour bar is drawn on a log scale.
    row_height : float, optional
        Height of each row in inches when automatic sizing is insufficient.
    output_path : str or pathlib.Path, optional
        File path (with or without an extension) used to persist the figure. The
        parent directory is created automatically when necessary.
    output_formats : sequence of str or str, default ("png",)
        File extensions saved when :paramref:`output_path` is supplied. Provide
        a string for a single format or a sequence for multiple formats.
    annot_font, label_font : mapping, optional
        Font dictionaries applied to annotations and axis labels respectively.
    cbar_kws : mapping, optional
        Additional keyword arguments forwarded to :func:`seaborn.heatmap`'s
        colour bar creation.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure and matplotlib.axes.Axes
        Matplotlib handles for the rendered heatmap.

    See Also
    --------
    plot_feature_latent_correlation_heatmap : High-level latent correlation
        visualisation building on this helper.
    plot_bubble_matrix : Bubble-chart alternative for matrix-style data.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng(0)
    >>> frame = pd.DataFrame(rng.normal(size=(3, 3)), columns=list("ABC"))
    >>> fig, ax = plot_matrix_heatmap(frame, cmap="magma")
    >>> plt.close(fig)
    """
    from matplotlib.colors import LogNorm

    annot_font_defaults = {"fontsize": 10}
    label_font_defaults = {"fontsize": 12}
    annot_font_config = dict(annot_font_defaults)
    if annot_font is not None:
        annot_font_config.update(dict(annot_font))
    label_font_config = dict(label_font_defaults)
    if label_font is not None:
        label_font_config.update(dict(label_font))
    annot_fontsize = annot_font_config.get("fontsize", 10)
    label_fontsize = label_font_config.get("fontsize", 12)

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # 计算图形大小
    if row_height is not None:
        num_rows = data.shape[1]
        figsize = (10, row_height * num_rows)
    else:
        figsize = (8, 14)

    # 创建图形和轴对象
    fig, ax = plt.subplots(figsize=figsize)
    
    # 确定是否使用对数映射，检查 vmin 是否大于 0
    if norm is not None:
        heatmap_norm = norm
    elif log_scale_cbar:
        if vmin is None or vmin <= 0:
            heatmap_norm = LogNorm(vmin=1, vmax=vmax)
        else:
            heatmap_norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        heatmap_norm = None

    cbar_kwargs = {
        "shrink": 0.95,
        "orientation": "horizontal",
        "label": cbar_label,
        "pad": 0.02,
    }

    if cbar_kws is not None:
        cbar_kwargs.update(dict(cbar_kws))

    # 使用 heatmap 绘制热图
    sns.heatmap(
        data,
        ax=ax,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.,
        linecolor='gray',
        cbar_kws=cbar_kwargs,
        vmin=vmin,
        vmax=vmax,
        norm=heatmap_norm,  # 应用对数或线性映射
        annot_kws=dict(annot_font_config),
    )

    # 设置列标签
    ax.set_xticklabels(list(data.columns), rotation=45, ha='left')
    plt.setp(ax.get_xticklabels(), **label_font_config)
    ax.set_xlabel('')
    # 将列标签放在上端
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # 设置行标签
    ax.set_yticklabels([f'{col}' for col in data.index], rotation=0)
    plt.setp(ax.get_yticklabels(), **label_font_config)
    # ax.set_yticklabels([f'{col} (N={len(gene_set_dict[col])})' for col in data.columns], rotation=0)
    ax.set_ylabel('')

    # 移除刻度线
    ax.tick_params(axis='both', which='both', length=0, labelsize=label_fontsize)
    
    # 获取 color bar 对象
    cbar = ax.collections[0].colorbar
    
    # 最后一步：套上端点标签的 Formatter
    apply_cbar_bound_labels(cbar, bound_ticks=bound_ticks)
    _apply_axis_font(ax, cbar.ax)
    # 保存图像文件，如果指定了输出路径
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
            suffix = f".{str(fmt).lstrip('.')}"
            fig.savefig(
                base_stem.with_suffix(suffix),
                dpi=360,
                pad_inches=0.1,
                bbox_inches="tight",
            )

    if ax.texts:
        plt.setp(ax.texts, **annot_font_config)

    return fig, ax


def plot_bubble_matrix(
    size_matrix: pd.DataFrame | np.ndarray,
    *,
    color_matrix: pd.DataFrame | np.ndarray | None = None,
    label_matrix: pd.DataFrame | np.ndarray | None = None,
    cmap: str | Colormap = "viridis",
    cbar_label: str = "set label",
    vmin: float | None = None,
    vmax: float | None = None,
    bound_ticks: str | None = None,
    size_label: str = "size legend",
    marker_shape: str = "o",
    show_grid: bool = True,
    output_path: str | Path | None = None,
    output_formats: Sequence[str] | str = ("png",),
    cbar_kws: Mapping[str, object] | None = None,
    annot_font: Mapping[str, object] | None = None,
    label_font: Mapping[str, object] | None = None,
) -> tuple[plt.Figure, Axes]:
    """Render a bubble chart for matrix-like data.

    Bubble radii encode :paramref:`size_matrix` while colours stem from
    :paramref:`color_matrix`. Labels inside each bubble default to
    :paramref:`size_matrix` and can be customised via :paramref:`label_matrix`.

    Parameters
    ----------
    size_matrix : pandas.DataFrame or numpy.ndarray
        Numeric matrix controlling bubble radii.
    color_matrix : pandas.DataFrame or numpy.ndarray, optional
        Values mapped to colours. Defaults to :paramref:`size_matrix`.
    label_matrix : pandas.DataFrame or numpy.ndarray, optional
        Text drawn inside each bubble. Defaults to :paramref:`size_matrix`.
    cmap : str or matplotlib.colors.Colormap, default "viridis"
        Palette used for the colour mapping.
    cbar_label : str, default "set label"
        Caption shown above the colour bar.
    vmin, vmax : float, optional
        Colour scale bounds; computed from the data when omitted.
    bound_ticks : {"both", "lower", "upper", None}, optional
        Controls whether colour bar end points are replaced with inequality
        markers. ``"both"`` labels the minimum as ``"≤"`` and the maximum as
        ``"≥"``. ``"lower"``/``"upper"`` affect a single extreme, while ``None``
        keeps the raw numeric ticks.
    size_label : str, default "size legend"
        Title of the manually drawn bubble size legend.
    marker_shape : str, default "o"
        Matplotlib marker symbol used to render the bubbles.
    show_grid : bool, default True
        Whether to display a background grid behind the bubbles.
    output_path : str or pathlib.Path, optional
        Base path used to persist the figure in :paramref:`output_formats`.
    output_formats : sequence of str or str, default ("png",)
        File extensions written when :paramref:`output_path` is provided.
    cbar_kws : mapping, optional
        Extra keyword arguments forwarded to :func:`matplotlib.pyplot.colorbar`.
    annot_font, label_font : mapping, optional
        Font dictionaries for annotations and tick/axis labels respectively.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure and matplotlib.axes.Axes
        Handles to the created figure and axes.

    See Also
    --------
    plot_feature_latent_correlation_bubble : Applies the bubble chart to
        feature/latent correlations.
    plot_matrix_heatmap : Heatmap counterpart for dense matrices.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> frame = pd.DataFrame([[1.0, 0.5], [0.2, 0.8]], columns=["z0", "z1"], index=["a", "b"])
    >>> fig, ax = plot_bubble_matrix(frame, label_matrix=frame.round(2))
    >>> plt.close(fig)
    """
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    
    annot_font_defaults = {"fontsize": 10}
    label_font_defaults = {"fontsize": 12}
    annot_font_config = dict(annot_font_defaults)
    if annot_font is not None:
        annot_font_config.update(dict(annot_font))
    label_font_config = dict(label_font_defaults)
    if label_font is not None:
        label_font_config.update(dict(label_font))
    annot_fontsize = annot_font_config.get("fontsize", 10)
    label_fontsize = label_font_config.get("fontsize", 12)
    # 检查输入是否为DataFrame，并提取行和列标签
    if isinstance(size_matrix, pd.DataFrame):
        row_labels = size_matrix.index.astype(str)
        col_labels = size_matrix.columns.astype(str)
    else:
        n, m = size_matrix.shape
        row_labels = np.arange(n).astype(str)
        col_labels = np.arange(m).astype(str)
        size_matrix = pd.DataFrame(size_matrix, index=row_labels, columns=col_labels)

    if label_matrix is None:
        label_matrix = size_matrix.copy()
    elif not isinstance(label_matrix, pd.DataFrame):
        label_matrix = pd.DataFrame(label_matrix, index=size_matrix.index, columns=size_matrix.columns)
    else:
        # 确保索引和列标签匹配
        label_matrix = label_matrix.reindex(index=size_matrix.index, columns=size_matrix.columns)

    if color_matrix is None:
        color_matrix = size_matrix.copy()
    elif not isinstance(color_matrix, pd.DataFrame):
        color_matrix = pd.DataFrame(color_matrix, index=size_matrix.index, columns=size_matrix.columns)
    else:
        # 确保索引和列标签匹配
        color_matrix = color_matrix.reindex(index=size_matrix.index, columns=size_matrix.columns)

    n, m = size_matrix.shape

    # 准备x和y的位置
    x = np.arange(m)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)

    sizes = size_matrix.values.flatten()
    labels = label_matrix.values.flatten()
    colors = color_matrix.values.flatten()
    X = X.flatten()
    Y = Y.flatten()

    # 创建掩码，过滤掉sizes或colors中的NaN值
    mask = (~np.isnan(sizes)) & (~np.isnan(colors))
    sizes = sizes[mask]
    labels = labels[mask]
    colors = colors[mask]
    X = X[mask]
    Y = Y[mask]

    # 定义圆圈大小，根据标签长度自动调整最小大小
    # 计算标签所需的最小气泡大小
    label_lengths = np.array([len(str(label)) for label in labels])
    # 假设每个字符宽度为字体大小的一半，可以调整该估计
    min_bubble_size = np.max(label_lengths) * annot_fontsize * 5

    min_size = min_bubble_size  # 最小圆圈大小
    max_size = 2000  # 最大圆圈大小

    # 使用np.nanmin和np.nanmax计算最小值和最大值
    size_min = np.nanmin(sizes)
    size_max = np.nanmax(sizes)

    if size_max == size_min:
        sizes_norm = np.ones_like(sizes)
    else:
        sizes_norm = (sizes - size_min) / (size_max - size_min)
    sizes_plot = sizes_norm * (max_size - min_size) + min_size

    # 设置颜色映射的最小值和最大值
    if vmin is None:
        vmin = np.nanmin(colors)
    if vmax is None:
        vmax = np.nanmax(colors)

    norm = Normalize(vmin=vmin, vmax=vmax)

    # 自动设置figsize
    figsize = (m * 0.8 + 2, n * 0.8 + 1)
    fig, ax = plt.subplots(figsize=figsize)

    # 主绘图区域
    scatter = ax.scatter(X, Y, s=sizes_plot, c=colors, cmap=cmap, norm=norm, marker=marker_shape)

    # 自动设置字体颜色
    cmap_instance = plt.get_cmap(cmap)
    color_values = cmap_instance(norm(colors))
    # 计算感知亮度
    luminance = (0.299 * color_values[:, 0] + 0.587 * color_values[:, 1] + 0.114 * color_values[:, 2])
    text_colors = ['white' if l < 0.5 else 'black' for l in luminance]

    # 添加文本标签
    for i in range(len(X)): # X, Y, labels 已经 flatten, 直接循环
        text_kwargs = dict(annot_font_config)
        text_kwargs.setdefault('ha', 'center')
        text_kwargs.setdefault('va', 'center')
        text_kwargs.setdefault('color', text_colors[i])
        ax.text(X[i], Y[i], f'{labels[i]}', **text_kwargs)

    # 设置坐标轴
    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(n))
    
    ax.xaxis.set_ticks_position('top') # 列标签置于上方
    ax.xaxis.set_label_position('top') 
    ax.set_xticklabels(col_labels, rotation=45, ha='left')  # 将对齐方式设置为左对齐
    plt.setp(ax.get_xticklabels(), **label_font_config)

    ax.set_yticklabels(row_labels, rotation=0)
    plt.setp(ax.get_yticklabels(), **label_font_config)
    ax.invert_yaxis()  # 反转y轴，使得第一个矩阵索引在顶部
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='both', labelsize=label_fontsize)

    # 控制背景网格显示
    ax.grid(show_grid)
    # ax.set_facecolor('none')
    
    # ==== 添加颜色条 ====
    # 自定义颜色条的位置和宽度
    # 调整 [left, bottom, width, height]，值的范围是 0 到 1，表示相对于fig的比例
    cbar_left = 0.88  # 颜色条的左边界
    cbar_width = 0.04  # 颜色条的宽度
    cbar_bottom = 0.11  # 颜色条的下边界
    cbar_height = 0.45  # 颜色条的高度

    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

    cbar = plt.colorbar(scatter, cax=cbar_ax, **(cbar_kws or {}))
   
    # 可以根据需要调整标签与颜色条之间的间距
    # cbar.ax.xaxis.labelpad = 10  # 设置标签与颜色条之间的间距，默认值为10，增大值可以增大间距
    apply_cbar_bound_labels(cbar, bound_ticks=bound_ticks)
    _apply_axis_font(ax, cbar.ax)
    # ==== 构造大小图例 ====
    # 自定义大小图例的位置和尺寸
    # 调整 [left, bottom, width, height]
    legend_left = 0.875  # 大小图例的左边界
    legend_width = 0.11  # 大小图例的宽度
    legend_bottom = 0.61  # 大小图例的下边界
    legend_height = 0.25  # 大小图例的高度

    size_legend_ax = fig.add_axes([legend_left, legend_bottom, legend_width, legend_height])

    # 选择4个大小值进行图例展示
    # TODO 实现传入一个列表，指定图例中显示的大小值及其标签
    size_values = np.linspace(size_min, size_max, num=3)
    size_labels = [f'{val:.1f}' for val in size_values]
    if size_max == size_min:
        size_values_norm = np.ones_like(size_values)
    else:
        size_values_norm = (size_values - size_min) / (size_max - size_min)
    size_values_plot = size_values_norm * (max_size - min_size) + min_size

    size_legend_ax.axis('off')

    # 计算图例中每个标记的位置
    legend_y = np.linspace(0.2, 0.8, len(size_values))

    # ==== 调整标记和标签之间的间距 ====
    # 可以修改 scatter 和 text 的 x 坐标，例如，将 0.35 和 0.6 改为其他值

    for y_pos, s, label in zip(legend_y, size_values_plot, size_labels):
        size_legend_ax.scatter([0.35], [y_pos], s=s, color='gray', marker=marker_shape)
        legend_text_kwargs = dict(annot_font_config)
        legend_text_kwargs.setdefault('va', 'center')
        legend_text_kwargs.setdefault('ha', 'left')
        size_legend_ax.text(0.92, y_pos, label, **legend_text_kwargs)
        # 在这里调整图例标记和标签之间的间距
        # 例如，修改上面 scatter 的 x 坐标 0.3 和 text 的 x 坐标 0.6

    size_legend_ax.set_xlim(0, 1)
    size_legend_ax.set_ylim(0, 1)
    size_legend_ax.set_title(size_label, fontdict=annot_font_config, loc='right')
    _apply_axis_font(ax, size_legend_ax)
    # size_legend_ax.title.set_position([0.5, 0.875])  # 进一步微调标题位置
    
    # 保存图像文件，如果指定了输出路径
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
            suffix = f".{str(fmt).lstrip('.')}"
            fig.savefig(
                base_stem.with_suffix(suffix),
                dpi=360,
                pad_inches=0.1,
                bbox_inches="tight",
            )
    
    # 返回 Figure 对象
    return fig, ax


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
        to ``target_name``. The values are concatenated with ``X`` before
        computing correlations and therefore do not depend on model outputs;
        any structure matching the row count of ``X`` is supported.
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
    target_columns: list[str] = []

    target_frame, target_columns = _prepare_targets_frame(
        targets,
        target_name,
        expected_length=len(feature_frame),
        index=feature_frame.index,
        align_label="targets",
    )
    if target_frame is not None:
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
        for column in target_columns:
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
    corr_df.attrs["target_columns"] = list(target_columns)
    pval_df.attrs["target_columns"] = list(target_columns)

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
    p_significance_level: float | None = 0.05,
    title: str | None = None,
    output_path: str | Path | None = None,
    output_formats: Sequence[str] | str = ("png",),
    max_dimension: int = 50,
    correlations: pd.DataFrame | None = None,
    p_values: pd.DataFrame | None = None,
    annot_font: Mapping[str, object] | None = None,
    label_font: Mapping[str, object] | None = None,
    bubble_kwargs: Mapping[str, object] | None = None,
):
    """Draw a bubble chart summarising feature/latent correlations.

    Each bubble encodes statistical significance through its radius while
    colours reflect signed correlation coefficients. Correlations are produced
    by :func:`compute_feature_latent_correlation` unless precomputed matrices
    are supplied.

    Parameters
    ----------
    model : suave.model.SUAVE or compatible object
        Fitted estimator exposing an :meth:`encode` method that maps ``X`` to
        latent representations.
    X : pandas.DataFrame
        Feature matrix aligned with the schema used during training.
    targets : pandas.Series, pandas.DataFrame, numpy.ndarray or mapping, optional
        Auxiliary outcomes to correlate with the latent space. They are joined
        with ``X`` before computing correlations, so any structure whose length
        matches ``X`` is supported.
    target_name : str, default "target"
        Name assigned to :paramref:`targets` when a one-dimensional sequence is
        provided without an explicit label.
    variables : sequence of str, optional
        Subset of feature or target columns to include. ``None`` keeps all
        available variables.
    latent_indices : sequence of int, optional
        Indices of the latent dimensions to analyse. ``None`` uses every
        dimension returned by :meth:`encode`.
    method : {"spearman", "pearson", "kendall"}, default "spearman"
        Correlation coefficient applied pairwise between features and latent
        variables.
    p_adjust : {"fdr_bh", "bonferroni", "holm", None}, default "fdr_bh"
        Multiplicity correction applied to the correlation p-values.
    p_significance_level : float, optional
        Significance threshold used to mask non-significant bubbles. When
        ``None`` the plot displays every coefficient regardless of the
        p-values.
    variable_name : mapping or callable, optional
        Custom mapping from feature identifiers to human-readable labels. When
        omitted the DataFrame index is used verbatim.
    title : str, optional
        Custom title applied to the main axes. Defaults to an empty string.
    output_path : str or pathlib.Path, optional
        Base file name used to persist the figure when provided.
    output_formats : sequence of str or str, default ("png",)
        File formats saved alongside :paramref:`output_path`.
    max_dimension : int, default 50
        Guard rail applied when :paramref:`variables` and
        :paramref:`latent_indices` are left unspecified. Larger matrices are
        truncated to ``max_dimension`` in both directions with a warning.
    correlations : pandas.DataFrame, optional
        Pre-computed correlation coefficients aligned with :paramref:`p_values`.
        When provided alongside :paramref:`p_values` the function skips calling
        :func:`compute_feature_latent_correlation`.
    p_values : pandas.DataFrame, optional
        Pre-computed p-values aligned with :paramref:`correlations`.
    annot_font : mapping, optional
        Font dictionary controlling text annotations. Defaults to
        ``{"fontsize": 10}``.
    label_font : mapping, optional
        Font dictionary for axis labels, ticks and titles. Defaults to
        ``{"fontsize": 12}``.
    bubble_kwargs : mapping, optional
        Extra keyword arguments forwarded to :func:`plot_bubble_matrix`.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure and matplotlib.axes.Axes
        Matplotlib handles for further customisation.

    See Also
    --------
    plot_bubble_matrix : Low-level bubble chart renderer used internally.
    plot_feature_latent_correlation_heatmap : Heatmap alternative for the same
        correlation matrices.
    plot_matrix_heatmap : Base heatmap routine that powers the heatmap view.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import pandas as pd
    >>> class _ToyModel:
    ...     def encode(self, frame):
    ...         return np.column_stack([frame["x"], frame["y"]])
    >>> X = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [2.0, 1.0, 0.0]})
    >>> fig, ax = plot_feature_latent_correlation_bubble(_ToyModel(), X, latent_indices=[0, 1])
    >>> plt.close(fig)
    """
    annot_font_defaults = {"fontsize": 10}
    label_font_defaults = {"fontsize": 12}
    annot_font_config = dict(annot_font_defaults)
    if annot_font is not None:
        annot_font_config.update(dict(annot_font))
    label_font_config = dict(label_font_defaults)
    if label_font is not None:
        label_font_config.update(dict(label_font))
    annot_fontsize = annot_font_config.get("fontsize", 10)
    label_fontsize = label_font_config.get("fontsize", 12)
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
        )
    elif correlations is not None and p_values is not None:
        corr_df = correlations.copy()
        pval_df = p_values.copy()
    else:
        raise ValueError(
            "correlations and p_values must either both be provided or both be omitted"
        )

    target_columns = list(corr_df.attrs.get("target_columns", []))
    if not target_columns:
        target_columns = [name for name in corr_df.index if name not in X.columns]

    if variables is not None:
        selected = list(dict.fromkeys(list(variables) + target_columns))
        corr_df = corr_df.loc[selected]
        pval_df = pval_df.loc[selected]
    if variable_name is not None:
        corr_df = corr_df.rename(index=variable_name)
        pval_df = pval_df.rename(index=variable_name)

    latent_labels = [_format_latent_label_for_display(name) for name in corr_df.columns]
    corr_df.columns = latent_labels
    pval_df.columns = latent_labels
        
    clipped_pvals = np.clip(pval_df, np.finfo(float).tiny, 1.0)
    with np.errstate(divide="ignore"):
        neg_log_p = -np.log10(clipped_pvals)
    neg_log_p = np.where(np.isfinite(neg_log_p), neg_log_p, np.nan)

    if p_significance_level is not None:
        if not (0.0 <= p_significance_level <= 1.0):
            raise ValueError("p_significance_level must be within [0, 1]")
        significance_mask = pval_df.to_numpy(dtype=float) <= p_significance_level
    else:
        significance_mask = np.isfinite(neg_log_p)

    valid_neg_log_p = np.where(significance_mask, neg_log_p, np.nan)
    formatted_corr = np.where(significance_mask, corr_df, np.nan)
    formatted_corr = pd.DataFrame(formatted_corr, columns=corr_df.columns, index=corr_df.index)
    formatted_corr = formatted_corr.map(lambda x: f'{x:.2f}' if not np.isnan(x) else np.nan)

    bubble_options = {
        "label_matrix": formatted_corr,
        "color_matrix": corr_df,
        "cmap": "RdYlBu_r",
        "size_label": r"$-\log_{10}(FDR)$" if _is_fdr_adjustment(p_adjust) else r"$-\log_{10}(p)$",
        "cbar_label": rf"{method.capitalize()} correlation",
        "bound_ticks": "both",
        "vmin": -0.6,
        "vmax": 0.6,
        "output_path": output_path,
        "output_formats": output_formats,
        "annot_font": annot_font_config,
        "label_font": label_font_config,
    }
    if bubble_kwargs is not None:
        bubble_options.update(dict(bubble_kwargs))

    fig, ax = plot_bubble_matrix(
        size_matrix=pd.DataFrame(
            valid_neg_log_p, columns=corr_df.columns, index=corr_df.index
        ),
        **bubble_options,
    )
    ax.tick_params(axis='both', which='both', labelsize=label_fontsize)
    ax.set_xticklabels(latent_labels, rotation=0, ha='center')
    plt.setp(ax.get_xticklabels(), **label_font_config)
    plt.setp(ax.get_yticklabels(), **label_font_config)
    ax.set_title(title or "", fontdict=label_font_config)
    if ax.texts:
        plt.setp(ax.texts, **annot_font_config)
    for extra_ax in fig.axes:
        if extra_ax is ax:
            continue
        _apply_axis_font(ax, extra_ax)
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
    value: str = "correlation",
    p_adjust: str | None = "fdr_bh",
    p_significance_level: float | None = None,
    pvalue_notation: str = "medical",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
    output_formats: Sequence[str] | str = ("png",),
    max_dimension: int = 50,
    latents: np.ndarray | None = None,
    correlations: pd.DataFrame | None = None,
    p_values: pd.DataFrame | None = None,
    annot_font: Mapping[str, object] | None = None,
    label_font: Mapping[str, object] | None = None,
    heatmap_kwargs: Mapping[str, object] | None = None,
) -> tuple[plt.Figure, Axes]:
    """Render feature/latent correlation or p-value heatmaps.

    The helper wraps :func:`compute_feature_latent_correlation` and offers a
    heatmap view of the resulting matrices. Setting :paramref:`value` to
    ``"correlation"`` (default) shows the coefficients on a diverging
    ``RdBu_r`` palette centred at zero, while ``"pvalue"`` colours
    ``-log10``-transformed p-values using ``Blues``. Existing correlation and
    p-value matrices can be reused through :paramref:`correlations` and
    :paramref:`p_values` to avoid recomputation.

    Parameters
    ----------
    model : suave.model.SUAVE or compatible object
        Fitted estimator providing an :meth:`encode` method returning latent
        representations of ``X``.
    X : pandas.DataFrame
        Feature matrix aligned with the schema used during training.
    targets : pandas.Series, pandas.DataFrame, numpy.ndarray or mapping, optional
        Targets or predictions appended as extra rows in the correlation matrix.
        They are concatenated with ``X`` prior to the correlation computation so
        any structure sharing ``len(X)`` is valid.
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
        Custom mapping from feature identifiers to human-readable labels. When
        omitted the DataFrame index is rendered verbatim.
    method : {"spearman", "pearson", "kendall"}, default "spearman"
        Correlation coefficient applied pairwise between features and latent
        variables.
    value : {"correlation", "pvalue"}, default "correlation"
        Controls whether heatmap colours represent correlation coefficients or
        ``-log10`` transformed p-values.
    p_adjust : {"fdr_bh", "bonferroni", "holm", None}, default "fdr_bh"
        Multiplicity correction applied to the correlation p-values. ``None``
        disables any adjustment.
    p_significance_level : float, optional
        Maximum allowable p-value (after adjustment when enabled). Entries
        exceeding the threshold are hidden by replacing them with ``NaN``.
        ``None`` keeps every value.
    pvalue_notation : {"medical", "scientific"}, default "medical"
        Annotation style applied when :paramref:`value` is ``"pvalue"``.
        ``"medical"`` follows biomedical reporting conventions with up to two
        decimal places, whereas ``"scientific"`` prints values using scientific
        notation.
    vmin, vmax : float, optional
        Lower and upper bounds supplied to :func:`seaborn.heatmap`.
    title : str, optional
        Custom title. Defaults to
        ``"{method.title()} correlation coefficients"`` or ``"Correlation
        p-values"`` depending on :paramref:`value`.
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
        Custom latent representations overriding :meth:`encode`.
    correlations : pandas.DataFrame, optional
        Pre-computed correlation coefficients aligned with
        :paramref:`p_values`.
    p_values : pandas.DataFrame, optional
        Pre-computed p-values aligned with :paramref:`correlations`.
    annot_font : mapping, optional
        Font configuration applied to the cell annotations. Defaults to
        ``{"fontsize": 10}``.
    label_font : mapping, optional
        Font configuration applied to labels, ticks and the title. Defaults to
        ``{"fontsize": 12}``.
    heatmap_kwargs : mapping, optional
        Additional keyword arguments forwarded to :func:`plot_matrix_heatmap`
        for fine-grained styling control.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure and matplotlib.axes.Axes
        Matplotlib handles containing the rendered heatmap.

    See Also
    --------
    compute_feature_latent_correlation : Return the correlation and p-value
        matrices for custom processing.
    plot_feature_latent_correlation_bubble : Bubble-chart overview of the same
        relationships.
    plot_matrix_heatmap : Base heatmap utility receiving ``heatmap_kwargs``.
    plot_bubble_matrix : Bubble-chart renderer used by the companion function.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import pandas as pd
    >>> class DummyModel:
    ...     def encode(self, frame):
    ...         return np.column_stack([frame["x"], frame["y"]])
    >>> frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    >>> fig, ax = plot_feature_latent_correlation_heatmap(DummyModel(), frame)
    >>> plt.close(fig)
    """
    annot_font_defaults = {"fontsize": 10}
    label_font_defaults = {"fontsize": 12}
    annot_font_config = dict(annot_font_defaults)
    if annot_font is not None:
        annot_font_config.update(dict(annot_font))
    label_font_config = dict(label_font_defaults)
    if label_font is not None:
        label_font_config.update(dict(label_font))
    annot_fontsize = annot_font_config.get("fontsize", 10)
    label_fontsize = label_font_config.get("fontsize", 12)

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

    if p_significance_level is not None:
        if not (0.0 <= p_significance_level <= 1.0):
            raise ValueError("p_significance_level must be within [0, 1]")

    use_fdr_label = _is_fdr_adjustment(p_adjust)
    corr_df = corr_df.copy()
    pval_df = pval_df.copy()

    if p_significance_level is not None:
        significance_mask = pval_df > p_significance_level
        corr_df = corr_df.mask(significance_mask)
        pval_df = pval_df.mask(significance_mask)

    n_features, n_latents = corr_df.shape
    feature_labels = [
        _resolve_variable_label(name, variable_name) for name in corr_df.index
    ]
    latent_labels = [_format_latent_label_for_display(name) for name in corr_df.columns]

    corr_display = corr_df.copy()
    corr_display.index = feature_labels
    corr_display.columns = latent_labels
    pval_display = pval_df.copy()
    pval_display.index = feature_labels
    pval_display.columns = latent_labels

    fig_width = max(7.0, 1.2 * n_latents)
    longest_label = max((len(label) for label in feature_labels), default=1)
    fig_height = max(5.0, 0.38 * n_features, 0.26 * longest_label)
    left_margin = min(0.4, 0.18 + 0.012 * longest_label)

    cmap = plt.cm.RdBu_r
    heatmap_vmin: float | None = vmin
    heatmap_vmax: float | None = vmax
    norm: Normalize | TwoSlopeNorm | None

    if value_lower == "correlation":
        cmap = plt.cm.RdBu_r
        norm = TwoSlopeNorm(vmin=heatmap_vmin, vcenter=0.0, vmax=heatmap_vmax)
        heatmap_data = corr_display
        annot_matrix = corr_display.applymap(
            lambda val: "" if not np.isfinite(val) else f"{val:.2f}"
        )
        colorbar_label = f"{method.title()} correlation"
    else:
        cmap = plt.cm.Blues
        notation = pvalue_notation.lower()
        if notation not in {"medical", "scientific"}:
            raise ValueError("pvalue_notation must be either 'medical' or 'scientific'")
        clipped = pval_display.clip(lower=np.finfo(float).tiny, upper=1.0)
        with np.errstate(divide="ignore"):
            neg_log = -np.log10(clipped)
        neg_log = neg_log.where(np.isfinite(neg_log))
        heatmap_data = neg_log
        finite_values = neg_log.to_numpy(dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size:
            vmin = float(np.nanmin(finite_values))
            vmax = float(np.nanmax(finite_values))
        else:
            vmin = vmax = 0.0
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
        heatmap_vmin, heatmap_vmax = norm.vmin, norm.vmax
        annot_matrix = pval_display.applymap(lambda val: _format_p_value(val, style=notation))
        colorbar_label = "$-\\log_{10}(FDR)$" if use_fdr_label else "$-\\log_{10}(p)$"

    heatmap_options = {
        "row_height": None,
        "cmap": cmap,
        "cbar_label": colorbar_label,
        "vmin": heatmap_vmin,
        "vmax": heatmap_vmax,
        "annot_font": annot_font_config,
        "label_font": label_font_config,
        "annot": annot_matrix,
        "fmt": "",
        "norm": norm,
    }
    if heatmap_kwargs is not None:
        # heatmap_options.update(dict(heatmap_kwargs))
        extra_options = dict(heatmap_kwargs)
        if ("vmin" in extra_options or "vmax" in extra_options) and "norm" not in extra_options:
            heatmap_options.pop("norm", None)
        heatmap_options.update(extra_options)
        
    fig, ax = plot_matrix_heatmap(heatmap_data, **heatmap_options)
    fig.set_size_inches(fig_width, fig_height)
    fig.subplots_adjust(left=left_margin, bottom=0.28, right=0.98)

    ax.set_xticklabels(latent_labels, rotation=0, ha="center")
    ax.set_yticklabels(feature_labels, rotation=0, va="center")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.tick_params(axis="both", which="both", labelsize=label_fontsize)
    plt.setp(ax.get_xticklabels(), **label_font_config)
    plt.setp(ax.get_yticklabels(), **label_font_config)
    if ax.texts:
        plt.setp(ax.texts, **annot_font_config)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for extra_ax in fig.axes:
        if extra_ax is ax:
            continue
        _apply_axis_font(ax, extra_ax)

    default_title = (
        f"{method.title()} correlation coefficients"
        if value_lower == "correlation"
        else "Correlation p-values"
    )
    ax.set_title(title or default_title, fontdict=label_font_config)

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
    colorbar.set_label(colorbar_title, fontsize=12)
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
        structures as :func:`compute_feature_latent_correlation` and the values
        are appended to ``X`` prior to computing correlations, independent of
        model predictions.
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

    targets_frame, target_ids = _prepare_targets_frame(
        y,
        target_name,
        expected_length=len(X_aligned),
        index=X_aligned.index,
        align_label="y",
    )

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
