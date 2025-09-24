"""Plotting helpers for SUAVE."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np


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
            history["train"].append(_coerce_float(train_metrics.get(name)))
            history["val"].append(_coerce_float(val_metrics.get(name)))

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
                    "title": "Validation AUROC",
                    "ylabel": "AUROC",
                    "plot_train": False,
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
