"""Plotting helpers for SUAVE."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


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
