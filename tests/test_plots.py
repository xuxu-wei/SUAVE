import matplotlib.pyplot as plt
import pytest

from suave.plots import TrainingPlotMonitor


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
