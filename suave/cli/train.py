"""Minimal CLI for training the TabVAE model."""

from __future__ import annotations

import argparse
import yaml
import numpy as np

from ..api import TabVAEClassifier
from ..utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TabVAE")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(0)
    # For demonstration purposes we train on random data matching the config's latent_dim
    input_dim = cfg.get("model", {}).get("latent_dim", 4)
    X = np.random.randn(100, input_dim)
    y = (X[:, 0] > 0).astype(int)

    model = TabVAEClassifier(input_dim=input_dim)
    model.fit(X, y, epochs=10)
    print("Training finished")


if __name__ == "__main__":
    main()
