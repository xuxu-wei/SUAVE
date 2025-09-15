"""CLI for generating synthetic samples from a trained model."""

from __future__ import annotations

import argparse

from ..api import SUAVE
from ..utils.io import load_model
from ..utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data with SUAVE")
    parser.add_argument("--model", type=str, help="Path to model weights", required=False)
    parser.add_argument("--n", type=int, default=10, help="Number of samples to generate")
    args = parser.parse_args()

    set_seed(0)
    model = SUAVE(input_dim=4)
    if args.model:
        load_model(model, args.model)
    df = model.generate(args.n)
    print(df.head())


if __name__ == "__main__":
    main()
