from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import pickle
import torch


def save_model(model: torch.nn.Module, path: str | Path) -> None:
    """Save model parameters to ``path``.

    The parent directory is created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str | Path) -> torch.nn.Module:
    """Load parameters from ``path`` into ``model`` and return it."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with Path(path).open("r") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Serialize ``obj`` using :mod:`pickle`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    """Load a pickled object from ``path``."""
    with Path(path).open("rb") as f:
        return pickle.load(f)
