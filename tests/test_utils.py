import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from suave.utils import set_random_seed


def test_set_random_seed_cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    set_random_seed(42)
