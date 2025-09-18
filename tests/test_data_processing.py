import numpy as np
import pandas as pd

from suave.data import inverse_standardize, standardize
from suave.types import Schema


def test_standardize_inverse_handles_mixed_types():
    schema = Schema(
        {
            "real_val": {"type": "real"},
            "positive": {"type": "pos"},
            "count": {"type": "count"},
            "category": {"type": "cat", "n_classes": 3},
            "ordinal": {"type": "ordinal", "n_classes": 4},
        }
    )
    frame = pd.DataFrame(
        {
            "real_val": [1.0, 2.0, np.nan, 4.0],
            "positive": [0.0, 3.0, 7.0, np.nan],
            "count": [0.0, 5.0, 2.0, 1.0],
            "category": ["a", "b", "a", "c"],
            "ordinal": [0, 2, 3, 1],
        }
    )

    transformed, stats = standardize(frame, schema)
    assert set(stats) == set(frame.columns)
    assert np.isfinite(transformed["real_val"].dropna()).all()
    assert np.isfinite(transformed["positive"].dropna()).all()
    assert np.isfinite(transformed["count"].dropna()).all()
    assert stats["count"]["offset"] == 1.0

    restored = inverse_standardize(transformed, schema, stats)
    for column in ["real_val", "positive", "count"]:
        original = frame[column].to_numpy()
        recovered = restored[column].to_numpy()
        mask = ~np.isnan(original)
        assert np.allclose(original[mask], recovered[mask], atol=1e-6)

    restored_cat = restored["category"].astype(str).tolist()
    assert restored_cat == frame["category"].tolist()
    assert (
        list(restored["ordinal"].astype(float))
        == frame["ordinal"].astype(float).tolist()
    )
