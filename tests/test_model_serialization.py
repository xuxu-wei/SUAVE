from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suave.model import SUAVE


def test_load_binary_archive_reports_torch_error(tmp_path):
    archive = tmp_path / "broken_model.pt"
    archive.write_bytes(b"PK\x03\x04broken archive contents")

    with pytest.raises(RuntimeError) as excinfo:
        SUAVE.load(archive)

    message = str(excinfo.value)
    assert "Failed to load binary SUAVE archive" in message
    assert "PytorchStreamReader failed reading zip archive" in message
    assert "UnicodeDecodeError" not in message
