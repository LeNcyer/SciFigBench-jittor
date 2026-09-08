import json
from pathlib import Path

import pytest


@pytest.fixture
def examples():
    return Path(__file__).resolve().parents[1] / "examples" / "synthetic"


@pytest.fixture
def write_rows(tmp_path):
    def write(name, rows):
        path = tmp_path / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path
    return write
