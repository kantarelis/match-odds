"""Shared fixtures for the serving tests.

The sample season's ``matches.parquet`` is **gitignored** — it is rebuilt offline from the committed
``tests/fixtures/sample/raw`` CSV (the CI notebook step does the same), so it is absent on a fresh
checkout. The ``sample_data_dir`` fixture therefore builds it into a tmp data dir and points the
config singleton there: hermetic, offline, and CI-safe (no reliance on a pre-built parquet, and the
source tree is never written to).
"""

import shutil

import pytest

from matchodds import config
from matchodds.data import matches

_SAMPLE_RAW = config.settings.repo_root / "tests" / "fixtures" / "sample" / "raw"


@pytest.fixture
def sample_data_dir(tmp_path, monkeypatch):
    """A tmp data dir seeded with the sample raw CSV and a freshly built ``matches.parquet``."""
    shutil.copytree(_SAMPLE_RAW, tmp_path / "raw")
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    matches.build()
    return tmp_path
