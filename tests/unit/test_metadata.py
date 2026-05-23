"""Smoke tests for package metadata and configuration."""

from matchodds import __version__
from matchodds.config import Settings, settings


def test_version_is_nonempty_string():
    assert isinstance(__version__, str)
    assert __version__


def test_random_seed_is_int():
    assert isinstance(settings.random_seed, int)


def test_leagues_is_nonempty_tuple():
    assert isinstance(settings.leagues, tuple)
    assert len(settings.leagues) > 0


def test_settings_honours_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MATCHODDS_DATA_DIR", str(tmp_path))
    overridden = Settings()
    assert overridden.data_dir == tmp_path
    assert overridden.raw_dir == tmp_path / "raw"
    assert overridden.processed_dir == tmp_path / "processed"
