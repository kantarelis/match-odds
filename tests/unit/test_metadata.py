"""Smoke tests for package metadata and configuration constants."""

from matchodds import __version__, config


def test_version_is_nonempty_string():
    assert isinstance(__version__, str)
    assert __version__


def test_random_seed_is_int():
    assert isinstance(config.RANDOM_SEED, int)


def test_leagues_is_nonempty_tuple():
    assert isinstance(config.LEAGUES, tuple)
    assert len(config.LEAGUES) > 0
