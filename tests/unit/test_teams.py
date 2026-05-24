"""Tests for team-name normalization (offline; committed registry + fixture)."""

from pathlib import Path

import pytest

from matchodds.data import teams
from matchodds.data.sources import DIVISIONS

_RAW_NAMES = Path(__file__).resolve().parents[1] / "fixtures" / "raw_team_names.txt"


def _raw_pairs() -> list[tuple[str, str]]:
    pairs = []
    for line in _RAW_NAMES.read_text().splitlines():
        if line.strip():
            league, name = line.split("\t", 1)
            pairs.append((league, name))
    return pairs


def test_fixture_covers_every_league():
    assert {league for league, _ in _raw_pairs()} == set(DIVISIONS)


def test_every_raw_name_resolves():
    for league, name in _raw_pairs():
        assert teams.normalize(name, league) in teams.canonical_names(league)


def test_unknown_team_raises():
    with pytest.raises(teams.UnknownTeamError):
        teams.normalize("Definitely Not A Team FC", "English Premier League")


def test_unknown_league_raises():
    with pytest.raises(teams.UnknownTeamError):
        teams.normalize("Arsenal", "Made Up League")


def test_normalize_strips_whitespace():
    league, name = _raw_pairs()[0]
    assert teams.normalize(f"  {name}  ", league) == name
