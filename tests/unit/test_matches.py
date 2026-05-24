"""Tests for the master-table builder (offline; uses the committed sample CSV)."""

import json
from pathlib import Path

from matchodds import config
from matchodds.data import matches

_SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "footballdata_sample.csv"


def _setup_raw(monkeypatch, tmp_path, *, extra_rows=""):
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    (raw / "E0_2324.csv").write_text(_SAMPLE.read_text() + extra_rows)
    return tmp_path


def test_build_matches_normalizes_and_validates(monkeypatch, tmp_path):
    _setup_raw(monkeypatch, tmp_path)
    table = matches.build_matches()
    assert len(table) == 2
    assert set(table["home"]) == {"Burnley", "Arsenal"}
    assert list(table.columns)[:7] == [
        "league",
        "date",
        "home",
        "away",
        "ft_home_goals",
        "ft_away_goals",
        "result",
    ]
    burnley = table[table["home"] == "Burnley"].iloc[0]
    assert burnley["league"] == "English Premier League"
    assert burnley["result"] == "A"
    assert int(burnley["ft_home_goals"]) == 0
    assert int(burnley["ft_away_goals"]) == 3
    assert float(burnley["odds_home"]) == 8.0  # B365H from the sample


def test_build_matches_dedupes(monkeypatch, tmp_path):
    duplicate = "E0,11/08/2023,20:00,Burnley,Man City,0,3,A,8,5.5,1.33\n"
    _setup_raw(monkeypatch, tmp_path, extra_rows=duplicate)
    table = matches.build_matches()
    assert len(table) == 2  # duplicate collapsed


def test_build_writes_parquet_and_meta(monkeypatch, tmp_path):
    _setup_raw(monkeypatch, tmp_path)
    matches.build()
    processed = tmp_path / "processed"
    assert (processed / "matches.parquet").exists()
    meta = json.loads((processed / "matches.meta.json").read_text())
    assert meta["rows"] == 2
    assert meta["leagues"]["English Premier League"] == 2
