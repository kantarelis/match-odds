"""Tests for the football-data.co.uk download client (offline; no network)."""

import datetime as dt
from pathlib import Path

from matchodds import config
from matchodds.data import sources

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "footballdata_sample.csv"


def test_season_code():
    assert sources.season_code(2023) == "2324"
    assert sources.season_code(2019) == "1920"
    assert sources.season_code(2009) == "0910"


def test_recent_seasons_july_cutover():
    assert sources.recent_seasons(3, today=dt.date(2024, 5, 1)) == [2021, 2022, 2023]
    assert sources.recent_seasons(3, today=dt.date(2024, 8, 1)) == [2022, 2023, 2024]


def test_file_url():
    assert sources.file_url("G1", 2023) == "https://www.football-data.co.uk/mmz4281/2324/G1.csv"


def test_load_raw_reads_core_columns():
    frame = sources.load_raw(FIXTURE)
    assert {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"} <= set(frame.columns)
    assert len(frame) == 2


def _patch_scope(monkeypatch, tmp_path, fake_download):
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    monkeypatch.setattr(config.settings, "leagues", ("English Premier League",))
    monkeypatch.setattr(config.settings, "seasons_back", 1)
    monkeypatch.setattr(sources, "_download", fake_download)


def test_download_all_writes_files_and_manifest(monkeypatch, tmp_path):
    sample = FIXTURE.read_bytes()
    calls = []

    def fake_download(url, timeout=sources._TIMEOUT, retries=sources._RETRIES):
        calls.append(url)
        return sample

    _patch_scope(monkeypatch, tmp_path, fake_download)
    manifest = sources.download_all(today=dt.date(2024, 5, 1))

    assert len(calls) == 1
    assert (tmp_path / "raw" / "E0_2324.csv").exists()
    assert (tmp_path / "raw" / "_manifest.json").exists()
    assert manifest["source"] == "football-data.co.uk"
    files = manifest["files"]
    assert isinstance(files, list) and len(files) == 1
    entry = files[0]
    assert entry["league"] == "English Premier League"
    assert entry["division"] == "E0"
    assert entry["season"] == 2023


def test_download_all_uses_cache(monkeypatch, tmp_path):
    sample = FIXTURE.read_bytes()
    calls = []

    def fake_download(url, timeout=sources._TIMEOUT, retries=sources._RETRIES):
        calls.append(url)
        return sample

    _patch_scope(monkeypatch, tmp_path, fake_download)
    sources.download_all(today=dt.date(2024, 5, 1))
    sources.download_all(today=dt.date(2024, 5, 1))  # cached: no second fetch
    assert len(calls) == 1
