"""End-to-end tests for the assembled feature table, materialisation, and the leakage contract."""

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from matchodds import config
from matchodds.features import pipeline

_SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_matches.csv"


def _load() -> pd.DataFrame:
    return pd.read_csv(_SYNTHETIC, parse_dates=["date"])


def _feature_names() -> list[str]:
    return [name for acc in pipeline.default_accumulators() for name in acc.feature_names]


def test_table_column_order_and_one_row_per_match():
    matches = _load()
    table = pipeline.build_feature_table(matches, pipeline.default_accumulators())
    expected = ["league", "date", "home", "away"] + _feature_names() + ["odds_home", "odds_draw", "odds_away", "result"]
    assert list(table.columns) == expected
    assert len(table) == len(matches)
    # Every feature family is represented.
    for prefix in ("elo_", "form_", "h2h_", "strength_", "season_"):
        assert any(col.startswith(prefix) for col in _feature_names())


def test_future_match_cannot_influence_the_past():
    matches = _load()
    base_table = pipeline.build_feature_table(matches, pipeline.default_accumulators())
    future = pd.DataFrame(
        [
            {
                "league": "Test League",
                "date": pd.Timestamp("2025-05-01"),
                "home": "Alpha",
                "away": "Beta",
                "ft_home_goals": 4,
                "ft_away_goals": 0,
                "result": "H",
                "odds_home": 2.00,
                "odds_draw": 3.30,
                "odds_away": 3.80,
                "source": "football-data",
            }
        ]
    )
    rebuilt = pipeline.build_feature_table(
        pd.concat([matches, future], ignore_index=True), pipeline.default_accumulators()
    )
    earlier = rebuilt[rebuilt["date"] < dt.date(2025, 5, 1)].reset_index(drop=True)
    pd.testing.assert_frame_equal(base_table, earlier)


def test_training_rows_equal_pointwise_features_calls():
    matches = _load()
    table = pipeline.build_feature_table(matches, pipeline.default_accumulators())
    feature_cols = _feature_names()
    # Cover a cold-start row, a mid-season row, and the last row.
    for idx in (0, 12, len(table) - 1):
        target = table.iloc[idx]
        fixture = pd.DataFrame(
            [{"league": target["league"], "home": target["home"], "away": target["away"], "date": target["date"]}]
        )
        served = pipeline.features(
            matches, fixture, date_cutoff=target["date"], accumulators=pipeline.default_accumulators()
        )
        served_row = served.iloc[0]
        for col in feature_cols:
            train_val, serve_val = target[col], served_row[col]
            assert (pd.isna(train_val) and pd.isna(serve_val)) or train_val == serve_val, f"{col} differs at row {idx}"


def test_build_writes_parquet_and_meta(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    _load().to_parquet(processed / "matches.parquet", index=False)

    table = pipeline.build()

    assert (processed / "features.parquet").exists()
    reloaded = pd.read_parquet(processed / "features.parquet")
    assert len(reloaded) == len(table) == len(_load())

    meta = json.loads((processed / "features.meta.json").read_text())
    assert meta["rows"] == len(table)
    assert set(_feature_names()) == set(meta["features"])
    assert "elo_diff" in meta["features"]
    assert "season_fraction" in meta["features"]
    assert meta["params"]["elo_k"] == config.settings.elo_k
    assert meta["params"]["rolling_window_n"] == config.settings.rolling_window_n
