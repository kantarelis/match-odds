"""Unit tests for the Dixon-Coles bivariate-Poisson goals model."""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from matchodds.features import pipeline
from matchodds.modeling.dixon_coles import DixonColesModel

_SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_matches.csv"

# A league with an unambiguous hierarchy: Strong outscores and out-defends Mid, which beats Weak.
# Every team still scores and concedes a non-zero amount, so the MLE stays well-conditioned.
_FIXTURES = [
    ("Strong", "Mid", 3, 1),
    ("Strong", "Weak", 4, 0),
    ("Mid", "Strong", 1, 3),
    ("Mid", "Weak", 3, 1),
    ("Weak", "Strong", 0, 4),
    ("Weak", "Mid", 1, 3),
]


def _feature_table() -> pd.DataFrame:
    matches = pd.read_csv(_SYNTHETIC, parse_dates=["date"])
    return pipeline.build_feature_table(matches, pipeline.default_accumulators())


def _hierarchy_league(rounds: int = 4) -> pd.DataFrame:
    rows = []
    day = dt.date(2022, 8, 6)
    for _ in range(rounds):
        for home, away, home_goals, away_goals in _FIXTURES:
            rows.append(
                {
                    "league": "Test League",
                    "date": day,
                    "home": home,
                    "away": away,
                    "ft_home_goals": home_goals,
                    "ft_away_goals": away_goals,
                }
            )
            day += dt.timedelta(days=7)
    return pd.DataFrame(rows)


def test_predict_proba_shape_and_sums_to_one():
    table = _feature_table()
    proba = DixonColesModel().fit(table).predict_proba(table)
    assert proba.shape == (len(table), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0.0).all()


def test_recovers_attack_and_defense_ordering():
    fit = DixonColesModel().fit(_hierarchy_league())._leagues["Test League"]
    # Higher attack = scores more; lower defense = concedes less (better).
    assert fit.attack["Strong"] > fit.attack["Mid"] > fit.attack["Weak"]
    assert fit.defense["Strong"] < fit.defense["Mid"] < fit.defense["Weak"]


def test_probabilities_are_in_home_draw_away_order():
    model = DixonColesModel().fit(_hierarchy_league())
    fixtures = pd.DataFrame(
        [
            {"league": "Test League", "home": "Strong", "away": "Weak"},
            {"league": "Test League", "home": "Weak", "away": "Strong"},
        ]
    )
    proba = model.predict_proba(fixtures)
    assert proba[0].argmax() == 0  # strong at home -> home win most likely
    assert proba[1].argmax() == 2  # strong away -> away win most likely


def test_unknown_team_and_league_fall_back_without_error():
    model = DixonColesModel().fit(_hierarchy_league())
    fixtures = pd.DataFrame(
        [
            {"league": "Test League", "home": "Strong", "away": "Newcomer"},  # unseen team
            {"league": "Unknown League", "home": "Strong", "away": "Weak"},  # unseen league
        ]
    )
    proba = model.predict_proba(fixtures)
    assert proba.shape == (2, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0.0).all()


def test_deterministic_fit():
    table = _feature_table()
    first = DixonColesModel().fit(table).predict_proba(table)
    second = DixonColesModel().fit(table).predict_proba(table)
    np.testing.assert_array_equal(first, second)
