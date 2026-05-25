"""Unit tests for the shared model-interface helpers."""

from matchodds.features import pipeline
from matchodds.modeling import base


def test_feature_columns_match_the_pipeline_features():
    columns = base.feature_columns()
    expected = [name for acc in pipeline.default_accumulators() for name in acc.feature_names]
    assert columns == expected
    assert len(columns) == 30
    # One column per family is present...
    for name in ("elo_diff", "form_home_ppg", "h2h_matches", "strength_away_ppg", "season_fraction"):
        assert name in columns
    # ...and identifiers, odds, goals, and the label are excluded.
    for name in ("league", "home", "odds_home", "ft_home_goals", "result"):
        assert name not in columns
