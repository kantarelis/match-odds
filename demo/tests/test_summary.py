"""Unit tests for the plain-English verbal summary."""

from demo.service import Probabilities
from demo.summary import verbal_summary


def test_clear_favourite_names_outcome_and_percentage():
    summary = verbal_summary(Probabilities(home_win=0.62, draw=0.23, away_win=0.15))

    assert "62%" in summary
    assert "home" in summary.lower()


def test_near_uniform_distribution_reads_as_close_match():
    summary = verbal_summary(Probabilities(home_win=0.34, draw=0.33, away_win=0.33))

    assert "close" in summary.lower()
