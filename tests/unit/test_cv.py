"""Unit tests for the time-ordered (forward-chaining) CV splitter."""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from matchodds import config
from matchodds.modeling.cv import TimeOrderedSplit

_SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_matches.csv"


def _daily(n: int) -> list[dt.date]:
    """``n`` consecutive single-row days."""
    base = dt.date(2024, 1, 1)
    return [base + dt.timedelta(days=offset) for offset in range(n)]


def test_expanding_window_is_strictly_temporal():
    dates = _daily(12)
    folds = list(TimeOrderedSplit(dates, n_splits=3).split())
    assert len(folds) == 3
    previous_train_len = 0
    for train_idx, test_idx in folds:
        assert len(train_idx) > previous_train_len  # train window expands each fold
        previous_train_len = len(train_idx)
        assert train_idx[0] == 0  # train always starts at the beginning
        assert test_idx[0] == train_idx[-1] + 1  # test is contiguous, right after train
        train_max = max(dates[i] for i in train_idx)
        test_min = min(dates[i] for i in test_idx)
        assert train_max < test_min  # the leakage guarantee
    # The first group is train-only; everything after it is tested exactly once.
    tested = sorted({int(i) for _, test in folds for i in test})
    assert tested == list(range(3, 12))


def test_a_day_is_never_split_across_the_boundary():
    days = _daily(6)
    dates = [day for day in days for _ in range(2)]  # two rows per day -> 12 rows
    for train_idx, test_idx in TimeOrderedSplit(dates, n_splits=2).split():
        train_days = {dates[i] for i in train_idx}
        test_days = {dates[i] for i in test_idx}
        assert train_days.isdisjoint(test_days)
        for day in test_days:  # every row of a tested day stays on the test side
            day_rows = {i for i, value in enumerate(dates) if value == day}
            assert day_rows <= set(test_idx.tolist())


def test_unsorted_dates_raise():
    with pytest.raises(ValueError):
        TimeOrderedSplit(list(reversed(_daily(8))), n_splits=3)


def test_too_few_distinct_dates_raise():
    with pytest.raises(ValueError):
        TimeOrderedSplit(_daily(3), n_splits=5)


def test_n_splits_must_be_at_least_two():
    with pytest.raises(ValueError):
        TimeOrderedSplit(_daily(10), n_splits=1)


def test_default_n_splits_comes_from_settings():
    splitter = TimeOrderedSplit(_daily(10))
    assert splitter.n_splits == config.settings.cv_splits
    assert splitter.get_n_splits() == config.settings.cv_splits
    assert len(list(splitter.split())) == config.settings.cv_splits


def test_splits_are_deterministic_and_reiterable():
    dates = _daily(10)
    one = TimeOrderedSplit(dates, n_splits=4)
    two = TimeOrderedSplit(dates, n_splits=4)
    for (a_train, a_test), (b_train, b_test) in zip(list(one.split()), list(two.split())):
        assert np.array_equal(a_train, b_train)
        assert np.array_equal(a_test, b_test)
    assert len(list(one.split())) == len(list(one.split())) == 4  # split() is reusable


def test_works_on_the_synthetic_fixture_dates():
    dates = pd.read_csv(_SYNTHETIC, parse_dates=["date"])["date"]
    folds = list(TimeOrderedSplit(dates).split())
    assert len(folds) == config.settings.cv_splits
    for train_idx, test_idx in folds:
        train_max = dates.iloc[train_idx.tolist()].max()
        test_min = dates.iloc[test_idx.tolist()].min()
        assert train_max < test_min
