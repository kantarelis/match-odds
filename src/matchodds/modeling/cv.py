"""Time-ordered (forward-chaining) cross-validation for match data.

Plain ``KFold`` / ``StratifiedKFold`` would leak the future into the past — a model evaluated on a
match must only ever have trained on **earlier** matches (CLAUDE.md -> ML / Modeling Discipline). This
splitter divides the chronologically-sorted rows into expanding-window folds: each test fold lies on
strictly later dates than its training rows, and a single calendar day is never divided across the
train/test boundary (the master table has no kickoff times, so a day is treated as one block).

:class:`TimeOrderedSplit` follows scikit-learn's splitter protocol (``split`` + ``get_n_splits``), so
an instance can be passed straight to ``CalibratedClassifierCV(cv=...)`` (Task 8); the dates it was
constructed with — not the ``X`` handed to ``split`` — drive the folds.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import numpy.typing as npt
import pandas as pd

from matchodds.config import settings

IntArray = npt.NDArray[np.intp]


def _compute_splits(dates: Iterable[object], n_splits: int) -> list[tuple[IntArray, IntArray]]:
    ordered = pd.to_datetime(pd.Series(list(dates)))
    if not ordered.is_monotonic_increasing:
        raise ValueError("dates must be sorted in ascending (chronological) order before splitting")
    codes = ordered.to_numpy()
    unique_dates = np.unique(codes)
    if len(unique_dates) < n_splits + 1:
        raise ValueError(
            f"need at least n_splits + 1 = {n_splits + 1} distinct dates to form {n_splits} folds, "
            f"got {len(unique_dates)}"
        )
    groups = np.array_split(unique_dates, n_splits + 1)
    # Cut after each group's last date (side="right" keeps every row of that day on the train side).
    cuts = [int(np.searchsorted(codes, group[-1], side="right")) for group in groups]
    splits: list[tuple[IntArray, IntArray]] = []
    for fold in range(n_splits):
        train_idx = np.arange(0, cuts[fold], dtype=np.intp)
        test_idx = np.arange(cuts[fold], cuts[fold + 1], dtype=np.intp)
        splits.append((train_idx, test_idx))
    return splits


class TimeOrderedSplit:
    """Expanding-window temporal CV over the date-sorted rows it is constructed with.

    ``dates`` is the chronological ``date`` column of those rows; the first group of dates is always
    train-only and each later group becomes one test fold, so the folds cover the data's tail.
    """

    def __init__(self, dates: Iterable[object], *, n_splits: int | None = None) -> None:
        self.n_splits = settings.cv_splits if n_splits is None else n_splits
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {self.n_splits}")
        self._splits = _compute_splits(dates, self.n_splits)

    def split(
        self, _x: object = None, _y: object = None, _groups: object = None
    ) -> Iterator[tuple[IntArray, IntArray]]:
        """Yield ``(train_idx, test_idx)`` per fold. The arguments exist only for the scikit-learn
        splitter protocol and are ignored — the folds come from the construction dates."""
        return iter(self._splits)

    def get_n_splits(self, _x: object = None, _y: object = None, _groups: object = None) -> int:
        """Number of folds (scikit-learn splitter protocol)."""
        return len(self._splits)
