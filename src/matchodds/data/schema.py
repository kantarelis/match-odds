"""Pydantic model for a single validated match record at the ingestion edge.

A ``Match`` is a fully-parsed, normalized result (canonical team / league names). It is the
validation boundary: each parsed source row is coerced into a ``Match`` before it is allowed into
the master DataFrame, so malformed or inconsistent rows fail loudly rather than silently polluting
the dataset.
"""

import datetime as dt
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_MIN_DATE = dt.date(1990, 1, 1)
_MAX_DATE = dt.date(2100, 1, 1)


class Match(BaseModel):
    """One validated, normalized match result (home perspective for ``result``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    home: str = Field(min_length=1)
    away: str = Field(min_length=1)
    league: str = Field(min_length=1)
    date: dt.date
    ft_home_goals: int = Field(ge=0)
    ft_away_goals: int = Field(ge=0)
    result: Literal["H", "D", "A"]
    odds_home: float | None = Field(default=None, gt=1.0)
    odds_draw: float | None = Field(default=None, gt=1.0)
    odds_away: float | None = Field(default=None, gt=1.0)
    source: Literal["football-data", "openfootball"]

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if not _MIN_DATE <= self.date <= _MAX_DATE:
            raise ValueError(f"match date {self.date!r} outside plausible range {_MIN_DATE}..{_MAX_DATE}")
        if self.home == self.away:
            raise ValueError("home and away teams must differ")
        if self.ft_home_goals > self.ft_away_goals:
            expected = "H"
        elif self.ft_home_goals < self.ft_away_goals:
            expected = "A"
        else:
            expected = "D"
        if self.result != expected:
            raise ValueError(
                f"result {self.result!r} inconsistent with score "
                f"{self.ft_home_goals}-{self.ft_away_goals} (expected {expected!r})"
            )
        return self
