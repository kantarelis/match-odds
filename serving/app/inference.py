"""Load the frozen artifact once and turn a fixture request into calibrated 1X2 probabilities.

The service is read-only inference (PLAN): on construction :class:`Inference` loads the committed
``models/v1.joblib`` and the master matches table, then every :meth:`Inference.predict` rebuilds the
single pre-match feature row with the *same* :func:`matchodds.features.pipeline.features` that built
the training table — no train/serve skew — and scores it with the loaded model. :func:`get_inference`
constructs the instance once for the app to share.
"""

from __future__ import annotations

import functools
import json

import joblib
import pandas as pd

from matchodds.config import settings
from matchodds.data import matches, teams
from matchodds.features import pipeline
from matchodds.modeling import base
from serving.app.schemas import OutcomeProbabilities, PredictRequest

_ARTIFACT = "v1.joblib"
_METADATA = "v1.metadata.json"


class UnknownFixtureError(ValueError):
    """A request named a league or team outside the in-scope registry — no prediction is fabricated."""


class Inference:
    """Loads the frozen model + matches table once; scores fixtures through the shared feature pipeline."""

    def __init__(self) -> None:
        self._model: base.OutcomeModel = joblib.load(settings.models_dir / _ARTIFACT)
        metadata = json.loads((settings.models_dir / _METADATA).read_text())
        self._feature_columns: list[str] = list(metadata["feature_columns"])
        self._matches = matches.load()
        self._guard_no_skew()

    def _guard_no_skew(self) -> None:
        """Fail fast at startup if the live pipeline's columns no longer match the trained artifact."""
        live = base.feature_columns()
        if live != self._feature_columns:
            raise RuntimeError(
                "feature columns drifted from the trained artifact (train/serve skew): "
                f"model expects {self._feature_columns}, pipeline produces {live}"
            )

    def _validate(self, request: PredictRequest) -> None:
        """Reject an out-of-scope league or unknown team with :class:`UnknownFixtureError` (-> HTTP 422)."""
        if request.league not in settings.leagues:
            raise UnknownFixtureError(f"unknown league {request.league!r}; in scope: {sorted(settings.leagues)}")
        try:
            known = teams.canonical_names(request.league)
        except teams.UnknownTeamError as exc:
            raise UnknownFixtureError(str(exc)) from exc
        unknown = [team for team in (request.home, request.away) if team not in known]
        if unknown:
            raise UnknownFixtureError(f"unknown team(s) {unknown} in {request.league!r}; expected canonical spellings")

    def predict(self, request: PredictRequest) -> OutcomeProbabilities:
        """Calibrated home/draw/away probabilities for a fixture; raises on an out-of-scope fixture."""
        self._validate(request)
        fixtures = pd.DataFrame(
            [{"league": request.league, "date": request.match_date, "home": request.home, "away": request.away}]
        )
        feature_row = pipeline.features(self._matches, fixtures, date_cutoff=request.match_date)
        row = self._model.predict_proba(feature_row)[0]
        return OutcomeProbabilities(home_win=float(row[0]), draw=float(row[1]), away_win=float(row[2]))


@functools.lru_cache(maxsize=1)
def get_inference() -> Inference:
    """Construct (once) the shared :class:`Inference` the API depends on."""
    return Inference()
