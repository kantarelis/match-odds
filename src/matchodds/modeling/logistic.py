"""Multinomial logistic regression over the engineered features.

A linear, interpretable discriminative model: a scikit-learn ``Pipeline`` that median-imputes the
cold-start ``NaN`` features (logistic regression cannot consume ``NaN``), standard-scales them, and
fits a multinomial logistic regression. Probabilities are returned in the fixed ``[H, D, A]`` order —
the fitted classes are scattered back into all three columns, so a fold that never saw one outcome
still produces a full ``(n, 3)`` row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matchodds.config import settings
from matchodds.modeling import base, metrics


class LogisticModel(base.OutcomeModel):
    """Median-impute -> standard-scale -> multinomial logistic regression (seeded)."""

    def __init__(self) -> None:
        # lbfgs (the default solver) gives a multinomial fit for multiclass targets. keep_empty_features
        # retains all-NaN columns (a possible early-fold cold start) so the column layout stays fixed.
        self._pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=1000, random_state=settings.random_seed)),
            ]
        )

    @property
    def estimator(self) -> Pipeline:
        """The underlying scikit-learn pipeline, exposed for calibration (see modeling.calibration)."""
        return self._pipeline

    def fit(self, table: pd.DataFrame, /) -> LogisticModel:
        features = np.asarray(table[base.feature_columns()].to_numpy(), dtype=np.float64)
        labels = metrics.encode_labels(table["result"])
        self._pipeline.fit(features, labels)
        return self

    def predict_proba(self, table: pd.DataFrame, /) -> base.FloatArray:
        features = np.asarray(table[base.feature_columns()].to_numpy(), dtype=np.float64)
        raw = np.asarray(self._pipeline.predict_proba(features), dtype=np.float64)
        classes = np.asarray(self._pipeline.classes_, dtype=np.intp)
        proba = np.zeros((features.shape[0], len(metrics.CLASSES)), dtype=np.float64)
        proba[:, classes] = raw
        return proba
