"""Gradient-boosted decision trees over the engineered features.

XGBoost is the non-linear discriminative model in the bake-off. It consumes the cold-start ``NaN``
features **natively** (learning a default split direction per feature), so — unlike logistic
regression — no imputation is needed. It is trained single-threaded with a fixed seed for
reproducible scores; probabilities are returned in the fixed ``[H, D, A]`` order via the same
class-scatter as the other models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from matchodds.config import settings
from matchodds.modeling import base, metrics


class XGBoostModel(base.OutcomeModel):
    """Multiclass gradient-boosted trees (``multi:softprob``), seeded and single-threaded."""

    def __init__(self) -> None:
        # num_class is left to XGBClassifier (it derives it from the labels at fit time); n_jobs=1
        # keeps the tree-build reduction order fixed so repeated fits give identical probabilities.
        self._model = XGBClassifier(
            objective="multi:softprob",
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            n_jobs=1,
            random_state=settings.random_seed,
        )

    @property
    def estimator(self) -> XGBClassifier:
        """The underlying XGBoost classifier, exposed for calibration (see modeling.calibration)."""
        return self._model

    def fit(self, table: pd.DataFrame, /) -> XGBoostModel:
        features = np.asarray(table[base.feature_columns()].to_numpy(), dtype=np.float64)
        labels = metrics.encode_labels(table["result"])
        self._model.fit(features, labels)
        return self

    def predict_proba(self, table: pd.DataFrame, /) -> base.FloatArray:
        features = np.asarray(table[base.feature_columns()].to_numpy(), dtype=np.float64)
        raw = np.asarray(self._model.predict_proba(features), dtype=np.float64)
        classes = np.asarray(self._model.classes_, dtype=np.intp)
        proba = np.zeros((features.shape[0], len(metrics.CLASSES)), dtype=np.float64)
        proba[:, classes] = raw
        return proba
