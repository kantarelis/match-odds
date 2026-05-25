"""Unit tests for the training bake-off and artifact freeze.

Offline + deterministic (PLAN Decision 10): the feature table is built from the committed synthetic
fixture and frozen into a ``tmp_path`` models dir — never touching the real ``models/v1.joblib``.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from matchodds.features import pipeline
from matchodds.modeling import train

_SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_matches.csv"

_REQUIRED_KEYS = (
    "random_seed",
    "cv_splits",
    "data_version",
    "selected_model",
    "calibration_method",
    "feature_columns",
    "models",
    "library_versions",
)


def _feature_table() -> pd.DataFrame:
    matches = pd.read_csv(_SYNTHETIC, parse_dates=["date"])
    return pipeline.build_feature_table(matches, pipeline.default_accumulators())


def test_writes_a_reloadable_artifact_and_valid_metadata(tmp_path):
    table = _feature_table()
    metadata = train.run(table, tmp_path)

    assert (tmp_path / "v1.joblib").exists()
    assert (tmp_path / "v1.metadata.json").exists()

    model = joblib.load(tmp_path / "v1.joblib")
    proba = model.predict_proba(table)
    assert proba.shape == (len(table), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    for key in _REQUIRED_KEYS:
        assert key in metadata
    assert set(metadata["models"]) == {"baseline", "logistic", "xgboost", "dixon_coles"}
    assert metadata["feature_columns"] == train.base.feature_columns()


def test_selection_is_by_log_loss_and_never_the_baseline(tmp_path):
    metadata = train.run(_feature_table(), tmp_path)
    models = metadata["models"]
    deployable = {name: score for name, score in models.items() if score["deployable"]}
    best = min(deployable, key=lambda name: deployable[name]["log_loss"])

    assert metadata["selected_model"] == best
    assert metadata["selected_model"] != "baseline"
    assert models["baseline"]["deployable"] is False


def test_metadata_metrics_are_deterministic(tmp_path):
    table = _feature_table()
    first = train.run(table, tmp_path / "a")
    second = train.run(table, tmp_path / "b")
    assert first["models"] == second["models"]
    assert first["selected_model"] == second["selected_model"]
    assert first["calibration_method"] == second["calibration_method"]
