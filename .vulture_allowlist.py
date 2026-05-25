"""Vulture allowlist — symbols defined now but not consumed until later epics.

Referenced here so `vulture src .vulture_allowlist.py` does not report them as dead code.
This is a project-wide configuration artifact (the gate-approved path in CLAUDE.md), not a
line-level suppression. Remove entries as their real consumers land in later epics.
"""

from matchodds import __metadata__
from matchodds.config import Settings, settings
from matchodds.data import sources, teams
from matchodds.data.schema import Match
from matchodds.features import pipeline
from matchodds.modeling import baselines, cv, dixon_coles, logistic, metrics, xgboost_model

__metadata__.__repository__

# config.Settings fields/properties are read by later epics (data, features, modelling, serving).
Settings.model_config
settings.random_seed
settings.leagues
settings.seasons_back
settings.football_data_base_url
settings.repo_root
settings.data_dir
settings.models_dir
settings.raw_dir
settings.processed_dir

# data.schema.Match — fields/validators consumed by the matches builder in Task 5.
Match.model_config
Match.home
Match.away
Match.league
Match.date
Match.ft_home_goals
Match.ft_away_goals
Match.result
Match.odds_home
Match.odds_draw
Match.odds_away
Match.source
Match._check_consistency

# data.sources — entrypoints called by `make data` / the matches builder (Task 5).
sources.download_all
sources.load_raw

# data.teams — normalize() consumed by the matches builder (Task 5).
teams.normalize

# features.pipeline.features — the point-in-time serving entrypoint; consumed by tests now and by
# the serving layer (Epic 05). (build_feature_table and matches.load now have real src consumers
# via pipeline.build / `make features`.)
pipeline.features

# modeling.metrics — scoring + reliability helpers consumed by tests now; by train.py (Task 9) and
# the modeling notebook (Task 10) later. (log_loss/brier_score/accuracy already have a src consumer
# via score_summary.)
metrics.encode_labels
metrics.decode_labels
metrics.reliability_curve
metrics.score_summary

# modeling.cv — temporal CV splitter consumed by tests now; by calibration.py (Task 8) and
# train.py (Task 9) later. The split/get_n_splits methods exist for the scikit-learn cv protocol.
cv.TimeOrderedSplit
cv.TimeOrderedSplit.split
cv.TimeOrderedSplit.get_n_splits

# modeling.baselines / logistic / xgboost_model — models consumed by train.py (Task 9).
# (feature_columns and the OutcomeModel fit / predict_proba names already have real src consumers.)
baselines.BookmakerBaseline
logistic.LogisticModel
xgboost_model.XGBoostModel
dixon_coles.DixonColesModel
