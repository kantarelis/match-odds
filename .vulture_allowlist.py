"""Vulture allowlist — symbols defined now but not consumed until later epics.

Referenced here so `vulture src .vulture_allowlist.py` does not report them as dead code.
This is a project-wide configuration artifact (the gate-approved path in CLAUDE.md), not a
line-level suppression. Remove entries as their real consumers land in later epics.
"""

from matchodds import __metadata__
from matchodds.config import Settings, settings
from matchodds.data import matches, sources, teams
from matchodds.data.schema import Match

__metadata__.__repository__

# config.Settings fields/properties are read by later epics (data, features, modelling, serving).
Settings.model_config
settings.random_seed
settings.rolling_window_n
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

# data.matches — load() consumed by notebooks / the feature pipeline (Epic 03).
matches.load
