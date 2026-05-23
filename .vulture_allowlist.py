"""Vulture allowlist — symbols defined now but not consumed until later epics.

Referenced here so `vulture src .vulture_allowlist.py` does not report them as dead code.
This is a project-wide configuration artifact (the gate-approved path in CLAUDE.md), not a
line-level suppression. Remove entries as their real consumers land in later epics.
"""

from matchodds import __metadata__
from matchodds.config import Settings, settings

__metadata__.__repository__

# config.Settings fields/properties are read by later epics (data, features, modelling, serving).
Settings.model_config
settings.random_seed
settings.rolling_window_n
settings.leagues
settings.seasons_back
settings.football_data_base_url
settings.openfootball_base_url
settings.repo_root
settings.data_dir
settings.models_dir
settings.raw_dir
settings.processed_dir
