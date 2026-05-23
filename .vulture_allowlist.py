"""Vulture allowlist — symbols defined now but not consumed until later epics.

Referenced here so `vulture src .vulture_allowlist.py` does not report them as dead code.
This is a project-wide configuration artifact (the gate-approved path in CLAUDE.md), not a
line-level suppression. Remove entries as their real consumers land in later epics.
"""

from matchodds import __metadata__, config

__metadata__.__repository__

config.RANDOM_SEED
config.REPO_ROOT
config.DATA_DIR
config.RAW_DIR
config.PROCESSED_DIR
config.MODELS_DIR
config.ROLLING_WINDOW_N
config.LEAGUES
