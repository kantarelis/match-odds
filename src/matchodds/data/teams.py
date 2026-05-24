"""Canonical team-name normalization for football-data.co.uk spellings.

``normalize(raw, league)`` maps a raw team name to its canonical form and raises
``UnknownTeamError`` for any name (or league) absent from the registry — new or renamed teams fail
loudly so they are reviewed and added deliberately. Regenerate the registry with
``tests/manual/build_team_registry.py`` after a fresh download.
"""

import json
from pathlib import Path

_REGISTRY_PATH = Path(__file__).with_name("canonical_teams.json")

# Cross-season spelling variants -> canonical. football-data.co.uk is internally consistent,
# so this is empty today; add entries if a future season introduces a variant spelling.
_ALIASES: dict[str, str] = {}


class UnknownTeamError(KeyError):
    """Raised when a raw team name (or league) is not in the canonical registry."""


def _load_registry() -> dict[str, frozenset[str]]:
    raw: dict[str, list[str]] = json.loads(_REGISTRY_PATH.read_text())
    return {league: frozenset(names) for league, names in raw.items()}


_CANONICAL: dict[str, frozenset[str]] = _load_registry()


def canonical_names(league: str) -> frozenset[str]:
    """The set of canonical team names known for ``league``."""
    if league not in _CANONICAL:
        raise UnknownTeamError(f"unknown league {league!r}")
    return _CANONICAL[league]


def normalize(raw: str, league: str) -> str:
    """Return the canonical name for ``raw`` in ``league``; raise UnknownTeamError if unknown."""
    known = canonical_names(league)
    name = _ALIASES.get(raw.strip(), raw.strip())
    if name not in known:
        raise UnknownTeamError(f"unknown team {raw!r} in {league!r}")
    return name
