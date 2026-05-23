"""Regenerate the canonical team-name registry from the downloaded football-data.co.uk CSVs.

One-off data-prep (not a pytest test). Run after a fresh download:

    .venv/bin/python tests/manual/build_team_registry.py

Writes ``src/matchodds/data/canonical_teams.json`` (the registry consumed by data.teams) and
``tests/fixtures/raw_team_names.txt`` (the offline corpus used by test_teams). Re-run when a new
season introduces promoted/renamed teams that `normalize()` rejects.
"""

import json

from matchodds.config import settings
from matchodds.data import sources

_REGISTRY = settings.repo_root / "src" / "matchodds" / "data" / "canonical_teams.json"
_RAW_NAMES = settings.repo_root / "tests" / "fixtures" / "raw_team_names.txt"
_DIV_TO_LEAGUE = {division: league for league, division in sources.DIVISIONS.items()}


def main() -> None:
    sources.download_all()
    per_league: dict[str, set[str]] = {league: set() for league in sources.DIVISIONS}
    for path in sorted(settings.raw_dir.glob("*.csv")):
        division = path.stem.split("_")[0]
        league = _DIV_TO_LEAGUE.get(division)
        if league is None:
            continue
        frame = sources.load_raw(path)
        for column in ("HomeTeam", "AwayTeam"):
            for value in frame[column].dropna().astype(str):
                name = value.strip()
                if name:
                    per_league[league].add(name)
    registry = {league: sorted(names) for league, names in per_league.items()}
    _REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
    _RAW_NAMES.write_text("\n".join(f"{lg}\t{nm}" for lg in registry for nm in registry[lg]) + "\n")
    print(f"wrote {sum(len(v) for v in registry.values())} names across {len(registry)} leagues")
    for league, names in registry.items():
        print(f"  {league}: {len(names)}")


if __name__ == "__main__":
    main()
