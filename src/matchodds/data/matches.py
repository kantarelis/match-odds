"""Build the canonical master matches table from the downloaded football-data.co.uk CSVs.

Pipeline: read each raw CSV -> parse + normalize team names (``data.teams``) -> validate every row
through the ``Match`` model (``data.schema``) -> assemble a DataFrame -> dedupe (one row per match)
-> sort by date -> write ``processed_dir/matches.parquet`` + a ``matches.meta.json`` sidecar.

Rows that are not completed matches (no date / goals / valid result — postponements, abandonments)
are skipped. An unknown team raises ``UnknownTeamError`` and fails the build loudly, so the
canonical registry is updated deliberately rather than silently dropping fixtures.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
from pydantic import ValidationError

from matchodds.config import settings
from matchodds.data import sources, teams
from matchodds.data.schema import Match

logger = logging.getLogger(__name__)

_DIV_TO_LEAGUE = {division: league for league, division in sources.DIVISIONS.items()}

_OUTPUT_COLUMNS = [
    "league",
    "date",
    "home",
    "away",
    "ft_home_goals",
    "ft_away_goals",
    "result",
    "odds_home",
    "odds_draw",
    "odds_away",
    "source",
]
_KEY = ["league", "date", "home", "away"]

# Closing-odds column triples in priority order (sharpest first), then opening odds as fallback.
_ODDS_COLUMNS: list[tuple[str, str, str]] = [
    ("PSCH", "PSCD", "PSCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
    ("AvgH", "AvgD", "AvgA"),
]


def _extract_odds(row: pd.Series[Any]) -> tuple[float | None, float | None, float | None]:
    columns = set(row.index)
    for home_col, draw_col, away_col in _ODDS_COLUMNS:
        if {home_col, draw_col, away_col} <= columns:
            try:
                home_odds, draw_odds, away_odds = float(row[home_col]), float(row[draw_col]), float(row[away_col])
            except TypeError, ValueError:
                continue
            if home_odds > 1.0 and draw_odds > 1.0 and away_odds > 1.0:
                return home_odds, draw_odds, away_odds
    return None, None, None


def _row_to_match(row: pd.Series[Any], league: str) -> Match | None:
    if pd.isna(row["_date"]) or pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
        return None
    result = str(row.get("FTR", "")).strip()
    if result not in ("H", "D", "A"):
        return None
    home = teams.normalize(str(row["HomeTeam"]), league)
    away = teams.normalize(str(row["AwayTeam"]), league)
    home_odds, draw_odds, away_odds = _extract_odds(row)
    try:
        return Match(
            home=home,
            away=away,
            league=league,
            date=row["_date"].date(),
            ft_home_goals=int(row["FTHG"]),
            ft_away_goals=int(row["FTAG"]),
            result=cast(Literal["H", "D", "A"], result),
            odds_home=home_odds,
            odds_draw=draw_odds,
            odds_away=away_odds,
            source="football-data",
        )
    except ValidationError:
        logger.warning("skipping inconsistent row in %s: %s vs %s", league, row.get("HomeTeam"), row.get("AwayTeam"))
        return None


def build_matches() -> pd.DataFrame:
    """Read every in-scope raw CSV into one validated, deduped, date-sorted matches DataFrame."""
    records: list[dict[str, Any]] = []
    for path in sorted(settings.raw_dir.glob("*.csv")):
        league = _DIV_TO_LEAGUE.get(path.stem.split("_")[0])
        if league is None:
            continue
        frame = sources.load_raw(path)
        if "Date" not in frame.columns:
            continue
        frame["_date"] = pd.to_datetime(frame["Date"], dayfirst=True, format="mixed", errors="coerce")
        for _, row in frame.iterrows():
            match = _row_to_match(row, league)
            if match is not None:
                records.append(match.model_dump())
    table = pd.DataFrame(records, columns=_OUTPUT_COLUMNS)
    if not table.empty:
        table = table.drop_duplicates(subset=_KEY, keep="first")
        table = table.sort_values(["date", "league", "home"]).reset_index(drop=True)
    return table


def _write_meta(table: pd.DataFrame, processed_dir: Path) -> None:
    by_league = table["league"].value_counts().to_dict()
    meta = {
        "built_at": dt.datetime.now(dt.UTC).isoformat(),
        "rows": int(len(table)),
        "leagues": {str(league): int(count) for league, count in by_league.items()},
        "date_range": [str(table["date"].min()), str(table["date"].max())] if not table.empty else [],
        "source_manifest": "raw/_manifest.json",
    }
    (processed_dir / "matches.meta.json").write_text(json.dumps(meta, indent=2))


def build(write: bool = True) -> pd.DataFrame:
    """Build the master matches table; optionally write the parquet + meta sidecar."""
    table = build_matches()
    if write:
        processed_dir = settings.processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        table.to_parquet(processed_dir / "matches.parquet", index=False)
        _write_meta(table, processed_dir)
    return table


def main() -> None:
    sources.download_all()
    table = build()
    logger.info("built %d matches across %d leagues", len(table), table["league"].nunique())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
