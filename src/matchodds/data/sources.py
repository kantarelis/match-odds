"""Download client for football-data.co.uk match CSVs, plus a version manifest.

football-data.co.uk publishes one CSV per league-season at ``mmz4281/{season}/{div}.csv``
(e.g. ``E0`` England, ``G1`` Greece) in a single consistent layout with full-time results and
bookmaker odds. This module downloads the in-scope league-seasons into ``settings.raw_dir``
(skipping cached files) and records a manifest of what was fetched, for reproducibility.
"""

import datetime as dt
import hashlib
import json
import logging
from pathlib import Path

import httpx
import pandas as pd

from matchodds.config import settings

logger = logging.getLogger(__name__)

# Canonical league name -> football-data.co.uk division code.
DIVISIONS: dict[str, str] = {
    "English Premier League": "E0",
    "Spanish La Liga": "SP1",
    "Italian Serie A": "I1",
    "German Bundesliga": "D1",
    "French Ligue 1": "F1",
    "Greek Super League": "G1",
}

_TIMEOUT = 30.0
_RETRIES = 3
_MANIFEST_NAME = "_manifest.json"


def season_code(start_year: int) -> str:
    """football-data season code for a season starting in ``start_year`` (2023 -> '2324')."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def recent_seasons(n: int, today: dt.date | None = None) -> list[int]:
    """The last ``n`` season start-years (most recent last); seasons roll over in July."""
    today = today or dt.date.today()
    latest = today.year if today.month >= 7 else today.year - 1
    return list(range(latest - n + 1, latest + 1))


def file_url(division: str, start_year: int) -> str:
    """URL of one league-season CSV on football-data.co.uk."""
    return f"{settings.football_data_base_url}{season_code(start_year)}/{division}.csv"


def _download(url: str, timeout: float = _TIMEOUT, retries: int = _RETRIES) -> bytes | None:
    """GET ``url`` with retries. Returns the body, or ``None`` if the season is not published (404)."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = httpx.get(url, timeout=timeout, follow_redirects=True)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as exc:
            last = exc
            logger.warning("download failed (attempt %d/%d) %s: %s", attempt, retries, url, exc)
    raise RuntimeError(f"failed to download {url} after {retries} attempts") from last


def download_all(force: bool = False, today: dt.date | None = None) -> dict[str, object]:
    """Download every in-scope league-season into ``raw_dir``; write and return the manifest."""
    raw_dir = settings.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    seasons = recent_seasons(settings.seasons_back, today=today)
    files: list[dict[str, object]] = []
    for league in settings.leagues:
        division = DIVISIONS[league]
        for start_year in seasons:
            filename = f"{division}_{season_code(start_year)}.csv"
            destination = raw_dir / filename
            if destination.exists() and not force:
                content = destination.read_bytes()
            else:
                fetched = _download(file_url(division, start_year))
                if fetched is None:
                    logger.info("not published, skipping: %s %s", league, season_code(start_year))
                    continue
                destination.write_bytes(fetched)
                content = fetched
            files.append(
                {
                    "league": league,
                    "division": division,
                    "season": start_year,
                    "url": file_url(division, start_year),
                    "filename": filename,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                }
            )
    manifest: dict[str, object] = {
        "source": "football-data.co.uk",
        "fetched_at": dt.datetime.now(dt.UTC).isoformat(),
        "seasons": seasons,
        "files": files,
    }
    (raw_dir / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def load_raw(path: Path) -> pd.DataFrame:
    """Read one downloaded football-data.co.uk CSV into a DataFrame (raw columns, latin-1)."""
    return pd.read_csv(path, encoding="latin-1")
