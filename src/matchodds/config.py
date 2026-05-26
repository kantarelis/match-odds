"""Project-wide configuration via pydantic-settings.

``settings`` is the single typed config object. Every field is overridable through a
``MATCHODDS_``-prefixed environment variable (e.g. ``MATCHODDS_DATA_DIR=/tmp/x``) — this is how CI
points the data notebook at committed sample fixtures without touching code.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Default leagues in scope: Greek Super League + the top-5 EU leagues.
_DEFAULT_LEAGUES: tuple[str, ...] = (
    "Greek Super League",
    "English Premier League",
    "Spanish La Liga",
    "Italian Serie A",
    "German Bundesliga",
    "French Ligue 1",
)


class Settings(BaseSettings):
    """Typed, environment-overridable project configuration (env prefix ``MATCHODDS_``)."""

    model_config = SettingsConfigDict(env_prefix="MATCHODDS_", extra="ignore")

    # Reproducibility — flows into every split, model, and calibrator (see CLAUDE.md → Determinism).
    random_seed: int = 42

    # Feature pipeline (Epic 03): rolling-form window, in matches per team.
    rolling_window_n: int = 5

    # Elo (Epic 03): starting rating, K-factor, and home-advantage bonus (rating points).
    elo_base: float = 1500.0
    elo_k: float = 20.0
    elo_home_advantage: float = 65.0

    # Modeling (Epic 04): number of forward-chaining temporal CV folds.
    cv_splits: int = 5

    # Dixon-Coles (Epic 04): goals summed per side when reading 1X2 off the score matrix, and the
    # optional exponential time-decay half-life in days (``None`` disables recency weighting).
    dc_max_goals: int = 10
    dc_half_life_days: int | None = None

    # Data scope.
    leagues: tuple[str, ...] = _DEFAULT_LEAGUES
    seasons_back: int = 10

    # Public data source (football-data.co.uk).
    football_data_base_url: str = "https://www.football-data.co.uk/mmz4281/"

    # Serving (Epic 05): uvicorn bind host/port and the runtime environment label reported by /env.
    serve_host: str = "127.0.0.1"
    serve_port: int = 8000
    environment: str = "local"

    # Filesystem layout. ``data_dir`` is the env-overridable root; raw/processed derive from it.
    repo_root: Path = _REPO_ROOT
    data_dir: Path = _REPO_ROOT / "data"
    models_dir: Path = _REPO_ROOT / "models"

    @property
    def raw_dir(self) -> Path:
        """Raw downloaded CSVs (gitignored)."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Cleaned master matches table and feature tables (gitignored)."""
        return self.data_dir / "processed"


settings = Settings()
