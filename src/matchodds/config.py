"""Project-wide configuration: reproducibility seed, filesystem paths, and modelling constants."""

from pathlib import Path

# Reproducibility — flows into every split, model, and calibrator (see CLAUDE.md → Determinism).
RANDOM_SEED: int = 42

# Filesystem layout, derived from the repo root rather than hard-coded absolute paths.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = REPO_ROOT / "models"

# Rolling-form window: number of prior matches per team used by the form feature (Epic 03).
ROLLING_WINDOW_N: int = 5

# Leagues in scope. Default set; Epic 02 may refine as source coverage is confirmed.
LEAGUES: tuple[str, ...] = (
    "Greek Super League",
    "English Premier League",
    "Spanish La Liga",
    "Italian Serie A",
    "German Bundesliga",
    "French Ligue 1",
)
