"""Repo-root-relative paths, so imports and file I/O work regardless of the caller's cwd."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "Figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
