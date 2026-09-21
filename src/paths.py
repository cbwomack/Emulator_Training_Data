# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5, Claude Sonnet 5, and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""Repo-root-relative paths, so imports and file I/O work regardless of the caller's cwd."""
from pathlib import Path

# This module lives in src/, so the repo root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "Figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
