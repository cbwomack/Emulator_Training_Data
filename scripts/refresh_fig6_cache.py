#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Refreshes Figure 6's individual-effects cache (data/plotting/{y_hat_baseline,
y_true,y_hat}_ind_effects.pkl) via utils_inverse.regenerate_fig6_individual_
effects_cache, now pointed at checkpoints/multi_fig4/ + checkpoints/
multi_retuned/ instead of the deprecated checkpoints/multi/*_subset* family
(see REVISIONS.md, 2026-08-14).

Usage:
    python scripts/refresh_fig6_cache.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import utils_inverse

if __name__ == "__main__":
    utils_inverse.regenerate_fig6_individual_effects_cache()
    print("done -> data/plotting/{y_hat_baseline,y_true,y_hat}_ind_effects.pkl")
