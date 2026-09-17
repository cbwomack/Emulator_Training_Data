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
effects_cache.

Reads the SMOOTHED families (checkpoints/multi_fig4_smooth/ +
checkpoints/multi_retuned_smooth/) and the CONSTANT-initial-condition
DAMIP/GeoMIP/all checkpoints. Previously it read the unsmoothed families with
init_cond='sine' - a sinusoid centred on zero, i.e. 250-375 years of negative
emissions, undefined for Sulfur and BC. That belongs to the SI initial-condition
sensitivity sweep, not a main-paper figure (REVISIONS.md, 2026-08-27). Before
that it read the deprecated checkpoints/multi/*_subset* family (2026-08-14).

This writes the SINGLE-SEED cache the figure needs for its ground-truth line and
its fallbacks. Figure 6's error bars come from a separate 50-seed cache built by
scripts/build_fig6_ind_effects_cache.py; both are required.

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
