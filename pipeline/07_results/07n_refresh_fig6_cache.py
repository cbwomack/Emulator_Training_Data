#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Refreshes Figure 6's individual-effects cache (data/plotting/{y_hat_baseline,
y_true,y_hat}_ind_effects.pkl) via utils_inverse.regenerate_fig6_individual_
effects_cache.

Usage:
    python pipeline/07_results/07n_07n_refresh_fig6_cache.py
"""
import os
import sys
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import utils_inverse

if __name__ == "__main__":
    utils_inverse.regenerate_fig6_individual_effects_cache()
    print("done -> data/plotting/{y_hat_baseline,y_true,y_hat}_ind_effects.pkl")
