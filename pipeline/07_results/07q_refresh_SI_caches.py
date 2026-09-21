#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5, Claude Sonnet 5, and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Companion script for pipeline/08_plotting/SI_plots.ipynb: refreshes the
single-agent (N2O/Sulfur/BC) optimal-emulator NRMSE cache
(data/SI_results/extended_results/optimal_<agent>_only.pkl) that the SI
extended-results stacked-bar figure reads. 

Usage:
    python SI_plots.py
"""
import os
import sys
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import utils_inverse


def main():
    utils_inverse.regenerate_SI_extended_results_cache()
    print("Saved data/SI_results/extended_results/optimal_{n2o,Sulfur,BC}_only.pkl")


if __name__ == "__main__":
    main()
