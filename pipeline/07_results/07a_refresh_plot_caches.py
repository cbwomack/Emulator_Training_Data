#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5, Claude Sonnet 5, and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Companion script for pipeline/08_plotting/paper_plots.ipynb: refreshes the two
data-aggregation caches the notebook's Figure 5 (performance summary) and
Figure 6 (individual forcing-agent effects) read.

Usage:
    python 07a_refresh_plot_caches.py             # regenerate both
    python 07a_refresh_plot_caches.py --figure 5  # just Figure 5's cache
    python 07a_refresh_plot_caches.py --figure 6  # just Figure 6's cache
"""
import argparse
import os
import sys
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import utils_inverse


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figure", choices=["5", "6"], default=None,
                         help="Regenerate only this figure's cache (default: regenerate both)")
    args = parser.parse_args()

    if args.figure in (None, "5"):
        utils_inverse.regenerate_fig4_all_agents_cache()
        print("Saved data/plotting/optimal_all_agents_subset.pkl")
    if args.figure in (None, "6"):
        utils_inverse.regenerate_fig6_individual_effects_cache()
        print("Saved data/plotting/y_hat_baseline_ind_effects.pkl, "
              "y_true_ind_effects.pkl, y_hat_ind_effects.pkl")


if __name__ == "__main__":
    main()
