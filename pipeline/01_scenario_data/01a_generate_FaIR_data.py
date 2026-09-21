#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5, Claude Sonnet 5, and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Companion script for 01a_generate_FaIR_data.ipynb: builds emissions + FaIR
temperature-response data for all 4 scenario sets (ScenarioMIP tier1/tier2,
DECK, GeoMIP, CS3) and pickles the result to data/FaIR_IO/scenario_data.pkl.
Runs standalone; the
notebook itself only loads this file and plots from it.

Usage:
    python 01a_generate_FaIR_data.py
    python 01a_generate_FaIR_data.py --save-path data/FaIR_IO/scenario_data.pkl
"""
import os
import sys
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import argparse

import run_fair

DEFAULT_SAVE_PATH = "data/FaIR_IO/scenario_data.pkl"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH,
                         help=f"Where to save the generated scenario data (default: {DEFAULT_SAVE_PATH})")
    args = parser.parse_args()

    run_fair.generate_and_save_scenario_data(save_path=args.save_path)
    print(f"Saved scenario data to {args.save_path}")


if __name__ == "__main__":
    main()
