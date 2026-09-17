#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Companion to scripts/6r_fig7_revision_top50_plot.py: the same Figure-7-style
bar chart (constant/sine/both vs. baseline, corrected revision MESM data,
1000-seed cache), but for seeds 0-49 - an arbitrary, unselected slice, not
the top50_overall cherry-picked selection - so the two can be compared
side by side to see how much of the top-50 panel's apparent improvement is
selection effect vs. what an equally-sized but unbiased seed sample already
shows. Same seed_agg="median_iqr" and panel/legend/save-path conventions as
6r, so the two figures are visually comparable one-to-one.

Usage:
    python 6s_fig7_revision_first50_plot.py
"""
import os
import sys
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import utils_plotting
from paths import DATA_DIR

SEED_CACHE_PATH = "data/SI_results/seed_uncertainty/fig7_revision_1000seed.pkl"
REV_ICS = ["constant", "sine", "both"]
TOP_PANEL_ICS = ["constant", "sine"]
FIRST_N = 50


def load_top_panel_data():
    """Identical to 6r_fig7_revision_top50_plot.py / 6q_fig7_revision_plot.py."""
    co2_data = []
    for ic in TOP_PANEL_ICS:
        e = np.loadtxt(f"{DATA_DIR}/MESM/emis_driven/MESM_inputs/opt_all_revision_{ic}.txt",
                       usecols=(2,), skiprows=2)
        co2_data.append(e)

    lat_coords = np.linspace(-88, 88, 46)
    lat_weights = np.cos(np.deg2rad(lat_coords))
    lat_weights = np.maximum(lat_weights, 1e-6)
    lat_weights = lat_weights / np.sum(lat_weights)

    global_mean_temp = []
    for ic in TOP_PANEL_ICS:
        with open(f"data/MESM/emis_driven/zonal_data_mean/optimized_revision/"
                  f"opt_all_revision_{ic}_mean.pkl", "rb") as f:
            zonal = pickle.load(f)
        global_mean_temp.append(np.average(zonal, weights=lat_weights, axis=1))

    return co2_data, global_mean_temp


def main():
    with open(SEED_CACHE_PATH, "rb") as f:
        seed_cache = pickle.load(f)
    print(f"loaded {len(seed_cache)}-seed cache")

    seed_idx = list(range(FIRST_N))  # unselected, not ranked by any score
    print(f"using seeds {seed_idx[0]}-{seed_idx[-1]} (arbitrary, unselected)")

    seed_baseline = [seed_cache[s]["baseline"] for s in seed_idx]
    seed_opt_list = [[seed_cache[s][f"optimal_{ic}"] for s in seed_idx] for ic in REV_ICS]

    co2_data, global_mean_temp = load_top_panel_data()

    scenario_keys = ['historical', 'H-ext', 'M', 'ML', 'L', 'VLLO-ext', 'VLHO',
                     'H-ext-OS', 'M-ext', 'ML-ext', 'L-ext', 'VLHO-ext',
                     '2xCO2', '1pctCO2', 'AA', 'CT']
    x_labels = [r'$\it{historical}$', r'$\it{H}$-$\it{ext}$', r'$\it{M}$', r'$\it{ML}$', r'$\it{L}$',
               r'$\it{VLLO}$-$\it{ext}$', r'$\it{VLHO}$', r'$\it{H}$-$\it{ext}$-$\it{OS}$',
               r'$\it{M}$-$\it{ext}$', r'$\it{ML}$-$\it{ext}$', r'$\it{L}$-$\it{ext}$',
               r'$\it{VLHO}$-$\it{ext}$', r'$\it{abrupt}$-$\it{2xCO2}$', r'$\it{1pctCO2}$',
               r'$\it{AA}$', r'$\it{CT}$']
    separator_indices = [6, 11, 13]
    group_labels = ['Priority 1', 'Priority 2', 'DECK', 'CS3']
    legend_labels = ['Const.', 'Sine', 'Both']

    utils_plotting.plot_scenario_difference_bars2(
        baseline_results=seed_baseline[0],
        optimized_results_list=[seed_opt_list[i][0] for i in range(len(REV_ICS))],
        scenario_keys=scenario_keys,
        legend_labels=legend_labels,
        co2_data=co2_data,
        global_mean_temp=global_mean_temp,
        x_labels=x_labels,
        separator_indices=separator_indices,
        group_labels=group_labels,
        save=True,
        figname='fig07_emic_summary_revision_first50',
        seed_baseline_results=seed_baseline,
        seed_optimized_results_list=seed_opt_list,
        seed_agg="median_iqr",
    )
    print(f"saved Figures/fig07_emic_summary_revision_first50.pdf (seeds 0-{FIRST_N-1} of {len(seed_cache)})")


if __name__ == "__main__":
    main()
