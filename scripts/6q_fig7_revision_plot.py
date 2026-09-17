#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Renders the actual Figure-7-style image for the revision comparison built by
scripts/6p_fig7_revision_vector_emulator.py - that script only produces the
seed-spread cache and a text/JSON report, no figure.

Reuses utils_plotting.plot_scenario_difference_bars2 unmodified (same
function/styling the published Figure 7 uses, utils_inverse.
load_fig7_emic_data / 5a_paper_plots.ipynb cell 21), with:

  - the bottom bar panel showing all three revision variants (constant,
    sine, gaussian) vs. baseline, seed-spread error bars from the 50-seed
    sweep (data/SI_results/seed_uncertainty/fig7_revision_seed_spread_MESM.pkl) -
    the function's `optimized_results_list`/`seed_optimized_results_list`
    are generic over how many IC series are plotted, so this needed no
    change to the function;
  - the top two time-series panels showing the constant and sine revision
    emissions + real-MESM global-mean response (the function hardcodes
    exactly two such panels, labeled "(a) ... (const. IC)" / "(b) ...
    (sine IC)" - both labels are literally accurate for the revision ICs
    too, so passing revision constant/sine data here needed no change
    either). Gaussian has no dedicated top panel for the same reason the
    original figure has none for "both": the function only supports two.

Usage:
    python 6q_fig7_revision_plot.py
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

SEED_CACHE_PATH = "data/SI_results/seed_uncertainty/fig7_revision_seed_spread_MESM.pkl"
REV_ICS = ["constant", "sine", "gaussian"]
TOP_PANEL_ICS = ["constant", "sine"]  # the two the function's top panels support


def load_seed_lists():
    with open(SEED_CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    seeds = sorted(cache.keys())
    seed_baseline = [cache[s]["baseline"] for s in seeds]
    seed_opt = {ic: [cache[s][f"optimal_{ic}"] for s in seeds] for ic in REV_ICS}
    return seeds, seed_baseline, seed_opt


def load_top_panel_data():
    """(co2_data, global_mean_temp), each a 2-element list (constant, sine),
    same construction as utils_inverse.load_fig7_emic_data: emissions read
    straight from the driver file, MESM truth area-weighted with cos(lat)
    on the synthetic linspace(-88,88,46) grid (not the files' own `gw`) so
    this matches that function's own weighting convention exactly."""
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
    seeds, seed_baseline, seed_opt = load_seed_lists()
    co2_data, global_mean_temp = load_top_panel_data()

    # Same 16-scenario set/order/grouping as the published Figure 7
    # (utils_inverse.load_fig7_emic_data) - identical eval sets, so this
    # revision figure is directly comparable panel-for-panel.
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
    legend_labels = ['Const.', 'Sine', 'Gaussian']

    utils_plotting.plot_scenario_difference_bars2(
        baseline_results=seed_baseline[0],
        optimized_results_list=[seed_opt[ic][0] for ic in REV_ICS],
        scenario_keys=scenario_keys,
        legend_labels=legend_labels,
        co2_data=co2_data,
        global_mean_temp=global_mean_temp,
        x_labels=x_labels,
        separator_indices=separator_indices,
        group_labels=group_labels,
        save=True,
        figname='fig07_emic_summary_revision',
        seed_baseline_results=seed_baseline,
        seed_optimized_results_list=[seed_opt[ic] for ic in REV_ICS],
    )
    print(f"saved Figures/fig07_emic_summary_revision.pdf ({len(seeds)} seeds)")


if __name__ == "__main__":
    main()
