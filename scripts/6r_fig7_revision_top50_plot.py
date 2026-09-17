#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Remakes the Figure-7-style bar chart restricted to the top-50-overall seeds
(of the 1000-seed sweep on the *corrected* revision MESM data - scripts/
6p_fig7_revision_vector_emulator.py / submit_fig7_revision_seed_sweep.py),
following the exact seed-selection protocol from 6m_fig7_seed_selection_
variants.ipynb: each seed's weighted-overall NRMSE (Tier1:7/Tier2:5/DECK:2/
CS3:2) is computed per optimized variant, "top50_overall" is the 50 seeds
with the lowest *mean* of the three optimized variants' scores (constant,
sine, both - no gaussian, matching this session's cluster runs), and the
same 50 seed indices pull baseline/constant/sine/both for those seeds too -
not an independently-best-50 baseline.

Uses utils_plotting.plot_scenario_difference_bars2 unmodified, with
seed_agg="median_iqr" (median + IQR error bars), matching 6m's convention
for seed-selected panels - the full-1000, non-selected figure
(scripts/6q_fig7_revision_plot.py) uses the default mean+/-std instead,
since that's the convention the existing published Figure 7 uses.

Top panels (a)/(b) reuse scripts/6q_fig7_revision_plot.py's construction:
the revision constant/sine emissions + real-MESM global-mean response (not
select-seed-restricted - those are the actual MESM ground truth, not per-
seed emulator results).

Usage:
    python 6r_fig7_revision_top50_plot.py
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
TOP_PANEL_ICS = ["constant", "sine"]  # the two the function's top panels support

WEIGHTS = {"Tier 1": 7, "Tier 2": 5, "DECK": 2, "CS3": 2}
W_SUM = sum(WEIGHTS.values())


def overall_score(seed_cache, seed, variant):
    entry = seed_cache[seed][variant]
    return sum(WEIGHTS[s] * entry[s]["mean"]["global"] for s in WEIGHTS) / W_SUM


def select_top50_overall(seed_cache):
    """Same protocol as 6m_fig7_seed_selection_variants.ipynb cell 3:
    top 50 seeds by the mean of the three optimized variants' own weighted
    scores (constant/sine/both - no gaussian here)."""
    n_seeds = len(seed_cache)
    scores = {v: np.array([overall_score(seed_cache, s, f"optimal_{v}") for s in range(n_seeds)])
             for v in REV_ICS}
    combined = np.mean([scores[v] for v in REV_ICS], axis=0)
    idx = list(np.argsort(combined)[:50])
    print(f"top50_overall: criterion range [{combined[np.array(idx)].min():.4f}, "
          f"{combined[np.array(idx)].max():.4f}] (full-1000 median={np.median(combined):.4f})")
    return idx


def load_top_panel_data():
    """(co2_data, global_mean_temp), each a 2-element list (constant, sine) -
    identical to scripts/6q_fig7_revision_plot.py's construction. Not seed-
    selected: this is the real MESM ensemble output, one fixed trajectory
    per IC, not a per-seed emulator result."""
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

    top50_idx = select_top50_overall(seed_cache)

    seed_baseline = [seed_cache[s]["baseline"] for s in top50_idx]
    seed_opt_list = [[seed_cache[s][f"optimal_{ic}"] for s in top50_idx] for ic in REV_ICS]

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
        figname='fig07_emic_summary_revision_top50_overall',
        seed_baseline_results=seed_baseline,
        seed_optimized_results_list=seed_opt_list,
        seed_agg="median_iqr",
    )
    print(f"saved Figures/fig07_emic_summary_revision_top50_overall.pdf (top-50-of-{len(seed_cache)} seeds)")


if __name__ == "__main__":
    main()
