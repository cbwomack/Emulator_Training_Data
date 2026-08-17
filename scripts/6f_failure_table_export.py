#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage 6f - Point 6 (failure-value transparency table): exports every raw
%-improvement value utils_plotting.plot_grouped_improvement_bars clips its
y-axis at ([-60, 100], set by plot_vertical_stacked_bars) and hatches below
(<=-60) for SI Fig 6 (the 5-panel single-forcing summary, CO2/CH4/N2O/
Sulfur/BC), as a CSV/markdown table with mean +/- std spread across seeds.

Reuses the exact same %-improvement formula as plot_grouped_improvement_bars
(reproduced here since that function computes it as a closure, not exposed
externally) - see _pct_improvement_for below, copied verbatim from
utils_plotting.py:1126-1148.

Covers all 5 single-forcing agents (SI Fig 6) plus Figure 4's multi-agent
panel (added 2026-08-14, once checkpoints/multi_fig4/'s regeneration and
data/SI_results/seed_uncertainty/fig4_seed_spread_all_agents.pkl landed -
see REVISIONS.md Session Log). The multi-agent cache shares the exact same
{seed: {"baseline":..., "optimal":...}} shape and TRAIN_SCENARIOS/
TEST_SCENARIOS naming as every single-forcing agent's cache, so no separate
formula/table logic is needed - it's just one more row group.

Usage:
    python 6f_failure_table_export.py
"""
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import numpy as np

import utils_inverse

OUT_DIR = Path("data/SI_results/failure_table")
MULTI_AGENT_SEED_SPREAD_PATH = Path("data/SI_results/seed_uncertainty/fig4_seed_spread_all_agents.pkl")

AGENT_LABELS = {"co2": "CO2", "ch4": "CH4", "n2o": "N2O", "Sulfur": "Sulfur", "BC": "BC"}
AGENT_ORDER = ["co2", "ch4", "n2o", "Sulfur", "BC"]

TRAIN_SCENARIOS = ["Opt. Tier 1", "Opt. Tier 2", "Opt. DECK", "Opt. CS3", "Opt. All"]
TEST_SCENARIOS = ["Tier 1", "Tier 2", "DECK", "CS3"]
LEG_LABELS = ["Priority 1", "Priority 2", "DECK", "CS3"]
WEIGHTS = [7, 5, 2, 2]

HATCH_LIMIT = -60
YLIM = (-60, 100)


def _pct_improvement_for(baseline_res, optimized_res):
    """Verbatim copy of utils_plotting.plot_grouped_improvement_bars's
    internal _pct_improvement_for closure (utils_plotting.py:1126-1148) -
    not exposed externally, so reproduced here rather than modifying that
    (tested, in-production) plotting code to expose it."""
    n_test = len(TEST_SCENARIOS)
    n_opts = len(TRAIN_SCENARIOS)
    w_arr = np.array(WEIGHTS)

    base_errors = np.zeros(n_test)
    opt_errors = np.zeros((n_test, n_opts))

    for i, test_key in enumerate(TEST_SCENARIOS):
        try:
            base_errors[i] = baseline_res[test_key]["mean"]
        except KeyError:
            base_errors[i] = np.nan

    for j, train_key in enumerate(TRAIN_SCENARIOS):
        for i, test_key in enumerate(TEST_SCENARIOS):
            try:
                opt_errors[i, j] = optimized_res[train_key][test_key]["mean"]
            except KeyError:
                opt_errors[i, j] = np.nan

    avg_base_error = np.average(base_errors, weights=w_arr)
    avg_opt_errors = np.average(opt_errors, axis=0, weights=w_arr)

    pct_improvement = (base_errors[:, None] - opt_errors) / base_errors[:, None]
    avg_improvement = (avg_base_error - avg_opt_errors) / avg_base_error

    return np.vstack([pct_improvement, avg_improvement]) * 100


def compute_agent_table(seed_baseline_results, seed_optimized_results):
    """Returns (mean, std) arrays, shape (len(LEG_LABELS)+1, len(TRAIN_SCENARIOS))."""
    n_seeds = len(seed_baseline_results)
    stacked = np.stack([
        _pct_improvement_for(seed_baseline_results[s], seed_optimized_results[s])
        for s in range(n_seeds)
    ], axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def main():
    data = utils_inverse.load_SI_extended_results_data_seed_sweep(agent_lower_list=AGENT_ORDER)
    row_labels = LEG_LABELS + ["Avg."]

    rows = []
    for agent_idx, agent_lower in enumerate(AGENT_ORDER):
        agent = AGENT_LABELS[agent_lower]
        seed_base = data["seed_baseline_results_list"][agent_idx]
        seed_opt = data["seed_optimized_results_list"][agent_idx]
        mean, std = compute_agent_table(seed_base, seed_opt)

        for i, row_label in enumerate(row_labels):
            for j, train_scen in enumerate(TRAIN_SCENARIOS):
                val_mean, val_std = float(mean[i, j]), float(std[i, j])
                hatched = val_mean <= HATCH_LIMIT
                clipped = val_mean < YLIM[0] or val_mean > YLIM[1]
                rows.append({
                    "agent": agent, "test_scenario": row_label, "train_scenario": train_scen,
                    "pct_improvement_mean": round(val_mean, 2), "pct_improvement_std": round(val_std, 2),
                    "hatched_in_figure": hatched, "clipped_by_ylim": clipped,
                })
                flag = " [HATCHED]" if hatched else (" [CLIPPED]" if clipped else "")
                print(f"[{agent}] {row_label:12s} x {train_scen:14s}: "
                      f"{val_mean:+7.2f} +/- {val_std:5.2f}%{flag}")

    # Figure 4's multi-agent panel - same shape/scenario-naming, one more row group.
    with open(MULTI_AGENT_SEED_SPREAD_PATH, "rb") as f:
        multi_seed_spread = pickle.load(f)
    seed_base_multi = [multi_seed_spread[s]["baseline"] for s in sorted(multi_seed_spread)]
    seed_opt_multi = [multi_seed_spread[s]["optimal"] for s in sorted(multi_seed_spread)]
    mean, std = compute_agent_table(seed_base_multi, seed_opt_multi)

    for i, row_label in enumerate(row_labels):
        for j, train_scen in enumerate(TRAIN_SCENARIOS):
            val_mean, val_std = float(mean[i, j]), float(std[i, j])
            hatched = val_mean <= HATCH_LIMIT
            clipped = val_mean < YLIM[0] or val_mean > YLIM[1]
            rows.append({
                "agent": "Multi-agent", "test_scenario": row_label, "train_scenario": train_scen,
                "pct_improvement_mean": round(val_mean, 2), "pct_improvement_std": round(val_std, 2),
                "hatched_in_figure": hatched, "clipped_by_ylim": clipped,
            })
            flag = " [HATCHED]" if hatched else (" [CLIPPED]" if clipped else "")
            print(f"[Multi-agent] {row_label:12s} x {train_scen:14s}: "
                  f"{val_mean:+7.2f} +/- {val_std:5.2f}%{flag}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "fig4_SI_fig6_failure_table.csv"
    fieldnames = ["agent", "test_scenario", "train_scenario", "pct_improvement_mean",
                  "pct_improvement_std", "hatched_in_figure", "clipped_by_ylim"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {csv_path}")

    n_hatched = sum(1 for r in rows if r["hatched_in_figure"])
    n_clipped = sum(1 for r in rows if r["clipped_by_ylim"])
    print(f"{n_hatched} rows hatched in the figure (<=-{-HATCH_LIMIT}%), "
          f"{n_clipped} rows outside the y-axis view {YLIM}")


if __name__ == "__main__":
    main()
