#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Computes [25th, 50th, 75th] percentile numbers for supplement_revision.tex's
numerical-results tables: one each for main-text Figures 3, 4, 5, 6 and 7,
plus the SI extended-results tables for CH4/N2O/Sulfur/BC.

Usage:
    python pipeline/07_results/07j_supplement_table_export.py
    python pipeline/07_results/07j_supplement_table_export.py --out data/SI_results/supplement_tables.json
    python pipeline/07_results/07j_supplement_table_export.py --table fig6   # just one table
"""
import os
import sys
import csv
import json
import pickle
import argparse
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import numpy as np

import utils_inverse
import utils_plotting

N_SEEDS = 50


def pctl(values, ndigits=4):
    """[25th, 50th, 75th] percentile, formatted to ndigits decimal places."""
    values = np.asarray(values, dtype=float)
    p25, p50, p75 = np.percentile(values, [25, 50, 75])
    return [round(float(p25), ndigits), round(float(p50), ndigits), round(float(p75), ndigits)]


# ==================================================================
# Figure 3: single-forcing baseline vs. Opt. Priority 1, on Priority 1
# ==================================================================
def compute_fig3_table():
    data = utils_inverse.load_fig3_single_forcing_data_seed_sweep()
    agents = ['CO2', 'CH4', 'N2O', 'Sulfur', 'BC']
    rows = []
    for i, agent in enumerate(agents):
        base_vals = data['seed_baseline_error_list'][i]
        opt_vals = [float(np.asarray(e['errors'])[-1]) for e in data['seed_errors_list'][i]]
        rows.append({"agent": agent, "config": "Baseline", "eval": "Priority 1", "pctl": pctl(base_vals)})
        rows.append({"agent": agent, "config": "Opt. Priority 1", "eval": "Priority 1", "pctl": pctl(opt_vals)})
    return rows


# ==================================================================
# Figure 5: multi-agent baseline vs. Opt. Priority 1, on Priority 1
# ==================================================================
def compute_fig5_table():
    data = utils_inverse.load_fig5_multi_forcing_data_seed_sweep()
    base_vals = data['seed_baseline_errors']
    opt_vals = [float(np.asarray(e['errors'])[-1]) for e in data['seed_errors']]
    return [
        {"agent": "Multi", "config": "Baseline", "eval": "Priority 1", "pctl": pctl(base_vals)},
        {"agent": "Multi", "config": "Opt. Priority 1", "eval": "Priority 1", "pctl": pctl(opt_vals)},
    ]


# ==================================================================
# Figure 4: CO2-only + Multi, full 5x5 optimization-target x eval-set grid
# ==================================================================
_FIG4_TRAIN_LABELS = ['Baseline', 'Opt. Priority 1', 'Opt. Priority 2', 'Opt. DECK', 'Opt. CS3', 'Opt. All']
_FIG4_TRAIN_KEYS = {  # cache['optimal'] keys, None for the baseline row
    'Baseline': None, 'Opt. Priority 1': 'Opt. Tier 1', 'Opt. Priority 2': 'Opt. Tier 2',
    'Opt. DECK': 'Opt. DECK', 'Opt. CS3': 'Opt. CS3', 'Opt. All': 'Opt. All',
}
_FIG4_EVAL_SETS = ['Priority 1', 'Priority 2', 'DECK', 'CS3', 'All']
_FIG4_EVAL_KEYS = {'Priority 1': 'Tier 1', 'Priority 2': 'Tier 2', 'DECK': 'DECK', 'CS3': 'CS3', 'All': 'All'}


def _fig4_half(cache_path, agent_label):
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    seeds = sorted(cache)
    rows = []
    for train_label in _FIG4_TRAIN_LABELS:
        train_key = _FIG4_TRAIN_KEYS[train_label]
        for eval_label in _FIG4_EVAL_SETS:
            eval_key = _FIG4_EVAL_KEYS[eval_label]
            if train_key is None:
                vals = [cache[s]["baseline"][eval_key]["mean"] for s in seeds]
            else:
                vals = [cache[s]["optimal"][train_key][eval_key]["mean"] for s in seeds]
            rows.append({"agent": agent_label, "config": train_label, "eval": eval_label, "pctl": pctl(vals)})
    return rows


def compute_fig4_table():
    co2 = _fig4_half('data/SI_results/seed_uncertainty/fig4_seed_spread_co2_only.pkl', 'CO2')
    multi = _fig4_half('data/SI_results/seed_uncertainty/fig4_seed_spread_all_agents_smooth.pkl', 'Multi')
    return {"co2": co2, "multi": multi}


# ==================================================================
# SI extended-results: CH4/N2O/Sulfur/BC, same 5x5 grid as Figure 4
# ==================================================================
_SI_EXT_CACHE = {
    'CH4': 'data/SI_results/seed_uncertainty/SI_extended_seed_spread_CH4.pkl',
    'N2O': 'data/SI_results/seed_uncertainty/SI_extended_seed_spread_N2O.pkl',
    'Sulfur': 'data/SI_results/seed_uncertainty/SI_extended_seed_spread_Sulfur_smooth.pkl',
    'BC': 'data/SI_results/seed_uncertainty/SI_extended_seed_spread_BC.pkl',
}


def compute_si_extended_table(agent):
    return _fig4_half(_SI_EXT_CACHE[agent], agent)


# ==================================================================
# Figure 6: R^2 forest-plot data (already median+p25+p75 across 50 seeds)
# ==================================================================
def compute_fig6_table(csv_path='data/SI_results/fig6_ood/fig6_ood_r2_table_fig6.csv'):
    with open(csv_path, newline='') as f:
        raw_rows = list(csv.DictReader(f))
    rows = []
    for r in raw_rows:
        rows.append({
            "scenario": r["scenario"],
            "config": r["emulator"],
            "metric": "R2",
            "pctl": [round(float(r["r2_p25"]), 3), round(float(r["r2_median"]), 3), round(float(r["r2_p75"]), 3)],
            "n_seeds": int(r["n_seeds"]),
        })
    return rows


# ==================================================================
# Figure 7: MESM, all scenario groups, over the first 50 seeds unselected -
# the same sample Figure 7 itself aggregates.
# ==================================================================
_FIG7_WEIGHTS = {"Tier 1": 7, "Tier 2": 5, "DECK": 2, "CS3": 2}
_FIG7_CONFIGS = ['baseline', 'optimal_constant', 'optimal_sine', 'optimal_both']
_FIG7_CONFIG_LABELS = {
    'baseline': 'Baseline', 'optimal_constant': 'Opt. Constant',
    'optimal_sine': 'Opt. Sine', 'optimal_both': 'Opt. Both',
}


def _fig7_scenarios(entry: dict) -> list[str]:
    """Scenario names in one evaluation set, excluding the pooled "mean" key."""
    return [k for k in entry if k != "mean"]


def compute_fig7_table(cache_path=utils_plotting.FIG07_SEED_CACHE):
    """Percentiles behind the Figure 7 supplement table.
    """
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    seeds = sorted(cache)[:utils_plotting.FIG07_N_SEEDS]

    eval_sets = ['Tier 1', 'Tier 2', 'DECK', 'CS3']
    rows = []
    for cfg in _FIG7_CONFIGS:
        for eval_set in eval_sets:
            scenarios = _fig7_scenarios(cache[seeds[0]]['baseline'][eval_set])
            vals = []
            for s in seeds:
                base = cache[s]['baseline'][eval_set]
                if cfg == 'baseline':
                    vals.append(np.mean([base[k]['global'] for k in scenarios]))
                else:
                    opt = cache[s][cfg][eval_set]
                    vals.append(np.mean([
                        (base[k]['global'] - opt[k]['global']) / base[k]['global'] * 100.0
                        for k in scenarios]))
            rows.append({"config": _FIG7_CONFIG_LABELS[cfg], "eval": eval_set,
                         "unit": 'NRMSE' if cfg == 'baseline' else 'pct_change',
                         "pctl": pctl(vals)})
    return {
        "n_total_seeds": len(cache), "n_selected": len(seeds), "selection": "first_n",
        "note": (f"Computed over the {len(seeds)} stochastic seeds shown in Figure 7, "
                 "unselected. Baseline rows are NRMSE; optimized rows are percent "
                 "change against the same seed's baseline, positive = lower error."),
        "rows": rows,
    }


TABLES = {
    "fig3": compute_fig3_table,
    "fig4": compute_fig4_table,
    "fig5": compute_fig5_table,
    "fig6": compute_fig6_table,
    "fig7": compute_fig7_table,
    "si_ch4": lambda: compute_si_extended_table('CH4'),
    "si_n2o": lambda: compute_si_extended_table('N2O'),
    "si_sulfur": lambda: compute_si_extended_table('Sulfur'),
    "si_bc": lambda: compute_si_extended_table('BC'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", choices=list(TABLES), default=None,
                         help="Compute only this table (default: all)")
    parser.add_argument("--out", default=None, help="Write results as JSON to this path")
    args = parser.parse_args()

    names = [args.table] if args.table else list(TABLES)
    results = {}
    for name in names:
        print(f"=== {name} ===", flush=True)
        try:
            results[name] = TABLES[name]()
        except FileNotFoundError as e:
            print(f"  SKIPPED (not yet available): {e}")
            results[name] = {"error": str(e)}
            continue
        print(json.dumps(results[name], indent=2)[:4000], flush=True)
        print(flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
