#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Aggregation diagnostics for Figures 3 and 4.

Usage:
    python pipeline/07_results/07g_aggregation_statistics.py                # both figures
    python pipeline/07_results/07g_aggregation_statistics.py --figure 3
    python pipeline/07_results/07g_aggregation_statistics.py --figure 4
    python pipeline/07_results/07g_aggregation_statistics.py --seeds 10     # quick pass
"""
import argparse
import csv
import os
import pickle
import sys
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import numpy as np

import utils_inverse

OUT_DIR = Path("data/SI_results/aggregation_stats")

AGENT_ORDER = ["co2", "ch4", "n2o", "Sulfur", "BC"]
AGENT_LABELS = {"co2": "CO2", "ch4": "CH4", "n2o": "N2O", "Sulfur": "Sulfur", "BC": "BC"}

TRAIN_SCENARIOS = ["Opt. Tier 1", "Opt. Tier 2", "Opt. DECK", "Opt. CS3", "Opt. All"]
TEST_SCENARIOS = ["Tier 1", "Tier 2", "DECK", "CS3"]
WEIGHTS = [7, 5, 2, 2]
PANELS = {
    "co2_only": "data/SI_results/seed_uncertainty/fig4_seed_spread_co2_only.pkl",
    "all_agents": "data/SI_results/seed_uncertainty/fig4_seed_spread_all_agents_smooth.pkl",
}

EARLY_LATE_SPLIT = 20


# ---------------------------------------------------------------------------
# Figure 3: trajectory aggregation and bump census
# ---------------------------------------------------------------------------

def _bump_census(curve):
    """Count and size the rises in one aggregate trajectory.
    """
    curve = np.asarray(curve, dtype=np.float64)
    step = np.diff(curve)
    rel = step / curve[:-1]

    def _slice(sl):
        s, r = step[sl], rel[sl]
        rising = s > 0
        return {
            "n_rises": int(rising.sum()),
            "total_rise": float(s[rising].sum()) if rising.any() else 0.0,
            "max_rel_bump_pct": float(r.max() * 100) if len(r) else 0.0,
            "argmax_iteration": int(np.argmax(r)) + (sl.start or 0) if len(r) else -1,
        }

    return {
        "all": _slice(slice(None)),
        "early": _slice(slice(0, EARLY_LATE_SPLIT)),
        "late": _slice(slice(EARLY_LATE_SPLIT, None)),
        "final": float(curve[-1]),
    }


def _aggregations(stacked):
    """Every candidate aggregation of a (n_seeds, n_iters) trajectory stack."""
    return {
        "mean": stacked.mean(axis=0),
        "median": np.median(stacked, axis=0),
        "p10": np.percentile(stacked, 10, axis=0),
        "p25": np.percentile(stacked, 25, axis=0),
        "p75": np.percentile(stacked, 75, axis=0),
        "p90": np.percentile(stacked, 90, axis=0),
        "min": stacked.min(axis=0),
        "max": stacked.max(axis=0),
    }


def _attribute_largest_late_bump(stacked, mean_curve):
    """How concentrated is the mean's worst late bump across seeds?
    """
    step = np.diff(mean_curve)
    if len(step) <= EARLY_LATE_SPLIT:
        return None
    k = int(np.argmax(step[EARLY_LATE_SPLIT:])) + EARLY_LATE_SPLIT
    per_seed = np.diff(stacked, axis=1)[:, k]
    total = float(step[k])
    n_seeds = stacked.shape[0]
    return {
        "iteration": k,
        "mean_step": total,
        "n_seeds_rising": int((per_seed > 0).sum()),
        "n_seeds": n_seeds,
        "top_seed_share": float(per_seed.max() / n_seeds / total) if total > 0 else float("nan"),
    }


def analyze_penalty_contamination(seeds):
    """How much of Figure 3's apparent instability was the smoothness penalty?
    """
    rows = []
    print("\n=== Was the apparent instability real error, or the penalty? ===")
    print(f"{'agent':8s} {'w':>8s} | {'raw maxbump%':>13s} {'corr maxbump%':>14s} | "
          f"{'raw rises':>10s} {'corr rises':>11s}")
    for agent_lower in AGENT_ORDER:
        tag = utils_inverse._FIG3_AGENT_TAGS[agent_lower]
        raw_stack, corr_stack, weight = [], [], None
        for seed in seeds:
            ckpt_dir = utils_inverse._FIG3_AGENT_DIRS.get(
                agent_lower, f"checkpoints/{agent_lower}_retuned/seed_sweep")
            path = f"{ckpt_dir}/inverse_constant_tier1_{tag}_seed{seed}.pkl"
            with open(path, "rb") as f:
                raw = pickle.load(f)
            if weight is None:
                weight, _ = utils_inverse.recover_smoothness_weight(raw)
            raw_stack.append(np.asarray(raw["errors"], dtype=np.float64))
            corr_stack.append(utils_inverse.recover_nrmse_trajectory(raw))

        raw_c = _bump_census(np.stack(raw_stack).mean(axis=0))
        corr_c = _bump_census(np.stack(corr_stack).mean(axis=0))
        rows.append({
            "agent": AGENT_LABELS[agent_lower], "smoothness_weight": weight,
            "raw_max_rel_bump_pct": round(raw_c["all"]["max_rel_bump_pct"], 3),
            "corrected_max_rel_bump_pct": round(corr_c["all"]["max_rel_bump_pct"], 3),
            "raw_n_rises": raw_c["all"]["n_rises"], "corrected_n_rises": corr_c["all"]["n_rises"],
            "raw_final": round(raw_c["final"], 6), "corrected_final": round(corr_c["final"], 6),
        })
        print(f"{AGENT_LABELS[agent_lower]:8s} {weight:8.1g} | "
              f"{raw_c['all']['max_rel_bump_pct']:13.2f} {corr_c['all']['max_rel_bump_pct']:14.2f} | "
              f"{raw_c['all']['n_rises']:10d} {corr_c['all']['n_rises']:11d}")
    return rows


def analyze_figure3(seeds):
    data = utils_inverse.load_fig3_single_forcing_data_seed_sweep(seeds=seeds)
    results, rows = {}, []

    for agent_lower, errs in zip(AGENT_ORDER, data["seed_errors_list"]):
        agent = AGENT_LABELS[agent_lower]
        stacked = np.stack([np.asarray(e["errors"], dtype=np.float64) for e in errs], axis=0)
        aggs = _aggregations(stacked)
        census = {name: _bump_census(curve) for name, curve in aggs.items()}

        per_seed_rises = [int((np.diff(stacked[i]) > 0).sum()) for i in range(stacked.shape[0])]
        attribution = _attribute_largest_late_bump(stacked, aggs["mean"])

        results[agent] = {
            "aggregations": aggs, "census": census,
            "per_seed_rises": per_seed_rises, "late_bump_attribution": attribution,
            "n_seeds": stacked.shape[0], "n_iters": stacked.shape[1],
        }

        for name in ("mean", "median"):
            c = census[name]
            rows.append({
                "agent": agent, "aggregation": name,
                "final": round(c["final"], 6),
                "n_rises_total": c["all"]["n_rises"],
                "n_rises_early": c["early"]["n_rises"],
                "n_rises_late": c["late"]["n_rises"],
                "total_rise": round(c["all"]["total_rise"], 6),
                "max_rel_bump_pct": round(c["all"]["max_rel_bump_pct"], 3),
                "max_bump_iteration": c["all"]["argmax_iteration"],
                "max_rel_bump_late_pct": round(c["late"]["max_rel_bump_pct"], 3),
            })

        print(f"\n[{agent}] {stacked.shape[0]} seeds x {stacked.shape[1]} iters")
        for name in ("mean", "median"):
            c = census[name]
            print(f"   {name:6s} final={c['final']:.6f}  rises: "
                  f"{c['all']['n_rises']:4d} total ({c['early']['n_rises']:3d} early, "
                  f"{c['late']['n_rises']:4d} late)  "
                  f"max bump {c['all']['max_rel_bump_pct']:+6.1f}% @it{c['all']['argmax_iteration']}"
                  f"  late-only {c['late']['max_rel_bump_pct']:+6.2f}%")
        print(f"   per-seed rises: median={int(np.median(per_seed_rises))} "
              f"min={min(per_seed_rises)} max={max(per_seed_rises)}")
        if attribution:
            print(f"   largest late bump in MEAN @it{attribution['iteration']}: "
                  f"{attribution['n_seeds_rising']}/{attribution['n_seeds']} seeds rising, "
                  f"worst seed supplies {attribution['top_seed_share'] * 100:.0f}% of it")

    return results, rows


# ---------------------------------------------------------------------------
# Figure 4: how the per-cell number depends on the aggregation and the metric
# ---------------------------------------------------------------------------

def _cell_values(seed_cache, seeds, train, test):
    """Per-seed (baseline, optimized) mean-NRMSE pair for one figure cell."""
    base = np.array([float(seed_cache[s]["baseline"][test]["mean"]) for s in seeds])
    opt = np.array([float(seed_cache[s]["optimal"][train][test]["mean"]) for s in seeds])
    return base, opt


def _weighted_avg_cell(seed_cache, seeds, train):
    """The 'Avg.' bar: scenario-count-weighted mean across the four eval sets,
    matching plot_grouped_improvement_bars' np.average(..., weights=WEIGHTS)."""
    w = np.array(WEIGHTS, dtype=np.float64)
    base = np.array([
        np.average([float(seed_cache[s]["baseline"][t]["mean"]) for t in TEST_SCENARIOS], weights=w)
        for s in seeds])
    opt = np.array([
        np.average([float(seed_cache[s]["optimal"][train][t]["mean"]) for t in TEST_SCENARIOS], weights=w)
        for s in seeds])
    return base, opt


def _metrics(base, opt):
    """Aggregations and metrics for one cell, from per-seed (base, opt) pairs.
    """
    ratios = (base - opt) / base * 100.0
    return {
        "mean_of_ratios": float(ratios.mean()),
        "median_of_ratios": float(np.median(ratios)),
        "ratio_of_medians": float((np.median(base) - np.median(opt)) / np.median(base) * 100.0),
        "std_of_ratios": float(ratios.std()),
        "iqr_of_ratios": float(np.percentile(ratios, 75) - np.percentile(ratios, 25)),
        "n_negative": int((ratios < 0).sum()),
        "n_seeds": int(len(ratios)),
        "median_baseline_nrmse": float(np.median(base)),
        "median_optimized_nrmse": float(np.median(opt)),
        "median_abs_diff": float(np.median(base - opt)),
        "median_log_ratio": float(np.median(np.log(opt / base))),
    }


def analyze_figure4(seeds):
    results, rows = {}, []

    for panel, path in PANELS.items():
        with open(path, "rb") as f:
            cache = pickle.load(f)
        available = [s for s in seeds if s in cache]
        results[panel] = {"cells": {}, "n_seeds": len(available)}
        print(f"\n=== Figure 4 panel: {panel} ({len(available)} seeds) ===")
        print(f"{'train':13s} {'test':8s} {'mean%':>8s} {'median%':>8s} {'ratioMed%':>10s} "
              f"{'IQR':>8s} {'neg':>5s} {'medAbsDiff':>11s}")

        for train in TRAIN_SCENARIOS:
            for test in TEST_SCENARIOS + ["Avg."]:
                if test == "Avg.":
                    base, opt = _weighted_avg_cell(cache, available, train)
                else:
                    base, opt = _cell_values(cache, available, train, test)
                m = _metrics(base, opt)
                results[panel]["cells"][(train, test)] = m
                rows.append({"panel": panel, "train_scenario": train, "test_scenario": test,
                             **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
                flag = "  <-- on-diagonal" if train == f"Opt. {test}" else ""
                print(f"{train:13s} {test:8s} {m['mean_of_ratios']:8.1f} {m['median_of_ratios']:8.1f} "
                      f"{m['ratio_of_medians']:10.1f} {m['iqr_of_ratios']:8.1f} "
                      f"{m['n_negative']:3d}/{m['n_seeds']:<2d} {m['median_abs_diff']:11.5f}{flag}")

    return results, rows


def analyze_priority1_decomposition(seeds):
    """Per-scenario breakdown of the surprising 'Opt. Priority 1 -> Priority 1' cell.
    """
    rows, results = [], {}
    for panel, path in PANELS.items():
        with open(path, "rb") as f:
            cache = pickle.load(f)
        available = [s for s in seeds if s in cache]
        scens = [k for k in cache[available[0]]["baseline"]["Tier 1"] if k != "mean"]

        print(f"\n=== 'Opt. Priority 1' -> Priority 1, per scenario ({panel}) ===")
        print(f"{'scenario':12s} {'medBase':>9s} {'medOpt':>9s} {'median%':>9s} {'mean%':>9s} {'medAbsDiff':>11s}")
        results[panel] = {}
        for scen in scens:
            base = np.array([float(cache[s]["baseline"]["Tier 1"][scen]) for s in available])
            opt = np.array([float(cache[s]["optimal"]["Opt. Tier 1"]["Tier 1"][scen]) for s in available])
            m = _metrics(base, opt)
            results[panel][scen] = m
            rows.append({"panel": panel, "scenario": scen,
                         **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}})
            print(f"{scen:12s} {m['median_baseline_nrmse']:9.4f} {m['median_optimized_nrmse']:9.4f} "
                  f"{m['median_of_ratios']:9.1f} {m['mean_of_ratios']:9.1f} {m['median_abs_diff']:11.5f}")
    return results, rows


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", choices=["3", "4", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=50, help="use seeds 0..N-1 (default 50)")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.figure in ("3", "both"):
        res3, rows3 = analyze_figure3(seeds)
        rows_contam = analyze_penalty_contamination(seeds)
        with open(OUT_DIR / "fig3_aggregation.pkl", "wb") as f:
            pickle.dump(res3, f)
        _write_csv(OUT_DIR / "fig3_aggregation.csv", rows3)
        _write_csv(OUT_DIR / "fig3_penalty_contamination.csv", rows_contam)

    if args.figure in ("4", "both"):
        res4, rows4 = analyze_figure4(seeds)
        resP, rowsP = analyze_priority1_decomposition(seeds)
        with open(OUT_DIR / "fig4_aggregation.pkl", "wb") as f:
            pickle.dump({"cells": res4, "priority1_decomposition": resP}, f)
        _write_csv(OUT_DIR / "fig4_aggregation.csv", rows4)
        _write_csv(OUT_DIR / "fig4_priority1_decomposition.csv", rowsP)


if __name__ == "__main__":
    main()
