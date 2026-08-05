#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Phase 0, Stage 6e Part C: does the manuscript's reported optimized-emulator-
vs-baseline improvement gap survive once the baseline is independently
tuned (Part B), rather than left at the copied-from-the-optimized-emulator
hyperparameters (K=400, lr=0.05, weight_decay=0.01)?

Runs BOTH baseline configs - "original" (copied) and "tuned" (Part B's
winning lr/weight_decay, K held at 400 for apples-to-apples training
budget with the optimized emulator's fixed K_inner=400 - see
6e_baseline_hp_search.py's module docstring) - across the same 5 seeds,
via evaluate_baseline_over_multiple_tests on the real CO2-only eval_sets
(Tier 1/Tier 2/DECK/CS3/All). Compares each against the optimized
emulator's already-finalized Stage 0b unified-config validate scores
(data/SI_results/hp_retune/best_config_unified.json - 10-seed means, same
NRMSE metric, same underlying scenario sets: build_group_emis_dicts maps
Stage 0b's group names 1:1 onto these eval_sets keys, confirmed directly
in utils_inverse.py).

Usage:
    python 6e_gap_persistence.py --mode run-one --config original --seed 0
    python 6e_gap_persistence.py --mode collect
    python 6e_gap_persistence.py --mode report
"""
import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import jax

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import utils_inverse

OUT_DIR = Path("data/SI_results/baseline_hp/gap_persistence")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGENTS = ["CO2"]
ACTIVE_AGENTS = ("CO2",)
MODE = "FaIR"
HIDDEN_SIZES = [16]
SEEDS = [0, 1, 2, 3, 4]

BASELINE_CONFIGS = {
    "original": {"K": 400, "lr": 0.05, "weight_decay": 0.01},
    "tuned": {"K": 400, "lr": 0.13292958927329326, "weight_decay": 0.0},
}

# eval_sets key -> Stage 0b group name, for pulling the matching optimized-emulator score
EVAL_TO_GROUP = {"Tier 1": "tier1", "Tier 2": "tier2", "DECK": "DECK", "CS3": "CS3", "All": "all"}
OPTIMIZED_SCORES_PATH = Path("data/SI_results/hp_retune/best_config_unified.json")

_SETUP_CACHE = {}


def get_setup():
    if "setup" not in _SETUP_CACHE:
        setup = utils_inverse.run_inverse_experiment_setup(
            AGENTS, ACTIVE_AGENTS, mode=MODE, CS3=True, DAMIP=False, GeoMIP=False,
            seed=0, idx_demo=None,
        )
        _SETUP_CACHE["setup"] = setup
    return _SETUP_CACHE["setup"]


def _result_path(config_name, seed):
    return OUT_DIR / f"gap_{config_name}_seed{seed}_result.json"


def run_one(config_name, seed):
    rp = _result_path(config_name, seed)
    if rp.is_file():
        print(f"[{config_name} seed={seed}] already done, skipping")
        return
    cfg = BASELINE_CONFIGS[config_name]
    setup = get_setup()
    t0 = time.perf_counter()
    paramsK, results, preds, gt = utils_inverse.evaluate_baseline_over_multiple_tests(
        emis_dict_train=setup["emis_dict_train_JAX"], eval_sets=setup["eval_sets"],
        key=jax.random.PRNGKey(seed), hidden_sizes=HIDDEN_SIZES,
        K=cfg["K"], lr=cfg["lr"], weight_decay=cfg["weight_decay"], mode=MODE,
    )
    eval_scores = {k: float(v["mean"]) for k, v in results.items()}
    dt = time.perf_counter() - t0
    row = {"config_name": config_name, "seed": seed, **cfg, "eval_scores": eval_scores, "seconds": round(dt, 2)}
    print(f"[{config_name} seed={seed}] {eval_scores} ({dt:.1f}s)", flush=True)
    tmp = rp.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, rp)


def collect():
    result_files = sorted(OUT_DIR.glob("gap_*_seed*_result.json"))
    if not result_files:
        print("no results yet")
        return
    rows = [json.load(open(p)) for p in result_files]
    out_path = OUT_DIR / "gap_results.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"collected {len(rows)} results -> {out_path}")


def report():
    results_path = OUT_DIR / "gap_results.json"
    if not results_path.is_file():
        print("no collected results yet, run --mode collect first")
        return
    if not OPTIMIZED_SCORES_PATH.is_file():
        print(f"missing optimized-emulator scores at {OPTIMIZED_SCORES_PATH}")
        return

    rows = json.load(open(results_path))
    optimized = json.load(open(OPTIMIZED_SCORES_PATH))["per_group_mean_score"]

    by_config = {}
    for r in rows:
        by_config.setdefault(r["config_name"], []).append(r)

    print(f"{'eval_set':>10} {'optimized':>10} {'baseline_orig':>14} {'gap_orig_%':>11} "
          f"{'baseline_tuned':>15} {'gap_tuned_%':>12}")
    out_table = {}
    for eval_set, group in EVAL_TO_GROUP.items():
        opt_score = optimized[group]
        row_out = {"optimized_score": opt_score}
        cells = [f"{eval_set:>10} {opt_score:>10.4g}"]
        for config_name in ["original", "tuned"]:
            rs = by_config.get(config_name, [])
            scores = [r["eval_scores"][eval_set] for r in rs if eval_set in r["eval_scores"]]
            if not scores:
                cells.append(f"{'--':>14} {'--':>11}" if config_name == "original" else f"{'--':>15} {'--':>12}")
                continue
            mean_score = float(np.mean(scores))
            gap_pct = 100.0 * (mean_score - opt_score) / mean_score
            row_out[f"baseline_{config_name}"] = mean_score
            row_out[f"gap_{config_name}_pct"] = gap_pct
            if config_name == "original":
                cells.append(f"{mean_score:>14.4g} {gap_pct:>10.1f}%")
            else:
                cells.append(f"{mean_score:>15.4g} {gap_pct:>11.1f}%")
        print(" ".join(cells))
        out_table[eval_set] = row_out

    out_path = OUT_DIR / "gap_persistence_table.json"
    with open(out_path, "w") as f:
        json.dump(out_table, f, indent=2)
    print(f"\nsaved -> {out_path}")
    print("\ngap_% = 100*(baseline_score - optimized_score)/baseline_score : "
          "the optimized emulator's relative NRMSE improvement over that baseline.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["run-one", "collect", "report"], required=True)
    parser.add_argument("--config", choices=list(BASELINE_CONFIGS), default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "run-one":
        if args.config is None or args.seed is None:
            raise SystemExit("run-one requires --config and --seed")
        run_one(args.config, args.seed)
    elif args.mode == "collect":
        collect()
    elif args.mode == "report":
        report()


if __name__ == "__main__":
    main()
