#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Extends Table S2's Multi row (tab:supp_opt_v_dyn) into a full Figure-4-style
grid: evaluates the multi-agent "dynamically rich" reference (sine-IC,
num_updates=0.

Usage:
    python pipeline/07_results/07p_build_multi_dynrich_cache.py --mode run-one --seed 0
    python pipeline/07_results/07p_build_multi_dynrich_cache.py --mode collect
"""
import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import jax
import utils_inverse

CONTROL_DIR = Path("checkpoints/iteration0_controls")
REPACK_DIR = Path("checkpoints/iteration0_controls/_repackaged")
PARTIAL_DIR = Path("data/SI_results/seed_uncertainty/partial_multi_dynrich")
FINAL_OUT = Path("data/SI_results/seed_uncertainty/multi_dynrich_eval_grid.pkl")

UNIFIED_CONFIG_PATH = "data/SI_results/hp_retune/multi/best_config_unified.json"
BASELINE_CONFIG_PATH = "data/SI_results/baseline_hp/k400_search_multi/best_baseline_config_K400.json"
AGENTS = ["CO2", "CH4", "N2O", "Sulfur", "BC"]
ACTIVE_AGENTS = ("CO2", "CH4", "N2O", "Sulfur", "BC")
IC = "sine" # Best performing dynamically rich initial guess


def run_one(seed):
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PARTIAL_DIR / f"seed{seed}.pkl"
    if out_path.exists():
        print(f"[seed={seed}] already done, skipping")
        return

    control_path = CONTROL_DIR / f"inverse_{IC}_tier1_multi_fig4_ic-{IC}_seed{seed}.pkl"
    with open(control_path, "rb") as f:
        control = pickle.load(f)

    REPACK_DIR.mkdir(parents=True, exist_ok=True)
    repacked_path = REPACK_DIR / f"inverse_{IC}_seed{seed}_repacked.pkl"
    with open(repacked_path, "wb") as f:
        pickle.dump({"U_traj": [control["U_init"]]}, f)

    unified_cfg = json.load(open(UNIFIED_CONFIG_PATH))["config"]
    baseline_cfg = json.load(open(BASELINE_CONFIG_PATH))["config"]

    setup = utils_inverse.run_inverse_experiment_setup(
        AGENTS, ACTIVE_AGENTS, mode="FaIR", CS3=True, DAMIP=False, GeoMIP=False,
        idx_demo=None, seed=seed,
        baseline_K=baseline_cfg["K"], baseline_lr=baseline_cfg["lr"],
        baseline_weight_decay=baseline_cfg["weight_decay"],
    )

    result = utils_inverse.evaluate_optimal_emulator(
        training_paths=[str(repacked_path)],
        train_scenarios=["Dyn. Rich"],
        eval_sets=setup["eval_sets"],
        params0=setup["params0"],
        agents=AGENTS,
        active_agents=ACTIVE_AGENTS,
        inactive_mode="zeros",
        historical_name="historical",
        key=jax.random.PRNGKey(seed),
        K=unified_cfg["K_inner"],
        lr=unified_cfg["lr_inner"],
        weight_decay=unified_cfg["wd_inner"],
        mode="FaIR",
        batch_size=unified_cfg["batch_size"],
    )

    with open(out_path, "wb") as f:
        pickle.dump({seed: result["Dyn. Rich"]}, f)
    print(f"[seed={seed}] done -> {out_path} (Tier 1 mean={result['Dyn. Rich']['Tier 1']['mean']:.4f})")


def collect(n_seeds=50):
    all_results = {}
    missing = []
    for seed in range(n_seeds):
        p = PARTIAL_DIR / f"seed{seed}.pkl"
        if not p.exists():
            missing.append(seed)
            continue
        with open(p, "rb") as f:
            all_results.update(pickle.load(f))
    if missing:
        print(f"missing seeds: {missing} ({len(missing)}/{n_seeds}) - not writing final cache yet")
        return
    FINAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_OUT, "wb") as f:
        pickle.dump(all_results, f)
    print(f"collected {len(all_results)} seeds -> {FINAL_OUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["run-one", "collect"], required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=50)
    args = parser.parse_args()

    if args.mode == "run-one":
        if args.seed is None:
            raise SystemExit("run-one requires --seed")
        run_one(args.seed)
    elif args.mode == "collect":
        collect(args.n_seeds)


if __name__ == "__main__":
    main()
