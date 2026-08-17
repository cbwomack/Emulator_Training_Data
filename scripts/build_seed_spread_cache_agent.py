#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage E (cluster-parallelized): build each agent's per-seed SI-extended-
results cache ({seed: {baseline, optimal}}, consumed by
utils_inverse.load_SI_extended_results_data_seed_sweep for SI Fig 6's
error bars) by running utils_inverse.regenerate_SI_extended_results_cache_
seed_sweep one seed at a time across a SLURM array, instead of the ~4-hour
serial loop CO2's own Fig-4 cache took when run as a single local process
(see REVISIONS.md, 2026-08-05/06 - that must not be repeated for the
remaining agents).

Each task computes exactly one seed's {baseline, optimal} pair and writes
it to its own partial file (idempotent, resumable - same pattern used
throughout Phase 0); --mode collect merges every agent's partial files into
the final data/SI_results/seed_uncertainty/SI_extended_seed_spread_{agent}.pkl.

Usage:
    python build_seed_spread_cache_agent.py --mode run-one --agent CH4 --seed 0
    python build_seed_spread_cache_agent.py --mode collect --agent CH4
"""
import os
import sys
import json
import pickle
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import utils_inverse

PARTIAL_DIR = Path("data/SI_results/seed_uncertainty/partial")
FINAL_DIR = Path("data/SI_results/seed_uncertainty")


def _partial_path(agent, seed):
    return PARTIAL_DIR / f"{agent}_seed{seed}.pkl"


def run_one(agent, seed):
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _partial_path(agent, seed)
    if out_path.exists():
        print(f"[{agent} seed={seed}] already done, skipping")
        return
    baseline_config_path = f"data/SI_results/baseline_hp/k400_search_{agent}/best_baseline_config_K400.json"
    utils_inverse.regenerate_SI_extended_results_cache_seed_sweep(
        agent, seeds=[seed], baseline_config_path=baseline_config_path,
        out_path=str(out_path),
    )
    print(f"[{agent} seed={seed}] done -> {out_path}")


def collect(agent, n_seeds=50):
    all_results = {}
    missing = []
    for seed in range(n_seeds):
        p = _partial_path(agent, seed)
        if not p.exists():
            missing.append(seed)
            continue
        with open(p, "rb") as f:
            partial = pickle.load(f)
        all_results.update(partial)
    if missing:
        print(f"[{agent}] missing seeds: {missing} ({len(missing)}/{n_seeds}) - not writing final cache yet")
        return
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINAL_DIR / f"SI_extended_seed_spread_{agent}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"[{agent}] collected {len(all_results)} seeds -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["run-one", "collect"], required=True)
    parser.add_argument("--agent", required=True, choices=["CH4", "N2O", "Sulfur", "BC"])
    parser.add_argument("--seed", type=int, default=None, help="run-one only")
    parser.add_argument("--n-seeds", type=int, default=50, help="collect only")
    args = parser.parse_args()

    if args.mode == "run-one":
        if args.seed is None:
            raise SystemExit("run-one requires --seed")
        run_one(args.agent, args.seed)
    elif args.mode == "collect":
        collect(args.agent, args.n_seeds)


if __name__ == "__main__":
    main()
