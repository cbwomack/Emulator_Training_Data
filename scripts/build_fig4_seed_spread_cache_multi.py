#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Multi-agent analog of build_seed_spread_cache_agent.py: build Figure 4's
multi-agent panel seed-spread cache ({seed: {baseline, optimal}}) by running
utils_inverse.regenerate_fig4_all_agents_cache_seed_sweep one seed at a time
across a SLURM array, reading checkpoints/multi_fig4/seed_sweep/ (see
REVISIONS.md, 2026-08-13, on why this checkpoint family exists and what it
replaces). Same idempotent partial-file + collect pattern as Stage E.

Usage:
    python build_fig4_seed_spread_cache_multi.py --mode run-one --seed 0
    python build_fig4_seed_spread_cache_multi.py --mode collect
"""
import os
import sys
import pickle
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import utils_inverse

PARTIAL_DIR = Path("data/SI_results/seed_uncertainty/partial_fig4_multi")
FINAL_PATH = Path("data/SI_results/seed_uncertainty/fig4_seed_spread_all_agents.pkl")


def _partial_path(seed):
    return PARTIAL_DIR / f"seed{seed}.pkl"


def run_one(seed):
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _partial_path(seed)
    if out_path.exists():
        print(f"[seed={seed}] already done, skipping")
        return
    utils_inverse.regenerate_fig4_all_agents_cache_seed_sweep(
        seeds=[seed], out_path=str(out_path),
    )
    print(f"[seed={seed}] done -> {out_path}")


def collect(n_seeds=50):
    all_results = {}
    missing = []
    for seed in range(n_seeds):
        p = _partial_path(seed)
        if not p.exists():
            missing.append(seed)
            continue
        with open(p, "rb") as f:
            partial = pickle.load(f)
        all_results.update(partial)
    if missing:
        print(f"missing seeds: {missing} ({len(missing)}/{n_seeds}) - not writing final cache yet")
        return
    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_PATH, "wb") as f:
        pickle.dump(all_results, f)
    print(f"collected {len(all_results)} seeds -> {FINAL_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["run-one", "collect"], required=True)
    parser.add_argument("--seed", type=int, default=None, help="run-one only")
    parser.add_argument("--n-seeds", type=int, default=50, help="collect only")
    args = parser.parse_args()

    if args.mode == "run-one":
        if args.seed is None:
            raise SystemExit("run-one requires --seed")
        run_one(args.seed)
    elif args.mode == "collect":
        collect(args.n_seeds)


if __name__ == "__main__":
    main()
