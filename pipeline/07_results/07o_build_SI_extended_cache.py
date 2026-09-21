#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Build each agent's per-seed SI-extended-
results cache ({seed: {baseline, optimal}}, consumed by
utils_inverse.load_SI_extended_results_data_seed_sweep for SI Fig 6's
error bars) by running utils_inverse.regenerate_SI_extended_results_cache_
seed_sweep one seed at a time across a SLURM array.

Usage:
    python 07o_build_SI_extended_cache.py --mode run-one --agent CH4 --seed 0
    python 07o_build_SI_extended_cache.py --mode collect --agent CH4
"""
import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import utils_inverse

FINAL_DIR = Path("data/SI_results/seed_uncertainty")


def _partial_dir(suffix=""):
    return Path(f"data/SI_results/seed_uncertainty/partial{suffix}")


def _partial_path(agent, seed, suffix=""):
    return _partial_dir(suffix) / f"{agent}_seed{seed}.pkl"


def run_one(agent, seed, checkpoint_dir=None, suffix=""):
    _partial_dir(suffix).mkdir(parents=True, exist_ok=True)
    out_path = _partial_path(agent, seed, suffix)
    if out_path.exists():
        print(f"[{agent} seed={seed}] already done, skipping")
        return
    baseline_config_path = f"data/SI_results/baseline_hp/k400_search_{agent}/best_baseline_config_K400.json"
    kwargs = {} if checkpoint_dir is None else {"checkpoint_dir": checkpoint_dir}
    utils_inverse.regenerate_SI_extended_results_cache_seed_sweep(
        agent, seeds=[seed], baseline_config_path=baseline_config_path,
        out_path=str(out_path), **kwargs,
    )
    print(f"[{agent} seed={seed}] done -> {out_path}")


def collect(agent, n_seeds=50, suffix=""):
    all_results = {}
    missing = []
    for seed in range(n_seeds):
        p = _partial_path(agent, seed, suffix)
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
    out_path = FINAL_DIR / f"SI_extended_seed_spread_{agent}{suffix}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"[{agent}] collected {len(all_results)} seeds -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["run-one", "collect"], required=True)
    parser.add_argument("--agent", required=True, choices=["CH4", "N2O", "Sulfur", "BC"])
    parser.add_argument("--seed", type=int, default=None, help="run-one only")
    parser.add_argument("--n-seeds", type=int, default=50, help="collect only")
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="seed_sweep directory to read optimal checkpoints from; defaults to "
             "checkpoints/{agent_lower}_retuned/seed_sweep. Point at e.g. "
             "checkpoints/Sulfur_smooth/seed_sweep for the smoothed arm.")
    parser.add_argument(
        "--suffix", default="",
        help="appended to both the partial directory and the final cache "
             "filename, e.g. _smooth -> SI_extended_seed_spread_Sulfur_smooth.pkl. "
             "Required whenever --checkpoint-dir is given.")
    args = parser.parse_args()

    if args.checkpoint_dir is not None and not args.suffix:
        raise SystemExit(
            "--checkpoint-dir requires --suffix, otherwise this run overwrites "
            f"data/SI_results/seed_uncertainty/SI_extended_seed_spread_{args.agent}.pkl "
            "and mixes its partials with the default arm's.")

    if args.mode == "run-one":
        if args.seed is None:
            raise SystemExit("run-one requires --seed")
        run_one(args.agent, args.seed, args.checkpoint_dir, args.suffix)
    elif args.mode == "collect":
        collect(args.agent, args.n_seeds, args.suffix)


if __name__ == "__main__":
    main()
