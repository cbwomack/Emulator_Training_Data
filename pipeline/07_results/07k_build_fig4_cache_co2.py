#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
CO2-only analog of 07l_build_fig4_cache_multi.py: build Figure 4's
CO2-only panel seed-spread cache ({seed: {baseline, optimal}}) by running
utils_inverse.regenerate_fig4_co2_only_cache_seed_sweep one seed at a time
across a SLURM array, reading checkpoints/co2_retuned/seed_sweep/.

Usage:
    python 07k_build_fig4_cache_co2.py --mode run-one --seed 0
    python 07k_build_fig4_cache_co2.py --mode collect
"""
import os
import sys
import pickle
import argparse
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import utils_inverse

SEED_UNC_DIR = Path("data/SI_results/seed_uncertainty")


def _partial_dir(suffix=""):
    return SEED_UNC_DIR / f"partial_fig4_co2{suffix}"


def _final_path(suffix=""):
    return SEED_UNC_DIR / f"fig4_seed_spread_co2_only{suffix}.pkl"


def _partial_path(seed, suffix=""):
    return _partial_dir(suffix) / f"seed{seed}.pkl"


def run_one(seed, checkpoint_dir=None, suffix=""):
    _partial_dir(suffix).mkdir(parents=True, exist_ok=True)
    out_path = _partial_path(seed, suffix)
    if out_path.exists():
        print(f"[seed={seed}] already done, skipping")
        return
    kwargs = {} if checkpoint_dir is None else {"checkpoint_dir": checkpoint_dir}
    utils_inverse.regenerate_fig4_co2_only_cache_seed_sweep(
        seeds=[seed], out_path=str(out_path), **kwargs,
    )
    print(f"[seed={seed}] done -> {out_path}")


def collect(n_seeds=50, suffix=""):
    all_results = {}
    missing = []
    for seed in range(n_seeds):
        p = _partial_path(seed, suffix)
        if not p.exists():
            missing.append(seed)
            continue
        with open(p, "rb") as f:
            partial = pickle.load(f)
        all_results.update(partial)
    if missing:
        print(f"missing seeds: {missing} ({len(missing)}/{n_seeds}) - not writing final cache yet")
        return
    final_path = _final_path(suffix)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"collected {len(all_results)} seeds -> {final_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["run-one", "collect"], required=True)
    parser.add_argument("--seed", type=int, default=None, help="run-one only")
    parser.add_argument("--n-seeds", type=int, default=50, help="collect only")
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="seed_sweep directory to read optimal checkpoints from; defaults to "
             "checkpoints/co2_retuned/seed_sweep.")
    parser.add_argument(
        "--suffix", default="",
        help="appended to both the partial directory and the final cache "
             "filename. Required whenever --checkpoint-dir is given.")
    args = parser.parse_args()

    if args.checkpoint_dir is not None and not args.suffix:
        raise SystemExit(
            "--checkpoint-dir requires --suffix, otherwise this run overwrites "
            "data/SI_results/seed_uncertainty/fig4_seed_spread_co2_only.pkl "
            "and mixes its partials with the default arm's.")

    if args.mode == "run-one":
        if args.seed is None:
            raise SystemExit("run-one requires --seed")
        run_one(args.seed, args.checkpoint_dir, args.suffix)
    elif args.mode == "collect":
        collect(args.n_seeds, args.suffix)


if __name__ == "__main__":
    main()
