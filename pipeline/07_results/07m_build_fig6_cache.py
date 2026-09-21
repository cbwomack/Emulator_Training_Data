#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Builds Figure 6's individual-effects seed-spread cache
({seed: {y_true, y_hat_baseline, y_hat}}) by running
utils_inverse.regenerate_fig6_individual_effects_cache_seed_sweep one seed at
a time across a SLURM array. 

Usage:
    python 07m_build_fig6_cache.py --mode run-one --seed 0
    python 07m_build_fig6_cache.py --mode collect
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
    return SEED_UNC_DIR / f"partial_fig6_ind_effects{suffix}"


def _final_path(suffix=""):
    return SEED_UNC_DIR / f"fig6_seed_spread_ind_effects{suffix}.pkl"


def _partial_path(seed, suffix=""):
    return _partial_dir(suffix) / f"seed{seed}.pkl"


def run_one(seed, init_cond, tier1_dir, multi_dir, suffix=""):
    _partial_dir(suffix).mkdir(parents=True, exist_ok=True)
    out_path = _partial_path(seed, suffix)
    if out_path.exists():
        print(f"[seed={seed}] already done, skipping")
        return
    kwargs = {}
    if tier1_dir is not None:
        kwargs["tier1_checkpoint_dir"] = tier1_dir
    if multi_dir is not None:
        kwargs["multi_checkpoint_dir"] = multi_dir
    utils_inverse.regenerate_fig6_individual_effects_cache_seed_sweep(
        seeds=[seed], init_cond=init_cond, out_path=str(out_path), **kwargs,
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
            all_results.update(pickle.load(f))
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
        "--init-cond", default="constant", choices=("constant", "sine"),
        help="which DAMIP/GeoMIP/all checkpoints to read. 'constant' for the "
             "main-paper figure; 'sine' only reproduces the legacy arm.")
    parser.add_argument(
        "--tier1-checkpoint-dir", default=None,
        help="defaults to checkpoints/multi_fig4_smooth/seed_sweep.")
    parser.add_argument(
        "--multi-checkpoint-dir", default=None,
        help="defaults to checkpoints/multi_retuned_smooth/seed_sweep.")
    parser.add_argument(
        "--suffix", default="",
        help="appended to the partial directory and the final cache filename. "
             "Required whenever a checkpoint dir or a non-default --init-cond "
             "is given, so two arms cannot share one cache.")
    args = parser.parse_args()

    non_default = (args.tier1_checkpoint_dir is not None
                   or args.multi_checkpoint_dir is not None
                   or args.init_cond != "constant")
    if non_default and not args.suffix:
        raise SystemExit(
            "--suffix is required when overriding --init-cond or a checkpoint "
            "directory, otherwise this run overwrites "
            "data/SI_results/seed_uncertainty/fig6_seed_spread_ind_effects.pkl "
            "and mixes its partials with the default arm's.")

    if args.mode == "run-one":
        if args.seed is None:
            raise SystemExit("run-one requires --seed")
        run_one(args.seed, args.init_cond, args.tier1_checkpoint_dir,
                args.multi_checkpoint_dir, args.suffix)
    elif args.mode == "collect":
        collect(args.n_seeds, args.suffix)


if __name__ == "__main__":
    main()
