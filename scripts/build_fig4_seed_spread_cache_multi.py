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

SEED_UNC_DIR = Path("data/SI_results/seed_uncertainty")


def _partial_dir(suffix=""):
    return SEED_UNC_DIR / f"partial_fig4_multi{suffix}"


def _final_path(suffix=""):
    return SEED_UNC_DIR / f"fig4_seed_spread_all_agents{suffix}.pkl"


def _partial_path(seed, suffix=""):
    return _partial_dir(suffix) / f"seed{seed}.pkl"


def run_one(seed, checkpoint_dir=None, suffix=""):
    _partial_dir(suffix).mkdir(parents=True, exist_ok=True)
    out_path = _partial_path(seed, suffix)
    if out_path.exists():
        print(f"[seed={seed}] already done, skipping")
        return
    # Only forward checkpoint_dir when asked, so the default path keeps
    # regenerate_fig4_all_agents_cache_seed_sweep's own default
    # (checkpoints/multi_fig4/seed_sweep) rather than duplicating it here.
    kwargs = {} if checkpoint_dir is None else {"checkpoint_dir": checkpoint_dir}
    utils_inverse.regenerate_fig4_all_agents_cache_seed_sweep(
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
             "checkpoints/multi_fig4/seed_sweep. Point at "
             "checkpoints/multi_fig4_smooth/seed_sweep for the smoothed arm.")
    parser.add_argument(
        "--suffix", default="",
        help="appended to both the partial directory and the final cache "
             "filename, e.g. _smooth -> fig4_seed_spread_all_agents_smooth.pkl. "
             "Required whenever --checkpoint-dir is given.")
    args = parser.parse_args()

    # A different checkpoint family is a different cache. Without a suffix the
    # smoothed run would silently overwrite the unsmoothed one it is meant to be
    # compared against, and the partial files would interleave two arms.
    if args.checkpoint_dir is not None and not args.suffix:
        raise SystemExit(
            "--checkpoint-dir requires --suffix, otherwise this run overwrites "
            "data/SI_results/seed_uncertainty/fig4_seed_spread_all_agents.pkl "
            "and mixes its partials with the default arm's.")

    if args.mode == "run-one":
        if args.seed is None:
            raise SystemExit("run-one requires --seed")
        run_one(args.seed, args.checkpoint_dir, args.suffix)
    elif args.mode == "collect":
        collect(args.n_seeds, args.suffix)


if __name__ == "__main__":
    main()
