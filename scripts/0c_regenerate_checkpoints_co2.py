#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Phase 0, Stage 0c: regenerate the CO2-only checkpoints on current HEAD,
using Stage 0b's unified optimizer hyperparameters (ONE shared config
across all 6 groups - data/SI_results/hp_retune/best_config_unified.json)
and Stage 6e's independently-tuned baseline hyperparameters
(data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json).

Every existing checkpoints/co2/*.pkl file predates a real, undocumented
pipeline change (confirmed in REVISIONS.md's session log - Jan26-Mar9 2026)
and was never regenerated, so today's code cannot reproduce it. This
writes fresh checkpoints to a NEW directory (checkpoints/co2_retuned/),
leaving the original checkpoints/co2/ completely untouched, per explicit
user direction - old and new checkpoints must remain independently
available for comparison, not overwritten.

Each group keeps its own init_cond/T/filter_hist - these define the
experiment itself (which scenarios, what initial condition), not a tunable
hyperparameter Stage 0b searched over. Only step_size/momentum/nesterov/
K_inner/lr_inner/wd_inner/smoothness_weight/batch_size are replaced by the
single shared unified config (identical across every group, by design -
see REVISIONS.md's Stage 0b redesign).

**Multi-seed (2026-08-05, Stage 6a rerun prep)**: `--seed` now accepts one or
more seeds and every seed writes to a NEW `checkpoints/co2_retuned/seed_sweep/`
subdirectory (checkpoint + baseline both tag-suffixed `_seed{N}`), giving each
seed its own independently-trained baseline as well as its own optimized
trajectory (same Stage 0 "fairness property": one seed drives both, via
run_inverse_experiment_setup). This lets Figure 4 report mean +/- spread
across seeds instead of a single-seed point estimate (reviewer Point 1).
The original flat `checkpoints/co2_retuned/*.pkl` (seed 0 only, written before
this change) is left untouched as a frozen snapshot, matching this repo's
convention of never reinterpreting an existing artifact in place - seed 0 is
simply re-run again, uniformly, as part of the seed_sweep/ directory.

Usage:
    python 0c_regenerate_checkpoints_co2.py                        # every group, seeds 0-49
    python 0c_regenerate_checkpoints_co2.py --group H-ext           # one group, seeds 0-49
    python 0c_regenerate_checkpoints_co2.py --seed 0                # one seed, every group
    python 0c_regenerate_checkpoints_co2.py --group H-ext --seed 3  # one group, one seed (SLURM array task)
"""
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import jax

import utils_inverse

AGENTS = ['CO2']
ACTIVE_AGENTS = ('CO2',)
MODE = 'FaIR'
CHECKPOINT_DIR = 'checkpoints/co2_retuned'  # NEW dir - checkpoints/co2/ (stale) left untouched
SEED_SWEEP_DIR = f'{CHECKPOINT_DIR}/seed_sweep'  # multi-seed rerun output (see module docstring)

UNIFIED_CONFIG_PATH = Path('data/SI_results/hp_retune/best_config_unified.json')
BASELINE_CONFIG_PATH = Path('data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json')

# Per-group experiment definition, unchanged from 3a_inverse_CO2_only.py's
# EXPERIMENTS dict - these are NOT tunable hyperparameters, just which
# scenarios/initial-condition each group optimizes against.
GROUP_DEFS = {
    'H-ext': {'init_cond': 'sine',     'T': 477, 'filter_hist': True},
    'tier1': {'init_cond': 'constant', 'T': 751, 'filter_hist': False},
    'tier2': {'init_cond': 'constant', 'T': 751, 'filter_hist': True},
    'DECK':  {'init_cond': 'constant', 'T': 751, 'filter_hist': False},
    'CS3':   {'init_cond': 'constant', 'T': 751, 'filter_hist': True},
    'all':   {'init_cond': 'constant', 'T': 751, 'filter_hist': False},
}
NUM_UPDATES = 1000  # production length, matches the original checkpoints
TAG = 'co2_only'


def load_configs():
    unified = json.load(open(UNIFIED_CONFIG_PATH))["config"]
    baseline = json.load(open(BASELINE_CONFIG_PATH))["config"]
    return unified, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=list(GROUP_DEFS), default=None,
                         help="Regenerate only this group's checkpoint (default: run all, in order)")
    parser.add_argument("--seed", type=int, nargs="+", default=list(range(50)),
                         help="One or more seeds for params0/baseline init (default 0-49, per "
                              "user direction for the Stage 6a UQ rerun). Pass a single value "
                              "(e.g. --seed 3) for one seed - the natural unit for a SLURM array task.")
    args = parser.parse_args()

    unified_cfg, baseline_cfg = load_configs()
    print(f"Unified optimizer config: {unified_cfg}")
    print(f"Baseline config: {baseline_cfg}")

    os.makedirs(SEED_SWEEP_DIR, exist_ok=True)

    to_run = [args.group] if args.group else list(GROUP_DEFS)

    for seed in args.seed:
        print(f"=== seed {seed} ===")

        # Only one task (default: the alphabetically/logically-first group, when
        # running the full array one-group-per-task) writes the shared baseline
        # checkpoint - every group's setup call trains an identical baseline
        # (same seed, same baseline hyperparameters), so writing it N times in
        # parallel would just race N parallel writers on the same file for no
        # benefit. `--group` runs (used by the SLURM array) skip the write
        # unless they are 'H-ext'; a no-`--group` (run-everything) invocation
        # always writes it once at the end of setup, before any group loop.
        write_baseline = (args.group is None) or (args.group == "H-ext")
        baseline_save_path = f"{SEED_SWEEP_DIR}/baseline_co2_only_seed{seed}.pkl" if write_baseline else None

        setup = utils_inverse.run_inverse_experiment_setup(
            AGENTS, ACTIVE_AGENTS, mode=MODE,
            CS3=True, DAMIP=False, GeoMIP=False,
            idx_demo=None, seed=seed,
            baseline_save_path=baseline_save_path,
            baseline_K=baseline_cfg["K"], baseline_lr=baseline_cfg["lr"],
            baseline_weight_decay=baseline_cfg["weight_decay"],
        )

        for name in to_run:
            print(f"Running group {name!r} (seed {seed})...")
            gdef = GROUP_DEFS[name]
            utils_inverse.run_inverse_experiment(
                setup,
                group=name,
                checkpoint_dir=SEED_SWEEP_DIR,
                tag=f"{TAG}_seed{seed}",
                num_updates=NUM_UPDATES,
                step_size=unified_cfg["step_size"],
                momentum=unified_cfg["momentum"],
                nesterov=unified_cfg["nesterov"],
                K_inner=unified_cfg["K_inner"],
                lr_inner=unified_cfg["lr_inner"],
                wd_inner=unified_cfg["wd_inner"],
                smoothness_weight=unified_cfg["smoothness_weight"],
                batch_size=unified_cfg["batch_size"],
                init_cond=gdef["init_cond"],
                T=gdef["T"],
                filter_hist=gdef["filter_hist"],
                checkpoint_every=50,
                resume_if_exists=True,
                preds_every=50,
                key=jax.random.PRNGKey(seed),
            )


if __name__ == "__main__":
    main()
