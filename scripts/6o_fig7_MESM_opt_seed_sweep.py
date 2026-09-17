#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage 6o: re-run Figure 7's CO2-only MESM inverse optimization with every
update from the revisions, across 50 stochastic seeds x 3 initial conditions.

This is the main-text Figure 7 experiment (panels (a)/(b): the optimized CO2
emissions trajectory and the temperature response it produces) redone on
current HEAD. Three things changed relative to the original run in
scripts/4a_inverse_CO2_only_MESM.py:

  1. SCM calibration. mode='MESM_tier1' selects utils_FaIR_JAX.
     MESM_TIER1_PARAMS - the calibration adopted in 2c_calibrate_MESM_both.ipynb
     (carbon cycle from 1pctCO2, thermal timescales from 1pctCO2 + all seven
     Tier 1 scenarios, ECS pinned to MESM's measured 3.205 K). The original ran
     against mode='MESM', whose ECS is ~11% too warm. See
     MESM_SCM_CALIBRATION.md.
  2. Optimizer / emulator hyperparameters. Stage 0b's unified config and Stage
     6e's retuned baseline config, replacing the original's hand-set
     step_size=5000/lr_inner=0.05/wd_inner=0.01.
  3. Seeds and initial conditions. 50 seeds (the original was a single
     deterministic run) x constant/sine/gaussian (the original ran 'sine' only
     for this group), so Figure 7 can report percentile trajectories and a
     median+IQR convergence band rather than one representative line.

Only the 'all' group ("Opt. All") is run - per the user's scoping for this
rerun, the other four groups are out of scope.

Writes to a NEW directory (checkpoints/co2_MESM_retuned/seed_sweep/), leaving
checkpoints/co2/inverse_*_co2_only_MESM.pkl completely untouched. That is not
just this repo's usual convention: real MIT Earth System Model ensemble output
(data/MESM/emis_driven/zonal_data/optimized/*.nc) was produced against those
exact trajectories, so overwriting them would invalidate ground truth that
cannot be regenerated without commissioning new MESM runs. See
4a_inverse_CO2_only_MESM.py's own warning banner.

Each seed's baseline emulator is trained with the same seed as its optimized
counterpart (the Stage 0 "fairness property" - run_inverse_experiment_setup
seeds params0 and the baseline together), and is saved per-seed so the
convergence figure can show the baseline's own seed spread rather than a
single number. The baseline does not depend on init_cond, so it is built once
per seed and reused across all three.

Usage:
    # One (seed, init_cond) pair - the natural unit for a SLURM array task
    python scripts/6o_fig7_MESM_opt_seed_sweep.py --seed 0 --init-cond constant

    # One seed, all three initial conditions
    python scripts/6o_fig7_MESM_opt_seed_sweep.py --seed 0

    # Everything (50 seeds x 3 ICs) serially - only sane for a smoke test
    python scripts/6o_fig7_MESM_opt_seed_sweep.py
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
MODE = 'MESM_tier1'
GROUP = 'all'
TAG = 'co2_only_MESM'

CHECKPOINT_DIR = 'checkpoints/co2_MESM_retuned'
SEED_SWEEP_DIR = f'{CHECKPOINT_DIR}/seed_sweep'

UNIFIED_CONFIG_PATH = Path('data/SI_results/hp_retune/best_config_unified.json')
BASELINE_CONFIG_PATH = Path('data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json')

INIT_CONDS = ['constant', 'sine', 'gaussian']

# The 'all' group's own experiment definition, unchanged from
# 4a_inverse_CO2_only_MESM.py's EXPERIMENTS['all'] - these say which scenarios
# and what horizon, they are not tunable hyperparameters. init_cond is the one
# field this script sweeps rather than fixes.
T = 751
FILTER_HIST = False

# Matches the post-revision mainline convention (0c_regenerate_checkpoints_co2.py,
# extended from 1000 on 2026-08-25 after the Stage A out-of-sample gate), NOT
# the original Figure 7 run's 1000.
#
# Overridable per run with --num-updates, which is what makes a NON-UNIFORM
# sweep possible (e.g. constant extended to 5000 while sine/gaussian stay at
# 2000 - the convergence figure plots each initial condition in its own panel,
# so panels may differ in length even though seeds within a panel may not).
# Extending is a resume, not a re-run: optimize_emissions_inverse deliberately
# leaves num_updates out of RESUME_INVARIANT_KEYS ("extending it is the entire
# point of resuming"), so raising this and re-running continues an existing
# checkpoint from where it stopped rather than starting over or erroring.
NUM_UPDATES = 2000


def load_configs():
    unified = json.load(open(UNIFIED_CONFIG_PATH))["config"]
    baseline = json.load(open(BASELINE_CONFIG_PATH))["config"]
    return unified, baseline


def run_one(seed, init_conds, unified_cfg, baseline_cfg, num_updates=NUM_UPDATES):
    os.makedirs(SEED_SWEEP_DIR, exist_ok=True)

    baseline_save_path = f'{SEED_SWEEP_DIR}/baseline_{TAG}_seed{seed}.pkl'

    setup = utils_inverse.run_inverse_experiment_setup(
        AGENTS, ACTIVE_AGENTS, mode=MODE,
        CS3=True, DAMIP=False, GeoMIP=False,
        idx_demo=None, seed=seed,
        baseline_save_path=baseline_save_path,
        baseline_K=baseline_cfg["K"], baseline_lr=baseline_cfg["lr"],
        baseline_weight_decay=baseline_cfg["weight_decay"],
    )

    for init_cond in init_conds:
        print(f"Running group {GROUP!r} init_cond={init_cond} seed={seed}...", flush=True)
        utils_inverse.run_inverse_experiment(
            setup,
            group=GROUP,
            checkpoint_dir=SEED_SWEEP_DIR,
            tag=f'{TAG}_seed{seed}',
            num_updates=num_updates,
            step_size=unified_cfg["step_size"],
            momentum=unified_cfg["momentum"],
            nesterov=unified_cfg["nesterov"],
            K_inner=unified_cfg["K_inner"],
            lr_inner=unified_cfg["lr_inner"],
            wd_inner=unified_cfg["wd_inner"],
            smoothness_weight=unified_cfg["smoothness_weight"],
            batch_size=unified_cfg["batch_size"],
            init_cond=init_cond,
            T=T,
            filter_hist=FILTER_HIST,
            checkpoint_every=50,
            resume_if_exists=True,
            preds_every=50,
            key=jax.random.PRNGKey(seed),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, nargs="+", default=list(range(50)),
                        help="One or more seeds for params0/baseline init (default 0-49). Pass a "
                             "single value for one seed - the natural unit for a SLURM array task.")
    parser.add_argument("--init-cond", choices=INIT_CONDS, default=None,
                        help="Run only this initial condition (default: all three, in order)")
    parser.add_argument("--num-updates", type=int, default=NUM_UPDATES,
                        help=f"Outer iterations (default {NUM_UPDATES}). Raising this on an "
                             f"existing checkpoint RESUMES it to the new total rather than "
                             f"restarting, so it is the way to extend one initial condition "
                             f"without touching the others.")
    args = parser.parse_args()

    unified_cfg, baseline_cfg = load_configs()
    print(f"Mode: {MODE}")
    print(f"Num updates: {args.num_updates}")
    print(f"Unified optimizer config: {unified_cfg}")
    print(f"Baseline config: {baseline_cfg}")

    init_conds = [args.init_cond] if args.init_cond else INIT_CONDS

    for seed in args.seed:
        print(f"=== seed {seed} ===")
        run_one(seed, init_conds, unified_cfg, baseline_cfg, num_updates=args.num_updates)


if __name__ == "__main__":
    main()
