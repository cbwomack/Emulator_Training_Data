#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
CO2-only MESM inverse optimization, 50 seeds x 3 initial conditions.

Produces main-text Figure 7's panels (a) and (b): the optimized CO2 emissions
trajectory and the temperature response it produces.

  * SCM calibration: mode='MESM_tier1' selects utils_FaIR_JAX.MESM_TIER1_PARAMS
    - carbon cycle from 1pctCO2, thermal timescales from 1pctCO2 plus all seven
    Tier 1 scenarios, ECS pinned to MESM's measured 3.205 K, as fitted in
    02c_calibrate_MESM.ipynb.
  * Hyperparameters: the unified search config and the retuned baseline config.
  * 50 seeds x constant/sine/gaussian, so Figure 7 can report percentile
    trajectories and a median+IQR convergence band rather than one line.

Only the 'all' group ("Opt. All") is run.

Each seed's baseline emulator is trained with the same seed as its optimized
counterpart. The baseline does not depend on init_cond, so it is
built once per seed and reused across all three.

Usage:
    # One (seed, init_cond) pair - the natural unit for a SLURM array task
    python 05b_optimize_MESM_emissions.py --seed 0 --init-cond constant
    python 05b_optimize_MESM_emissions.py --seed 0      # one seed, all three ICs
    python 05b_optimize_MESM_emissions.py               # 50 x 3, serially

"""
import os
import sys
import json
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

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
T = 751
FILTER_HIST = False
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
