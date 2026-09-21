#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
regenerate the CO2-only checkpoints ,
using the hyperparameter search's unified optimizer hyperparameters

Usage:
    python 03b_regenerate_checkpoints_co2.py                        # every group, seeds 0-49
    python 03b_regenerate_checkpoints_co2.py --group H-ext           # one group, seeds 0-49
    python 03b_regenerate_checkpoints_co2.py --seed 0                # one seed, every group
    python 03b_regenerate_checkpoints_co2.py --group H-ext --seed 3  # one group, one seed (SLURM array task)
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
MODE = 'FaIR'
CHECKPOINT_DIR = 'checkpoints/co2_retuned'
SEED_SWEEP_DIR = f'{CHECKPOINT_DIR}/seed_sweep' 

UNIFIED_CONFIG_PATH = Path('data/SI_results/hp_retune/best_config_unified.json')
BASELINE_CONFIG_PATH = Path('data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json')

GROUP_DEFS = {
    'H-ext': {'init_cond': 'sine',     'T': 477, 'filter_hist': True},
    'tier1': {'init_cond': 'constant', 'T': 751, 'filter_hist': False},
    'tier2': {'init_cond': 'constant', 'T': 751, 'filter_hist': True},
    'DECK':  {'init_cond': 'constant', 'T': 751, 'filter_hist': False},
    'CS3':   {'init_cond': 'constant', 'T': 751, 'filter_hist': True},
    'all':   {'init_cond': 'constant', 'T': 751, 'filter_hist': False},
}
NUM_UPDATES = 2000
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
                         help="One or more seeds for params0/baseline init (default 0-49). Pass a "
                              "single value (e.g. --seed 3) for one seed - the natural unit "
                              "for a SLURM array task.")
    args = parser.parse_args()

    unified_cfg, baseline_cfg = load_configs()
    print(f"Unified optimizer config: {unified_cfg}")
    print(f"Baseline config: {baseline_cfg}")

    os.makedirs(SEED_SWEEP_DIR, exist_ok=True)

    to_run = [args.group] if args.group else list(GROUP_DEFS)

    for seed in args.seed:
        print(f"=== seed {seed} ===")

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
