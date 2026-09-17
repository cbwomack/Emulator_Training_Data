#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
SIf sensitivity re-run: redo the three CO2-only supplementary sensitivity
sweeps (initial condition / architecture / features) from
supplementary_notebooks/SIf_sensitivity.ipynb using Stage 0b's retuned
unified hyperparameter config (data/SI_results/hp_retune/best_config_unified.json)
and Stage 6e's retuned baseline config
(data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json), at
num_updates=1000 (matching the SI figures' own "over 1000 iterations"
captions - NOT the mainline pipeline's now-2000-step convention) across 50
stochastic seeds, so the resulting figures can report median+IQR like
Figure 3 instead of a single deterministic run.

The original notebook used ad hoc hardcoded hyperparameters (step_size=1e3
for the IC section, 1e2 for architecture/features; smoothness_weight=1e-5
throughout) with zero seed variation, and its architecture section had a
leftover-variable bug that made every run silently use init_cond='sine'
regardless of which condition the loop claimed to test. This script fixes
that by making init_cond an explicit per-call argument, never inherited
from an outer loop.

Writes to NEW seed_sweep/ subdirectories under each existing
data/SI_results/sensitivity_{initial_condition,architecture,features}/
directory, leaving the existing single-seed *.pkl caches one level up
untouched.

One setup (baseline MLP + training data + trained baseline emulator) is
built once per (structural condition, seed) and reused across every
initial-condition variant that condition needs - mirroring
0c_regenerate_checkpoints_co2.py's "one setup per seed, loop over
conditions" pattern, since the baseline emulator does not depend on
init_cond. The initial-condition sweep's own baseline is structurally
identical to the mainline CO2 baseline (hidden_sizes=[16], default EMA
windows), so it reuses the existing checkpoints/co2_retuned/seed_sweep/
baseline cache rather than retraining/rewriting an identical one.

Usage:
    # Initial-condition sweep: one task per seed, loops constant/gaussian/sine internally
    python scripts/SIf_sensitivity_seed_sweep.py --sweep ic --seed 0

    # Architecture sweep: one task per (architecture, seed), loops constant/sine internally
    python scripts/SIf_sensitivity_seed_sweep.py --sweep architecture --condition 16_16 --seed 0

    # Features sweep: one task per (feature-set, seed), loops constant/sine internally
    python scripts/SIf_sensitivity_seed_sweep.py --sweep features --condition medium --seed 0
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
GROUP = 'all'          # T=751, filter_hist=False - matches SIf_sensitivity.ipynb's setup exactly
T = 751
FILTER_HIST = False
NUM_UPDATES = 1000     # per the SI figures' own "over 1000 iterations" captions - do not use the
                        # mainline pipeline's NUM_UPDATES=2000 convention here

UNIFIED_CONFIG_PATH = Path('data/SI_results/hp_retune/best_config_unified.json')
BASELINE_CONFIG_PATH = Path('data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json')

ARCH_DEFS = {'8': [8], '16': [16], '32': [32], '16_16': [16, 16]}
FEATURE_DEFS = {
    'short':  (1.0, 5.0, 10.0),
    'medium': (30.0, 50.0, 70.0),
    'long':   (50.0, 100.0, 200.0),
}
DEFAULT_HIDDEN_SIZES = [16]
DEFAULT_EMA_WINDOWS = (5.0, 30.0, 100.0)

SWEEP_DIRS = {
    'ic':           'data/SI_results/sensitivity_initial_condition/seed_sweep',
    'architecture': 'data/SI_results/sensitivity_architecture/seed_sweep',
    'features':     'data/SI_results/sensitivity_features/seed_sweep',
}


def load_configs():
    unified = json.load(open(UNIFIED_CONFIG_PATH))["config"]
    baseline = json.load(open(BASELINE_CONFIG_PATH))["config"]
    return unified, baseline


def run_one(sweep, condition, seed, unified_cfg, baseline_cfg):
    checkpoint_dir = SWEEP_DIRS[sweep]
    os.makedirs(checkpoint_dir, exist_ok=True)

    if sweep == 'ic':
        hidden_sizes = DEFAULT_HIDDEN_SIZES
        ema_windows_years = DEFAULT_EMA_WINDOWS
        init_conds = ['constant', 'gaussian', 'sine']
        tag_suffix = ''
        # Structurally identical to the mainline CO2 baseline - reuse it
        # instead of retraining/rewriting an identical one per seed.
        baseline_save_path = None
    elif sweep == 'architecture':
        hidden_sizes = ARCH_DEFS[condition]
        ema_windows_years = DEFAULT_EMA_WINDOWS
        init_conds = ['constant', 'sine']
        tag_suffix = f'_{condition}'
        baseline_save_path = f'{checkpoint_dir}/baseline_co2_only_{condition}_seed{seed}.pkl'
    elif sweep == 'features':
        hidden_sizes = DEFAULT_HIDDEN_SIZES
        ema_windows_years = FEATURE_DEFS[condition]
        init_conds = ['constant', 'sine']
        tag_suffix = f'_{condition}'
        baseline_save_path = f'{checkpoint_dir}/baseline_co2_only_{condition}_seed{seed}.pkl'
    else:
        raise ValueError(f"Unknown sweep {sweep!r}")

    setup = utils_inverse.run_inverse_experiment_setup(
        AGENTS, ACTIVE_AGENTS, mode=MODE,
        hidden_sizes=hidden_sizes, ema_windows_years=ema_windows_years,
        CS3=True, DAMIP=False, GeoMIP=False,
        idx_demo=None, seed=seed,
        baseline_save_path=baseline_save_path,
        baseline_K=baseline_cfg["K"], baseline_lr=baseline_cfg["lr"],
        baseline_weight_decay=baseline_cfg["weight_decay"],
    )

    for init_cond in init_conds:
        print(f"Running sweep={sweep} condition={condition} init_cond={init_cond} seed={seed}...", flush=True)
        utils_inverse.run_inverse_experiment(
            setup,
            group=GROUP,
            checkpoint_dir=checkpoint_dir,
            tag=f"co2_only{tag_suffix}_seed{seed}",
            num_updates=NUM_UPDATES,
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
            ema_windows_years=ema_windows_years,
            key=jax.random.PRNGKey(seed),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", choices=["ic", "architecture", "features"], required=True)
    parser.add_argument(
        "--condition", default=None,
        help="Required for --sweep architecture (one of 8/16/32/16_16) or --sweep features "
             "(one of short/medium/long). Ignored for --sweep ic.")
    parser.add_argument(
        "--seed", type=int, nargs="+", default=list(range(50)),
        help="One or more seeds (default 0-49). Pass a single value for one seed - the natural "
             "unit for a SLURM array task.")
    args = parser.parse_args()

    if args.sweep == 'architecture' and args.condition not in ARCH_DEFS:
        parser.error(f"--sweep architecture requires --condition in {list(ARCH_DEFS)}")
    if args.sweep == 'features' and args.condition not in FEATURE_DEFS:
        parser.error(f"--sweep features requires --condition in {list(FEATURE_DEFS)}")

    unified_cfg, baseline_cfg = load_configs()
    print(f"Unified optimizer config: {unified_cfg}")
    print(f"Baseline config: {baseline_cfg}")

    for seed in args.seed:
        run_one(args.sweep, args.condition, seed, unified_cfg, baseline_cfg)


if __name__ == "__main__":
    main()
