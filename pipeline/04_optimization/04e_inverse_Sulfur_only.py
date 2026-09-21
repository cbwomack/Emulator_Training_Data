#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5, Claude Sonnet 5, and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Companion script for pipeline/04_optimization/04e_inverse_Sulfur_only.ipynb: runs the Sulfur-only optimization
experiments used to build checkpoints/Sulfur/inverse_*.pkl checkpoints. 

Usage:
    python 04e_inverse_Sulfur_only.py                        # run every experiment below, in order
    python 04e_inverse_Sulfur_only.py --experiment H-ext    # run just one
    python 04e_inverse_Sulfur_only.py --baseline-save-path checkpoints/.../baseline_x.pkl
"""
import os
import sys
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import argparse

import utils_inverse

AGENTS = ['Sulfur']
ACTIVE_AGENTS = ('Sulfur',)
MODE = 'FaIR'
CHECKPOINT_DIR = 'checkpoints/Sulfur'
DEFAULT_BASELINE_SAVE_PATH = None

EXPERIMENTS = {
    'H-ext':     {
        'group': 'H-ext',
        'tag': 'Sulfur_only',
        'num_updates': 1000,
        'step_size': 5000.0,
        'momentum': 0.9,
        'nesterov': True,
        'K_inner': 400,
        'lr_inner': 0.05,
        'wd_inner': 0.01,
        'init_cond': 'constant',
        'T': 477,
        'filter_hist': True,
        'smoothness_weight': 0.0,
        'checkpoint_every': 50,
        'resume_if_exists': True,
        'preds_every': 50,
    },
    'tier1':     {
        'group': 'tier1',
        'tag': 'Sulfur_only2',
        'num_updates': 1000,
        'step_size': 10000.0,
        'momentum': 0.9,
        'nesterov': True,
        'K_inner': 400,
        'lr_inner': 0.05,
        'wd_inner': 0.01,
        'init_cond': 'constant',
        'T': 751,
        'filter_hist': False,
        'smoothness_weight': 5e-06,
        'checkpoint_every': 50,
        'resume_if_exists': True,
        'preds_every': 50,
    },
    'tier2':     {
        'group': 'tier2',
        'tag': 'Sulfur_only',
        'num_updates': 1000,
        'step_size': 10000.0,
        'momentum': 0.9,
        'nesterov': True,
        'K_inner': 400,
        'lr_inner': 0.05,
        'wd_inner': 0.01,
        'init_cond': 'constant',
        'T': 751,
        'filter_hist': True,
        'smoothness_weight': 0.0,
        'checkpoint_every': 50,
        'resume_if_exists': True,
        'preds_every': 50,
    },
    'DECK':     {
        'group': 'DECK',
        'tag': 'Sulfur_only',
        'num_updates': 1000,
        'step_size': 10000.0,
        'momentum': 0.9,
        'nesterov': True,
        'K_inner': 400,
        'lr_inner': 0.05,
        'wd_inner': 0.01,
        'init_cond': 'constant',
        'T': 751,
        'filter_hist': False,
        'smoothness_weight': 0.0,
        'checkpoint_every': 50,
        'resume_if_exists': True,
        'preds_every': 50,
    },
    'CS3':     {
        'group': 'CS3',
        'tag': 'Sulfur_only',
        'num_updates': 1000,
        'step_size': 10000.0,
        'momentum': 0.9,
        'nesterov': True,
        'K_inner': 400,
        'lr_inner': 0.05,
        'wd_inner': 0.01,
        'init_cond': 'constant',
        'T': 751,
        'filter_hist': True,
        'smoothness_weight': 0.0,
        'checkpoint_every': 50,
        'resume_if_exists': True,
        'preds_every': 50,
    },
    'all':     {
        'group': 'all',
        'tag': 'Sulfur_only',
        'num_updates': 1000,
        'step_size': 10000.0,
        'momentum': 0.9,
        'nesterov': True,
        'K_inner': 400,
        'lr_inner': 0.05,
        'wd_inner': 0.01,
        'init_cond': 'constant',
        'T': 751,
        'filter_hist': False,
        'smoothness_weight': 0.0,
        'checkpoint_every': 50,
        'resume_if_exists': True,
        'preds_every': 50,
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=list(EXPERIMENTS), default=None,
                         help="Run only this experiment (default: run all, in order)")
    parser.add_argument("--baseline-save-path", default=DEFAULT_BASELINE_SAVE_PATH,
                         help="Where to save the baseline emulator (default: matches the notebook's current setting)")
    args = parser.parse_args()

    setup = utils_inverse.run_inverse_experiment_setup(
        AGENTS, ACTIVE_AGENTS, mode=MODE,
        CS3=True, DAMIP=False, GeoMIP=False,
        baseline_save_path=args.baseline_save_path,
    )

    to_run = [args.experiment] if args.experiment else list(EXPERIMENTS)
    for name in to_run:
        print(f"Running experiment {name!r}...")
        utils_inverse.run_inverse_experiment(
            setup, checkpoint_dir=CHECKPOINT_DIR, **EXPERIMENTS[name],
        )


if __name__ == "__main__":
    main()
