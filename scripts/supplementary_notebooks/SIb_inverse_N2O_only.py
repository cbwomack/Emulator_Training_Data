#!/usr/bin/env python
"""
Companion script for supplementary_notebooks/SIb_inverse_N2O_only.ipynb: runs the N2O-only bilevel inverse-optimization
experiments used to build checkpoints/n2o/inverse_*.pkl checkpoints. Runs
standalone (no notebook/kernel needed) so long jobs can be scheduled; the
notebook itself only loads the resulting checkpoints and plots them.

Usage:
    python SIb_inverse_N2O_only.py                        # run every experiment below, in order
    python SIb_inverse_N2O_only.py --experiment H-ext    # run just one
    python SIb_inverse_N2O_only.py --baseline-save-path checkpoints/.../baseline_x.pkl
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import utils_inverse

AGENTS = ['N2O']
ACTIVE_AGENTS = ('N2O',)
MODE = 'FaIR'
CHECKPOINT_DIR = 'checkpoints/n2o'
DEFAULT_BASELINE_SAVE_PATH = 'checkpoints/n2o/baseline_n2o_only.pkl'

# Exact hyperparameters transcribed programmatically from supplementary/3c_inverse_N2O_only.ipynb - see
# PROGRESS.md for the extraction method. Preserved as-is, including known
# oddities (e.g. duplicate 'all' entries, "2"-suffixed checkpoint tags)
# rather than silently "fixing" them; those are flagged for Phase 5.
EXPERIMENTS = {
    'H-ext':     {
        'group': 'H-ext',
        'tag': 'n2o_only',
        'num_updates': 1000,
        'step_size': 1000.0,
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
        'tag': 'n2o_only',
        'num_updates': 1000,
        'step_size': 50.0,
        'momentum': 0.9,
        'nesterov': True,
        'K_inner': 400,
        'lr_inner': 0.05,
        'wd_inner': 0.01,
        'init_cond': 'constant',
        'T': 751,
        'filter_hist': False,
        'smoothness_weight': 0,
        'checkpoint_every': 50,
        'resume_if_exists': False,
        'preds_every': 50,
    },
    'tier2':     {
        'group': 'tier2',
        'tag': 'n2o_only',
        'num_updates': 1000,
        'step_size': 50.0,
        'momentum': 0.9,
        'nesterov': True,
        'K_inner': 400,
        'lr_inner': 0.05,
        'wd_inner': 0.01,
        'init_cond': 'constant',
        'T': 751,
        'filter_hist': True,
        'smoothness_weight': 0,
        'checkpoint_every': 50,
        'resume_if_exists': False,
        'preds_every': 50,
    },
    'DECK':     {
        'group': 'DECK',
        'tag': 'n2o_only',
        'num_updates': 1000,
        'step_size': 50.0,
        'momentum': 0.9,
        'nesterov': True,
        'K_inner': 400,
        'lr_inner': 0.05,
        'wd_inner': 0.01,
        'init_cond': 'constant',
        'T': 751,
        'filter_hist': False,
        'smoothness_weight': 0,
        'checkpoint_every': 50,
        'resume_if_exists': False,
        'preds_every': 50,
    },
    'CS3':     {
        'group': 'CS3',
        'tag': 'n2o_only',
        'num_updates': 1000,
        'step_size': 50.0,
        'momentum': 0.9,
        'nesterov': True,
        'K_inner': 400,
        'lr_inner': 0.05,
        'wd_inner': 0.01,
        'init_cond': 'constant',
        'T': 751,
        'filter_hist': True,
        'smoothness_weight': 0,
        'checkpoint_every': 50,
        'resume_if_exists': False,
        'preds_every': 50,
    },
    'all':     {
        'group': 'all',
        'tag': 'n2o_only',
        'num_updates': 1000,
        'step_size': 50.0,
        'momentum': 0.9,
        'nesterov': True,
        'K_inner': 400,
        'lr_inner': 0.05,
        'wd_inner': 0.01,
        'init_cond': 'constant',
        'T': 751,
        'filter_hist': False,
        'smoothness_weight': 0,
        'checkpoint_every': 50,
        'resume_if_exists': False,
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
