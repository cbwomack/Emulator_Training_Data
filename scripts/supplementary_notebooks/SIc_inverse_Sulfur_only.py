#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5 and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Companion script for supplementary_notebooks/SIc_inverse_Sulfur_only.ipynb: runs the Sulfur-only bilevel inverse-optimization
experiments used to build checkpoints/Sulfur/inverse_*.pkl checkpoints. Runs
standalone (no notebook/kernel needed) so long jobs can be scheduled; the
notebook itself only loads the resulting checkpoints and plots them.

Usage:
    python SIc_inverse_Sulfur_only.py                        # run every experiment below, in order
    python SIc_inverse_Sulfur_only.py --experiment H-ext    # run just one
    python SIc_inverse_Sulfur_only.py --baseline-save-path checkpoints/.../baseline_x.pkl
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import utils_inverse

AGENTS = ['Sulfur']
ACTIVE_AGENTS = ('Sulfur',)
MODE = 'FaIR'
CHECKPOINT_DIR = 'checkpoints/Sulfur'
DEFAULT_BASELINE_SAVE_PATH = None

# Exact hyperparameters transcribed programmatically from supplementary/3d_inverse_Sulfur_only.ipynb - see
# PROGRESS.md for the extraction method. Preserved as-is, including known
# oddities (e.g. duplicate 'all' entries, "2"-suffixed checkpoint tags)
# rather than silently "fixing" them; those are flagged for Phase 5.
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
