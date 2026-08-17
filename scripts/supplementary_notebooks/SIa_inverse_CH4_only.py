#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5 and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Companion script for supplementary_notebooks/SIa_inverse_CH4_only.ipynb: runs the CH4-only bilevel inverse-optimization
experiments used to build checkpoints/ch4/inverse_*.pkl checkpoints. Runs
standalone (no notebook/kernel needed) so long jobs can be scheduled; the
notebook itself only loads the resulting checkpoints and plots them.

Usage:
    python SIa_inverse_CH4_only.py                        # run every experiment below, in order
    python SIa_inverse_CH4_only.py --experiment H-ext    # run just one
    python SIa_inverse_CH4_only.py --baseline-save-path checkpoints/.../baseline_x.pkl
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import utils_inverse

AGENTS = ['CH4']
ACTIVE_AGENTS = ('CH4',)
MODE = 'FaIR'
CHECKPOINT_DIR = 'checkpoints/ch4'
DEFAULT_BASELINE_SAVE_PATH = None

# Exact hyperparameters transcribed programmatically from supplementary/3b_inverse_CH4_only.ipynb - see
# PROGRESS.md for the extraction method. Preserved as-is, including known
# oddities (e.g. duplicate 'all' entries, "2"-suffixed checkpoint tags)
# rather than silently "fixing" them; those are flagged for Phase 5.
#
# tier1/DECK/CS3/all added 2026-08-11 (Phase 0 extension to CH4): the
# original extraction only ever captured H-ext/tier2, but SI Fig 6 and
# Figure 3 both need the full 6-group set every other agent has. `T`/
# `filter_hist` per group are copied from the pattern confirmed identical
# across every agent that does have all 6 groups (CO2/N2O/BC): tier1
# T=751/filter_hist=False, DECK T=751/False, CS3 T=751/True, all T=751/False
# - these are experiment-definition fields, not tuned hyperparameters, so
# they transfer directly. `step_size`/`momentum`/etc. below are placeholders
# (copied from tier2) - irrelevant in practice, since Stage 0c's checkpoint
# regeneration overwrites every tunable field with the Stage 0b unified
# per-agent config and only reads init_cond/T/filter_hist from here.
EXPERIMENTS = {
    'H-ext':     {
        'group': 'H-ext',
        'tag': 'ch4_only',
        'num_updates': 1000,
        'step_size': 10000.0,
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
        'tag': 'ch4_only',
        'num_updates': 1000,
        'step_size': 100000.0,
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
    'tier2':     {
        'group': 'tier2',
        'tag': 'ch4_only2',
        'num_updates': 1000,
        'step_size': 100000.0,
        'momentum': 0.9,
        'nesterov': True,
        'K_inner': 400,
        'lr_inner': 0.05,
        'wd_inner': 0.01,
        'init_cond': 'constant',
        'T': 751,
        'filter_hist': True,
        'smoothness_weight': 1e-06,
        'checkpoint_every': 50,
        'resume_if_exists': False,
        'preds_every': 50,
    },
    'DECK':     {
        'group': 'DECK',
        'tag': 'ch4_only',
        'num_updates': 1000,
        'step_size': 100000.0,
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
        'tag': 'ch4_only',
        'num_updates': 1000,
        'step_size': 100000.0,
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
        'tag': 'ch4_only',
        'num_updates': 1000,
        'step_size': 100000.0,
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
