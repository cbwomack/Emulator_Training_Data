#!/usr/bin/env python
"""
Companion script for supplementary_notebooks/SIe_inverse_all_agents_subset.ipynb: runs the all-forcing-agent (subset scenarios) bilevel inverse-optimization
experiments used to build checkpoints/multi/inverse_*.pkl checkpoints. Runs
standalone (no notebook/kernel needed) so long jobs can be scheduled; the
notebook itself only loads the resulting checkpoints and plots them.

Usage:
    python SIe_inverse_all_agents_subset.py                        # run every experiment below, in order
    python SIe_inverse_all_agents_subset.py --experiment tier1    # run just one
    python SIe_inverse_all_agents_subset.py --baseline-save-path checkpoints/.../baseline_x.pkl
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import utils_inverse

AGENTS = ['CO2', 'CH4', 'N2O', 'Sulfur', 'BC']
ACTIVE_AGENTS = ('CO2', 'CH4', 'N2O', 'Sulfur', 'BC')
MODE = 'FaIR'
CHECKPOINT_DIR = 'checkpoints/multi'
DEFAULT_BASELINE_SAVE_PATH = 'checkpoints/multi/baseline_all_agents_subset.pkl'

# Exact hyperparameters transcribed programmatically from supplementary/3g_inverse_all_agents_subset.ipynb - see
# PROGRESS.md for the extraction method. Preserved as-is, including known
# oddities (e.g. duplicate 'all' entries, "2"-suffixed checkpoint tags)
# rather than silently "fixing" them; those are flagged for Phase 5.
EXPERIMENTS = {
    'tier1':     {
        'group': 'tier1',
        'tag': 'all_agents_subset2',
        'num_updates': 1000,
        'step_size': {'CO2': 300.0, 'CH4': 900.0, 'N2O': 80.0, 'Sulfur': 800.0, 'BC': 40.0},
        'momentum': 0.99,
        'nesterov': True,
        'K_inner': 400,
        'lr_inner': 0.05,
        'wd_inner': 0.01,
        'init_cond': 'sine',
        'T': 751,
        'filter_hist': False,
        'smoothness_weight': 0.0001,
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
