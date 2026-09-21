#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Regenerates checkpoints for plotting fig4

Usage:
    python 03b_regenerate_checkpoints_multi_fig4.py                        # every group, seeds 0-49
    python 03b_regenerate_checkpoints_multi_fig4.py --group tier2           # one group, seeds 0-49
    python 03b_regenerate_checkpoints_multi_fig4.py --group tier2 --seed 3  # one group, one seed (SLURM array task)
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

AGENTS = ["CO2", "CH4", "N2O", "Sulfur", "BC"]
ACTIVE_AGENTS = ("CO2", "CH4", "N2O", "Sulfur", "BC")
MODE = "FaIR"
CHECKPOINT_DIR = "checkpoints/multi_fig4"
SEED_SWEEP_DIR = f"{CHECKPOINT_DIR}/seed_sweep"

UNIFIED_CONFIG_PATH = Path("data/SI_results/hp_retune/multi/best_config_unified.json")
BASELINE_CONFIG_PATH = Path("data/SI_results/baseline_hp/k400_search_multi/best_baseline_config_K400.json")

GROUP_DEFS = {
    "tier1": {"init_cond": "constant", "T": 751, "filter_hist": False},
    "tier2": {"init_cond": "constant", "T": 751, "filter_hist": True},
    "DECK":  {"init_cond": "constant", "T": 751, "filter_hist": False},
    "CS3":   {"init_cond": "constant", "T": 751, "filter_hist": True},
    "all":   {"init_cond": "constant", "T": 751, "filter_hist": False},
}
NUM_UPDATES = 2000 
TAG = "multi_fig4"


def load_configs():
    unified = json.load(open(UNIFIED_CONFIG_PATH))["config"]
    baseline = json.load(open(BASELINE_CONFIG_PATH))["config"]
    return unified, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=list(GROUP_DEFS), default=None,
                         help="Regenerate only this group's checkpoint (default: run all, in order)")
    parser.add_argument("--seed", type=int, nargs="+", default=list(range(50)),
                         help="One or more seeds (default 0-49, matching the single-forcing "
                              "the seed sweep UQ protocol). Pass a single value for one seed - the "
                              "natural unit for a SLURM array task.")
    parser.add_argument("--smoothness-weight", type=float, default=None,
                         help="Override the tuned config's smoothness_weight. Pass this to "
                              "build a smoothed arm; combine with --penalty-form normalized "
                              "and --out-dir so the unsmoothed arm on disk is left intact.")
    parser.add_argument("--penalty-form", choices=("legacy", "normalized"), default="legacy",
                         help="Which smoothness penalty to apply. 'legacy' is the historical "
                              "unnormalized sum and is the default so existing behaviour is "
                              "unchanged; 'normalized' is the dimensionless, agent-count- and "
                              "length-normalized form (utils_inverse.smoothness_penalty_terms).")
    parser.add_argument("--out-dir", default=None,
                         help="Override the checkpoint directory. Required in practice when "
                              "--smoothness-weight is given: writing a differently-regularized "
                              "run into the default directory would overwrite the arm it is "
                              "meant to be compared against.")
    args = parser.parse_args()

    unified_cfg, baseline_cfg = load_configs()
    print(f"[multi_fig4] Unified optimizer config: {unified_cfg}")
    print(f"[multi_fig4] Baseline config: {baseline_cfg}")

    checkpoint_dir = args.out_dir or CHECKPOINT_DIR
    seed_sweep_dir = f"{checkpoint_dir}/seed_sweep"

    if args.smoothness_weight is not None:
        smoothness_weight = args.smoothness_weight
    else:
        smoothness_weight = unified_cfg["smoothness_weight"]
    if (args.out_dir is None
            and (smoothness_weight != unified_cfg["smoothness_weight"]
                 or args.penalty_form != "legacy")):
        raise SystemExit(
            f"refusing to write a differently-regularized run into {checkpoint_dir}: "
            f"smoothness_weight={smoothness_weight!r} penalty_form={args.penalty_form!r} "
            f"versus the tuned {unified_cfg['smoothness_weight']!r}/legacy. "
            f"Pass --out-dir to keep the existing multi_fig4 arm intact.")

    os.makedirs(seed_sweep_dir, exist_ok=True)

    to_run = [args.group] if args.group else list(GROUP_DEFS)

    for seed in args.seed:
        print(f"=== [multi_fig4] seed {seed} ===")

        write_baseline = (args.group is None) or (args.group == "tier1")
        baseline_save_path = f"{seed_sweep_dir}/baseline_{TAG}_seed{seed}.pkl" if write_baseline else None

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
                checkpoint_dir=seed_sweep_dir,
                tag=f"{TAG}_seed{seed}",
                num_updates=NUM_UPDATES,
                step_size=unified_cfg["step_size"],
                momentum=unified_cfg["momentum"],
                nesterov=unified_cfg["nesterov"],
                K_inner=unified_cfg["K_inner"],
                lr_inner=unified_cfg["lr_inner"],
                wd_inner=unified_cfg["wd_inner"],
                smoothness_weight=smoothness_weight,
                penalty_form=args.penalty_form,
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
