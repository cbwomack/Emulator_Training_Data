#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Multi-agent analog of scripts/0c_regenerate_checkpoints_agent.py: regenerate
checkpoints/multi/'s H-ext/tier1/DAMIP/GeoMIP/all checkpoints on current
HEAD, using 0b_hyperparameter_retune_multi.py's per-agent step_size dict
config (data/SI_results/hp_retune/multi/best_config_unified.json) and
6e_baseline_hp_search_k400_multi.py's own baseline config (CO2's tuned
config confirmed NOT to transfer to the multi-agent case - see REVISIONS.md).

NUM_UPDATES=1000 and the 50-seed sweep both match the single-forcing
convention exactly, per user direction (2026-08-13) - the old multi
checkpoints used inconsistent update counts per group (500-10000, see
REVISIONS.md Session Log), unlike every single-forcing agent's uniform
1000.

Writes fresh checkpoints to a NEW checkpoints/multi_retuned/ directory,
leaving the original checkpoints/multi/ (stale, pre-Phase-0, referenced by
Figs 4/5/6/7) completely untouched - same convention as every other agent's
Phase 0 regeneration this session.

Usage:
    python 0c_regenerate_checkpoints_multi.py                        # every group, seeds 0-49
    python 0c_regenerate_checkpoints_multi.py --group H-ext           # one group, seeds 0-49
    python 0c_regenerate_checkpoints_multi.py --group H-ext --seed 3  # one group, one seed (SLURM array task)
"""
import os
import sys
import json
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import jax

import utils_inverse

MODE = "FaIR"
NUM_UPDATES = 1000  # matches the single-forcing convention, per user direction 2026-08-13
AGENTS = ["CO2", "CH4", "N2O", "Sulfur", "BC"]
ACTIVE_AGENTS = ("CO2", "CH4", "N2O", "Sulfur", "BC")
TAG = "all_agents"

CHECKPOINT_DIR = "checkpoints/multi_retuned"
SEED_SWEEP_DIR = f"{CHECKPOINT_DIR}/seed_sweep"

UNIFIED_CONFIG_PATH = Path("data/SI_results/hp_retune/multi/best_config_unified.json")
BASELINE_CONFIG_PATH = Path("data/SI_results/baseline_hp/k400_search_multi/best_baseline_config_K400.json")

GROUPS = ["H-ext", "tier1", "DAMIP", "GeoMIP", "all"]


def load_3b_experiments_module():
    spec = importlib.util.spec_from_file_location(
        "_exp_mod_3b", PROJECT_ROOT / "scripts" / "3b_inverse_all_agents.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_group_defs():
    """Per-group init_cond/T/filter_hist - NOT tunable hyperparameters, just
    which scenarios/initial-condition each group optimizes against. H-ext/
    DAMIP/GeoMIP/all come from 3b_inverse_all_agents.py's own EXPERIMENTS
    dict; tier1 has no entry there (see 6c_ood_scenario_ssp370_lowntcf.py's
    module docstring on this same gap) so it's defined explicitly here,
    matching the single-forcing agents' own tier1 convention (constant/751/
    False) - the same definition used in 0b_hyperparameter_retune_multi.py's
    cheap-search/validate for consistency between search and regeneration.
    """
    exp_mod = load_3b_experiments_module()
    defs = {g: {"init_cond": exp_mod.EXPERIMENTS[g]["init_cond"],
                "T": exp_mod.EXPERIMENTS[g]["T"],
                "filter_hist": exp_mod.EXPERIMENTS[g]["filter_hist"]}
            for g in ["H-ext", "DAMIP", "GeoMIP", "all"]}
    defs["tier1"] = {"init_cond": "constant", "T": 751, "filter_hist": False}
    return defs


def load_configs():
    unified = json.load(open(UNIFIED_CONFIG_PATH))["config"]
    baseline = json.load(open(BASELINE_CONFIG_PATH))["config"]
    return unified, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=GROUPS, default=None,
                         help="Regenerate only this group's checkpoint (default: run all, in order)")
    parser.add_argument("--seed", type=int, nargs="+", default=list(range(50)),
                         help="One or more seeds (default 0-49, matching the single-forcing "
                              "Stage 6a UQ protocol). Pass a single value for one seed - the "
                              "natural unit for a SLURM array task.")
    args = parser.parse_args()

    unified_cfg, baseline_cfg = load_configs()
    group_defs = build_group_defs()
    print(f"[multi] Unified optimizer config: {unified_cfg}")
    print(f"[multi] Baseline config: {baseline_cfg}")

    os.makedirs(SEED_SWEEP_DIR, exist_ok=True)

    to_run = [args.group] if args.group else GROUPS

    for seed in args.seed:
        print(f"=== [multi] seed {seed} ===")

        write_baseline = (args.group is None) or (args.group == "H-ext")
        baseline_save_path = f"{SEED_SWEEP_DIR}/baseline_{TAG}_seed{seed}.pkl" if write_baseline else None

        setup = utils_inverse.run_inverse_experiment_setup(
            AGENTS, ACTIVE_AGENTS, mode=MODE,
            CS3=True, DAMIP=True, GeoMIP=True,
            idx_demo=None, seed=seed,
            baseline_save_path=baseline_save_path,
            baseline_K=baseline_cfg["K"], baseline_lr=baseline_cfg["lr"],
            baseline_weight_decay=baseline_cfg["weight_decay"],
        )

        for name in to_run:
            print(f"Running group {name!r} (seed {seed})...")
            gdef = group_defs[name]
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
