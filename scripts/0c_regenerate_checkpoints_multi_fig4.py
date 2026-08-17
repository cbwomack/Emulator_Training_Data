#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Figure 4's multi-agent panel (and Figure 5, and Figure 6's individual-effects
cache) has never actually depended on checkpoints/multi_retuned/ (this
session's 0c_regenerate_checkpoints_multi.py, group set {H-ext, tier1,
DAMIP, GeoMIP, all}) - it depends on a SEPARATE, older checkpoint family,
checkpoints/multi/*_subset*.pkl, dated January 2026 (pre-dates this whole
revision effort), group set {tier1, tier2, DECK, CS3, all} (matching every
single-forcing agent's own convention, since plot_vertical_stacked_bars
shares one train_scenarios/test_scenarios/x_labels axis across all panels
in Figure 4). Inspecting those files directly showed wildly inconsistent
update counts (1,401 to 20,001 across groups, vs. this session's uniform
1000) - the same class of staleness problem as everything else this
session, discovered only once Stage 6f's Fig 4 extension traced through
load_fig4_data()/load_fig5_multi_forcing_data()'s actual default paths.

Per explicit user direction (2026-08-13): the checkpoints/multi/*_subset*
family is DEPRECATED. This script regenerates the {tier1, tier2, DECK, CS3,
all} group set fresh, on current code, REUSING this session's already-found
multi-agent Phase 0 hyperparameters (data/SI_results/hp_retune/multi/
best_config_unified.json + data/SI_results/baseline_hp/k400_search_multi/
best_baseline_config_K400.json - no new search needed, since those
hyperparameters are agent-general, not group-specific, exactly as for
0c_regenerate_checkpoints_multi.py's own {H-ext, DAMIP, GeoMIP, all} set).
GROUP_DEFS below is copied verbatim from 0c_regenerate_checkpoints_co2.py's
own GROUP_DEFS (minus 'H-ext', which Figure 4 excludes - see
regenerate_fig4_co2_only_cache_seed_sweep's docstring) - these are NOT
tunable hyperparameters, just which scenarios/initial-condition each group
optimizes against, and must match every single-forcing agent's own
convention exactly for Figure 4's shared axis to be meaningful.

Writes to a NEW checkpoints/multi_fig4/ directory - distinct from both
checkpoints/multi/ (deprecated _subset family, left untouched) and
checkpoints/multi_retuned/ (this session's {H-ext,tier1,DAMIP,GeoMIP,all}
set, serves Stage 6c(b) only, also left untouched) - to avoid any ambiguity
about which group-def scheme a given 'tier1'/'all' checkpoint was produced
under, even though tier1's own def happens to be identical between the two
schemes.

Usage:
    python 0c_regenerate_checkpoints_multi_fig4.py                        # every group, seeds 0-49
    python 0c_regenerate_checkpoints_multi_fig4.py --group tier2           # one group, seeds 0-49
    python 0c_regenerate_checkpoints_multi_fig4.py --group tier2 --seed 3  # one group, one seed (SLURM array task)
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

AGENTS = ["CO2", "CH4", "N2O", "Sulfur", "BC"]
ACTIVE_AGENTS = ("CO2", "CH4", "N2O", "Sulfur", "BC")
MODE = "FaIR"
CHECKPOINT_DIR = "checkpoints/multi_fig4"
SEED_SWEEP_DIR = f"{CHECKPOINT_DIR}/seed_sweep"

UNIFIED_CONFIG_PATH = Path("data/SI_results/hp_retune/multi/best_config_unified.json")
BASELINE_CONFIG_PATH = Path("data/SI_results/baseline_hp/k400_search_multi/best_baseline_config_K400.json")

# Verbatim copy of 0c_regenerate_checkpoints_co2.py's GROUP_DEFS, minus
# 'H-ext' (excluded from Figure 4, per regenerate_fig4_co2_only_cache_seed_
# sweep's own convention) - NOT tunable, just which scenarios/init-cond each
# group optimizes against; must match every single-forcing agent's own
# scheme for Figure 4's shared train_scenarios/test_scenarios axis.
GROUP_DEFS = {
    "tier1": {"init_cond": "constant", "T": 751, "filter_hist": False},
    "tier2": {"init_cond": "constant", "T": 751, "filter_hist": True},
    "DECK":  {"init_cond": "constant", "T": 751, "filter_hist": False},
    "CS3":   {"init_cond": "constant", "T": 751, "filter_hist": True},
    "all":   {"init_cond": "constant", "T": 751, "filter_hist": False},
}
NUM_UPDATES = 1000  # matches the single-forcing convention (and this session's other multi-agent regen)
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
                              "Stage 6a UQ protocol). Pass a single value for one seed - the "
                              "natural unit for a SLURM array task.")
    args = parser.parse_args()

    unified_cfg, baseline_cfg = load_configs()
    print(f"[multi_fig4] Unified optimizer config: {unified_cfg}")
    print(f"[multi_fig4] Baseline config: {baseline_cfg}")

    os.makedirs(SEED_SWEEP_DIR, exist_ok=True)

    to_run = [args.group] if args.group else list(GROUP_DEFS)

    for seed in args.seed:
        print(f"=== [multi_fig4] seed {seed} ===")

        write_baseline = (args.group is None) or (args.group == "tier1")
        baseline_save_path = f"{SEED_SWEEP_DIR}/baseline_{TAG}_seed{seed}.pkl" if write_baseline else None

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
