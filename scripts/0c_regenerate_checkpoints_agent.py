#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Phase 0, Stage 0c (agent-generalized): regenerate CH4/N2O/Sulfur/BC
checkpoints on current HEAD, using Stage 0b's per-agent unified optimizer
config (data/SI_results/hp_retune/{agent}/best_config_unified.json) and a
baseline config resolved per-agent by Stage C's transfer check
(scripts/6e_baseline_transfer_check.py) - CO2's tuned
best_baseline_config_K400.json where it transfers, that agent's own dense
K=400 search winner where it doesn't.

Generalizes scripts/0c_regenerate_checkpoints_co2.py (left untouched - its
checkpoints/co2_retuned/ artifacts are not to be disturbed). One structural
difference from the CO2 script: GROUP_DEFS (each group's init_cond/T/
filter_hist) is loaded dynamically from the agent's own SIa-SId companion
script's EXPERIMENTS dict, not hardcoded - CO2's H-ext uses init_cond='sine'
while every other agent's H-ext uses 'constant' (confirmed directly), so
hardcoding a shared GROUP_DEFS would silently get this wrong for one agent
or another. --baseline-config-path has NO default (matching this repo's
"no hyperparameter has a silent default" convention for these regeneration
scripts) - the caller must explicitly resolve which baseline config to use,
per Stage C's per-agent verdict, rather than risk an unreviewed assumption.

Writes fresh checkpoints to a NEW checkpoints/{agent_lower}_retuned/
directory (agent_lower matches the existing convention: lowercase except
Sulfur/BC, which keep their capitalization - see
regenerate_SI_extended_results_cache), leaving any existing stale
checkpoints/{agent_lower}/ completely untouched.

Multi-seed by default (50 seeds, matching CO2's Stage 6a protocol) -
--seed accepts one or more seeds; every seed writes to
checkpoints/{agent_lower}_retuned/seed_sweep/, tag-suffixed _seed{N}, with
its own independently-trained baseline (same Stage 0 fairness property: one
seed drives both baseline and optimized-emulator init).

Usage:
    python 0c_regenerate_checkpoints_agent.py --agent CH4 \\
        --baseline-config-path data/SI_results/baseline_hp/k400_search_CH4/best_baseline_config_K400.json
    python 0c_regenerate_checkpoints_agent.py --agent CH4 --group H-ext --seed 3 \\
        --baseline-config-path ...   # one group, one seed (SLURM array task)
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

MODE = 'FaIR'
NUM_UPDATES = 1000  # production length, matches CO2's regenerated checkpoints

AGENT_CONFIG = {
    "CH4":    {"module": "SIa_inverse_CH4_only.py", "agents": ["CH4"],    "active_agents": ("CH4",),    "tag": "ch4_only"},
    "N2O":    {"module": "SIb_inverse_N2O_only.py", "agents": ["N2O"],    "active_agents": ("N2O",),    "tag": "n2o_only"},
    "Sulfur": {"module": "SIc_inverse_Sulfur_only.py", "agents": ["Sulfur"], "active_agents": ("Sulfur",), "tag": "Sulfur_only"},
    "BC":     {"module": "SId_inverse_BC_only.py", "agents": ["BC"],     "active_agents": ("BC",),     "tag": "BC_only"},
}
GROUPS = ['H-ext', 'tier1', 'tier2', 'DECK', 'CS3', 'all']


def _agent_lower(agent):
    return agent if agent in ("Sulfur", "BC") else agent.lower()


def load_experiments_module(agent):
    spec = importlib.util.spec_from_file_location(
        f"_exp_mod_{agent}",
        PROJECT_ROOT / "scripts" / "supplementary_notebooks" / AGENT_CONFIG[agent]["module"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_group_defs(agent):
    """Per-group init_cond/T/filter_hist, read from the agent's own
    EXPERIMENTS dict - NOT a tunable hyperparameter, just which
    scenarios/initial-condition each group optimizes against."""
    exp_mod = load_experiments_module(agent)
    return {g: {"init_cond": exp_mod.EXPERIMENTS[g]["init_cond"],
                "T": exp_mod.EXPERIMENTS[g]["T"],
                "filter_hist": exp_mod.EXPERIMENTS[g]["filter_hist"]}
            for g in GROUPS}


def load_configs(agent, baseline_config_path):
    unified_path = Path(f"data/SI_results/hp_retune/{agent}/best_config_unified.json")
    unified = json.load(open(unified_path))["config"]
    baseline = json.load(open(baseline_config_path))["config"]
    return unified, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=list(AGENT_CONFIG), required=True)
    parser.add_argument("--baseline-config-path", required=True,
                         help="Path to the baseline config JSON to use for this agent "
                              "(Stage C's transfer-check verdict decides which one)")
    parser.add_argument("--group", choices=GROUPS, default=None,
                         help="Regenerate only this group's checkpoint (default: run all, in order)")
    parser.add_argument("--seed", type=int, nargs="+", default=list(range(50)),
                         help="One or more seeds for params0/baseline init (default 0-49, matching "
                              "CO2's Stage 6a UQ protocol). Pass a single value (e.g. --seed 3) for "
                              "one seed - the natural unit for a SLURM array task.")
    args = parser.parse_args()

    agent = args.agent
    cfg = AGENT_CONFIG[agent]
    agent_lower = _agent_lower(agent)
    checkpoint_dir = f"checkpoints/{agent_lower}_retuned"
    seed_sweep_dir = f"{checkpoint_dir}/seed_sweep"

    unified_cfg, baseline_cfg = load_configs(agent, args.baseline_config_path)
    group_defs = build_group_defs(agent)
    print(f"[{agent}] Unified optimizer config: {unified_cfg}")
    print(f"[{agent}] Baseline config ({args.baseline_config_path}): {baseline_cfg}")

    os.makedirs(seed_sweep_dir, exist_ok=True)

    to_run = [args.group] if args.group else GROUPS

    for seed in args.seed:
        print(f"=== [{agent}] seed {seed} ===")

        # Only one task writes the shared baseline (identical content every
        # time - same seed, same baseline hyperparameters) - avoids N
        # parallel writers racing on the same file. Matches the CO2 script's
        # convention: H-ext (or a no-`--group` run) writes it.
        write_baseline = (args.group is None) or (args.group == "H-ext")
        baseline_save_path = f"{seed_sweep_dir}/baseline_{cfg['tag']}_seed{seed}.pkl" if write_baseline else None

        setup = utils_inverse.run_inverse_experiment_setup(
            cfg["agents"], cfg["active_agents"], mode=MODE,
            CS3=True, DAMIP=False, GeoMIP=False,
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
                checkpoint_dir=seed_sweep_dir,
                tag=f"{cfg['tag']}_seed{seed}",
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
