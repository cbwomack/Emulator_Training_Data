#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
regenerate CH4/N2O/Sulfur/BC
checkpoints , using the hyperparameter search's per-agent unified optimizer
config (data/SI_results/hp_retune/{agent}/best_config_unified.json)

Writes fresh checkpoints to a NEW checkpoints/{agent_lower}_retuned/
directory

Multi-seed by default (50 seeds);
--seed accepts one or more seeds.

Usage:
    python 03b_regenerate_checkpoints_agent.py --agent CH4 \\
        --baseline-config-path data/SI_results/baseline_hp/k400_search_CH4/best_baseline_config_K400.json
    python 03b_regenerate_checkpoints_agent.py --agent CH4 --group H-ext --seed 3 \\
        --baseline-config-path...   # one group, one seed (SLURM array task)
"""
import os
import sys
import json
import importlib.util
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import argparse
import jax

import utils_inverse

MODE = 'FaIR'
NUM_UPDATES = 2000

AGENT_CONFIG = {
    "CH4":    {"module": "04c_inverse_CH4_only.py", "agents": ["CH4"],    "active_agents": ("CH4",),    "tag": "ch4_only"},
    "N2O":    {"module": "04d_inverse_N2O_only.py", "agents": ["N2O"],    "active_agents": ("N2O",),    "tag": "n2o_only"},
    "Sulfur": {"module": "04e_inverse_Sulfur_only.py", "agents": ["Sulfur"], "active_agents": ("Sulfur",), "tag": "Sulfur_only"},
    "BC":     {"module": "04f_inverse_BC_only.py", "agents": ["BC"],     "active_agents": ("BC",),     "tag": "BC_only"},
}
GROUPS = ['H-ext', 'tier1', 'tier2', 'DECK', 'CS3', 'all']


def _agent_lower(agent):
    return agent if agent in ("Sulfur", "BC") else agent.lower()


def load_experiments_module(agent):
    spec = importlib.util.spec_from_file_location(
        f"_exp_mod_{agent}",
        PROJECT_ROOT / "pipeline" / "04_optimization" / AGENT_CONFIG[agent]["module"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_group_defs(agent):
    """Per-group init_cond/T/filter_hist, read from the agent's own
    EXPERIMENTS dict"""
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
                              "(the transfer check's transfer-check verdict decides which one)")
    parser.add_argument("--group", choices=GROUPS, default=None,
                         help="Regenerate only this group's checkpoint (default: run all, in order)")
    parser.add_argument("--seed", type=int, nargs="+", default=list(range(50)),
                         help="One or more seeds for params0/baseline init (default 0-49, matching "
                              "CO2's the seed sweep UQ protocol). Pass a single value (e.g. --seed 3) for "
                              "one seed - the natural unit for a SLURM array task.")
    parser.add_argument("--smoothness-weight", type=float, default=None,
                         help="Override the tuned config's smoothness_weight. Pass this to "
                              "build a smoothed arm; combine with --penalty-form normalized "
                              "and --out-dir so the unsmoothed arm on disk is left intact.")
    parser.add_argument("--penalty-form", choices=("legacy", "normalized"), default="legacy",
                         help="Which smoothness penalty to apply. 'legacy' is the historical "
                              "unnormalized sum and is the default so existing behaviour is "
                              "unchanged; 'normalized' is the dimensionless, agent-count- and "
                              "length-normalized form (see utils_inverse.smoothness_penalty_terms).")
    parser.add_argument("--out-dir", default=None,
                         help="Override the checkpoint directory. Required in practice when "
                              "--smoothness-weight is given: writing a differently-regularized "
                              "run into the default directory would overwrite the arm it is "
                              "meant to be compared against.")
    args = parser.parse_args()

    agent = args.agent
    cfg = AGENT_CONFIG[agent]
    agent_lower = _agent_lower(agent)
    checkpoint_dir = args.out_dir or f"checkpoints/{agent_lower}_retuned"
    seed_sweep_dir = f"{checkpoint_dir}/seed_sweep"

    unified_cfg, baseline_cfg = load_configs(agent, args.baseline_config_path)
    group_defs = build_group_defs(agent)
    print(f"[{agent}] Unified optimizer config: {unified_cfg}")
    print(f"[{agent}] Baseline config ({args.baseline_config_path}): {baseline_cfg}")

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
            f"Pass --out-dir to keep the existing unsmoothed arm intact.")

    os.makedirs(seed_sweep_dir, exist_ok=True)

    to_run = [args.group] if args.group else GROUPS

    for seed in args.seed:
        print(f"=== [{agent}] seed {seed} ===")
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
