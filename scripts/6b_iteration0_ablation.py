#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage 6b - Point 2 (iteration-0 ablation): for Figures 2, 3, 4, 5, checks how
much of each figure's reported skill actually comes from the outer-loop
emissions optimization, versus what "iteration 0, no refinement" (i.e. just
the initial emissions guess, fit through the same inner loop once) would
already have given.

Two parts:

(a) --mode extract (free, no new compute): every existing checkpoint already
    stores U_traj[0]/errors[0] - the initial-condition trajectory and its
    loss, from *before* the first outer-loop update (see
    utils_inverse.optimize_emissions_inverse's "Initialization" branch).
    Reads every single-forcing agent's regenerated seed-sweep checkpoints
    (checkpoints/{agent}_retuned/seed_sweep/, all 50 seeds where available -
    this is genuinely free since the data's already on disk, so it gives a
    full iteration-0-vs-final mean+-spread comparison, not just a point
    estimate) plus whatever checkpoints/multi/*.pkl exist (single-run only,
    read-only, untouched - see module note below).

(b) --mode control (new, cheap compute - one task per (agent, group, ic)):
    explicit constant/sine/gaussian initial-condition controls, evaluated the
    same way the real runs were (run_inverse_experiment_setup +
    run_inverse_experiment) but with num_updates=0, so only a single
    inner-loop fit happens and no outer-loop refinement - the "no
    optimization at all, just this IC" control the reviewer asked for. Only
    runs the IC(s) that do NOT already match the group's own native
    init_cond (no point re-deriving what (a) already gives for free). Uses
    seed=0 only (a bounded point-estimate control, not a full seed sweep -
    the seed-uncertainty question is Stage 6a's, not this stage's).

Scope note on checkpoints/multi/: Figure 5 loads
inverse_constant_tier1_all_agents_subset2.pkl, but 3b_inverse_all_agents.py's
EXPERIMENTS dict (the only documented source of multi-agent hyperparameters)
has no 'tier1' entry at all - it only has H-ext/DAMIP/GeoMIP/all. Guessing at
the missing multi-agent 'tier1' hyperparameters to build a (b)-style IC
control for Figures 4/5's multi-agent panels risks feeding a fabricated
number into the reviewer response (the same concern already flagged for
Stage 6d's MESM time-axis question), so this script covers Figures 4/5's
multi-agent components with (a) only (read the existing checkpoint,
unmodified) and leaves a (b)-style control for those panels as an open
question rather than guessing.

Usage:
    python 6b_iteration0_ablation.py --mode extract
    python 6b_iteration0_ablation.py --mode control --agent CO2 --group H-ext --ic constant
    python 6b_iteration0_ablation.py --mode collect
"""
import os
import sys
import json
import pickle
import argparse
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import numpy as np

import utils_inverse

OUT_DIR = Path("data/SI_results/iteration0_ablation")
CONTROL_CKPT_DIR = Path("checkpoints/iteration0_controls")
N_SEEDS = 50
ICS = ["constant", "sine", "gaussian"]

# --- CO2 (checkpoints/co2_retuned/seed_sweep/), matching 0c_regenerate_checkpoints_co2.py ---
CO2_GROUP_DEFS = {
    'H-ext': {'init_cond': 'sine',     'T': 477, 'filter_hist': True},
    'tier1': {'init_cond': 'constant', 'T': 751, 'filter_hist': False},
    'tier2': {'init_cond': 'constant', 'T': 751, 'filter_hist': True},
    'DECK':  {'init_cond': 'constant', 'T': 751, 'filter_hist': False},
    'CS3':   {'init_cond': 'constant', 'T': 751, 'filter_hist': True},
    'all':   {'init_cond': 'constant', 'T': 751, 'filter_hist': False},
}

# --- CH4/N2O/Sulfur/BC (checkpoints/{agent_lower}_retuned/seed_sweep/), matching
# 0c_regenerate_checkpoints_agent.py ---
AGENT_CONFIG = {
    "CO2":    {"agents": ["CO2"],    "active_agents": ("CO2",),    "tag": "co2_only",
               "checkpoint_dir": "checkpoints/co2_retuned/seed_sweep",
               "unified_config_path": "data/SI_results/hp_retune/best_config_unified.json",
               "group_defs": CO2_GROUP_DEFS},
    "CH4":    {"agents": ["CH4"],    "active_agents": ("CH4",),    "tag": "ch4_only",
               "checkpoint_dir": "checkpoints/ch4_retuned/seed_sweep",
               "unified_config_path": "data/SI_results/hp_retune/CH4/best_config_unified.json",
               "module": "SIa_inverse_CH4_only.py"},
    "N2O":    {"agents": ["N2O"],    "active_agents": ("N2O",),    "tag": "n2o_only",
               "checkpoint_dir": "checkpoints/n2o_retuned/seed_sweep",
               "unified_config_path": "data/SI_results/hp_retune/N2O/best_config_unified.json",
               "module": "SIb_inverse_N2O_only.py"},
    "Sulfur": {"agents": ["Sulfur"], "active_agents": ("Sulfur",), "tag": "Sulfur_only",
               "checkpoint_dir": "checkpoints/Sulfur_retuned/seed_sweep",
               "unified_config_path": "data/SI_results/hp_retune/Sulfur/best_config_unified.json",
               "module": "SIc_inverse_Sulfur_only.py"},
    "BC":     {"agents": ["BC"],     "active_agents": ("BC",),     "tag": "BC_only",
               "checkpoint_dir": "checkpoints/BC_retuned/seed_sweep",
               "unified_config_path": "data/SI_results/hp_retune/BC/best_config_unified.json",
               "module": "SId_inverse_BC_only.py"},
    # Added for Table S2's Multi row (tab:supp_opt_v_dyn): the blocker noted
    # in the module docstring above (multi-agent tier1 hyperparameters
    # unknown) is resolved now that Figure 4/5's multi-agent panels have
    # their own dedicated 0c_regenerate_checkpoints_multi_fig4.py config -
    # reused verbatim here (tag/checkpoint_dir/unified_config_path all copied
    # from that script) so the (a)-extract Opt.Iter.0/Opt.Final numbers and
    # this script's (b)-control Unopt.Rich number come from the exact same
    # smoothed-arm checkpoints Figures 4/5 themselves report.
    "Multi":  {"agents": ["CO2", "CH4", "N2O", "Sulfur", "BC"],
               "active_agents": ("CO2", "CH4", "N2O", "Sulfur", "BC"), "tag": "multi_fig4",
               "checkpoint_dir": "checkpoints/multi_fig4_smooth/seed_sweep",
               "unified_config_path": "data/SI_results/hp_retune/multi/best_config_unified.json",
               "group_defs": {"tier1": {"init_cond": "constant", "T": 751, "filter_hist": False}}},
}
GROUPS = ['H-ext', 'tier1', 'tier2', 'DECK', 'CS3', 'all']

# Only these (agent, group) pairs actually feed Figures 2/3/4 - no need to
# extract/control every group for every agent.
FIG_GROUPS = {
    "CO2": GROUPS,                 # Fig 2 (H-ext), Fig 4 (tier1/tier2/DECK/CS3/all)
    "CH4": ["tier1"],               # Fig 3
    "N2O": ["tier1"],               # Fig 3
    "Sulfur": ["tier1"],            # Fig 3
    "BC": ["tier1"],                # Fig 3
    "Multi": ["tier1"],             # Table S2 (tab:supp_opt_v_dyn)
}

# Multi-agent checkpoints referenced by Figs 4/5 - (a)-only, read verbatim,
# never regenerated (see module docstring).
#
# 2026-08-27: repointed. The old single-run files (checkpoints/multi/inverse_
# constant_{all,tier1}_all_agents_subset{,2}.pkl) no longer exist - Figures 4/5
# moved to the 50-seed sweeps during the 2000-iteration migration, and this
# block had been silently printing "not found, skipping" ever since, leaving
# the multi-agent panels with no iteration-0 comparand at all.
#
# Both arms are extracted because they answer different questions and the
# choice belongs at plot time, not here: the unsmoothed families back the SI
# "pure noise as forcing" argument, the smoothed ones back Figures 5/6.
# Extracting both is free (these files are already on disk) and avoids
# guessing which arm a given panel will end up reporting.
MULTI_FAMILIES = {
    "Fig4_multi":        {"dir": "checkpoints/multi_fig4/seed_sweep",
                          "tag": "multi_fig4",
                          "groups": ["tier1", "tier2", "DECK", "CS3", "all"]},
    "Fig4_multi_smooth": {"dir": "checkpoints/multi_fig4_smooth/seed_sweep",
                          "tag": "multi_fig4",
                          "groups": ["tier1", "tier2", "DECK", "CS3", "all"]},
    "Fig5_multi":        {"dir": "checkpoints/multi_retuned/seed_sweep",
                          "tag": "all_agents",
                          "groups": ["tier1", "H-ext", "all"]},
    "Fig5_multi_smooth": {"dir": "checkpoints/multi_retuned_smooth/seed_sweep",
                          "tag": "all_agents",
                          "groups": ["tier1", "H-ext", "all"]},
}


def _load_experiments_module(agent):
    spec = importlib.util.spec_from_file_location(
        f"_exp_mod_{agent}",
        PROJECT_ROOT / "scripts" / "supplementary_notebooks" / AGENT_CONFIG[agent]["module"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def group_defs_for(agent):
    cfg = AGENT_CONFIG[agent]
    if "group_defs" in cfg:
        return cfg["group_defs"]
    exp_mod = _load_experiments_module(agent)
    return {g: {"init_cond": exp_mod.EXPERIMENTS[g]["init_cond"],
                "T": exp_mod.EXPERIMENTS[g]["T"],
                "filter_hist": exp_mod.EXPERIMENTS[g]["filter_hist"]}
            for g in GROUPS}


def unified_config_for(agent):
    return json.load(open(AGENT_CONFIG[agent]["unified_config_path"]))["config"]


# ----------------------------------------------------------------------
# (a) extract: read-only, iteration-0 vs. final, across every seed
# ----------------------------------------------------------------------
def extract():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"single_forcing": {}, "multi": {}}

    for agent, groups in FIG_GROUPS.items():
        cfg = AGENT_CONFIG[agent]
        ckpt_dir = Path(cfg["checkpoint_dir"])
        tag = cfg["tag"]
        gdefs = group_defs_for(agent)
        for group in groups:
            native_ic = gdefs[group]["init_cond"]
            iter0, final = [], []
            iter0_obj, final_obj = [], []
            missing = []
            for seed in range(N_SEEDS):
                p = ckpt_dir / f"inverse_{native_ic}_{group}_{tag}_seed{seed}.pkl"
                if not p.exists():
                    missing.append(seed)
                    continue
                ckpt = utils_inverse.load_inverse_ckpt(str(p))
                # errors[] stores the FULL objective (NRMSE + w * penalty(U)),
                # not the error term. At iteration 0 a constant IC has dU=0 so
                # the two coincide, but the final entry does not whenever w>0
                # (CO2 1e-6, N2O 3e-6, BC 3e-5, Sulfur/multi smoothed arms).
                # Reading errors[-1] as NRMSE therefore understates the
                # improvement for exactly the penalized agents. Same root cause
                # as the 6c OOD self-check bug (REVISIONS.md, 2026-08-27).
                nrmse = utils_inverse.recover_nrmse_trajectory(ckpt)
                iter0.append(float(nrmse[0]))
                final.append(float(nrmse[-1]))
                iter0_obj.append(float(ckpt["errors"][0]))
                final_obj.append(float(ckpt["errors"][-1]))
            results["single_forcing"][(agent, group)] = {
                "iteration0_errors": iter0, "final_errors": final, "missing_seeds": missing,
                "iteration0_objective": iter0_obj, "final_objective": final_obj,
            }
            print(f"[extract] {agent}/{group}: {len(iter0)}/{N_SEEDS} seeds "
                  f"(iter0 mean={sum(iter0)/len(iter0):.4f}, final mean={sum(final)/len(final):.4f})"
                  if iter0 else f"[extract] {agent}/{group}: no seeds found yet")

    for name, fam in MULTI_FAMILIES.items():
        ckpt_dir = Path(fam["dir"])
        if not ckpt_dir.is_dir():
            print(f"[extract] {name}: {ckpt_dir} not found, skipping family")
            continue
        for group in fam["groups"]:
            iter0, final, iter0_obj, final_obj, missing = [], [], [], [], []
            for seed in range(N_SEEDS):
                p = ckpt_dir / f"inverse_constant_{group}_{fam['tag']}_seed{seed}.pkl"
                if not p.exists():
                    missing.append(seed)
                    continue
                ckpt = utils_inverse.load_inverse_ckpt(str(p))
                nrmse = utils_inverse.recover_nrmse_trajectory(ckpt)
                iter0.append(float(nrmse[0]))
                final.append(float(nrmse[-1]))
                iter0_obj.append(float(ckpt["errors"][0]))
                final_obj.append(float(ckpt["errors"][-1]))
            if not iter0:
                print(f"[extract] {name}/{group}: no seeds found, skipping")
                continue
            results["multi"][(name, group)] = {
                "iteration0_errors": iter0, "final_errors": final, "missing_seeds": missing,
                "iteration0_objective": iter0_obj, "final_objective": final_obj,
                "checkpoint_dir": str(ckpt_dir),
            }
            print(f"[extract] {name}/{group}: {len(iter0)}/{N_SEEDS} seeds "
                  f"(iter0 mean={sum(iter0)/len(iter0):.4f}, "
                  f"final mean={sum(final)/len(final):.4f})")

    out_path = OUT_DIR / "extract_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[extract] wrote {out_path}")


# ----------------------------------------------------------------------
# (b) control: one (agent, group, ic) SLURM-array task, seed=0, num_updates=0
# ----------------------------------------------------------------------
def control(agent, group, ic, seed=0):
    CONTROL_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = AGENT_CONFIG[agent]
    gdefs = group_defs_for(agent)
    gdef = gdefs[group]
    if ic == gdef["init_cond"]:
        print(f"[control] {agent}/{group}/{ic}: matches the group's native init_cond, "
              f"already covered for free by --mode extract, skipping")
        return

    unified_cfg = unified_config_for(agent)
    # CO2's own baseline search predates the per-agent Phase 0 extension and
    # lives at a path with no agent suffix; every other agent has its own
    # dense K=400 search (Stage 6e confirmed CO2's tuned baseline config does
    # NOT transfer to the other four agents - see REVISIONS.md - so there is
    # no cross-agent fallback here: an agent with no search of its own yet
    # raises rather than silently substituting CO2's untransferable config).
    # k400_search_{agent} preserves each single-forcing agent's own casing
    # (k400_search_CH4/N2O/Sulfur/BC) but the multi-agent search directory is
    # lowercase (k400_search_multi) - matches 0c_regenerate_checkpoints_
    # multi_fig4.py's BASELINE_CONFIG_PATH exactly.
    if agent == "CO2":
        baseline_cfg_path = Path("data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json")
    elif agent == "Multi":
        baseline_cfg_path = Path("data/SI_results/baseline_hp/k400_search_multi/best_baseline_config_K400.json")
    else:
        baseline_cfg_path = Path(f"data/SI_results/baseline_hp/k400_search_{agent}/best_baseline_config_K400.json")
    if not baseline_cfg_path.exists():
        raise FileNotFoundError(
            f"{baseline_cfg_path} missing - {agent}'s own K=400 baseline search "
            f"(scripts/6e_baseline_hp_search_k400{'_agent' if agent != 'CO2' else ''}.py) "
            f"hasn't finished yet"
        )
    baseline_cfg = json.load(open(baseline_cfg_path))["config"]

    tag = f"{cfg['tag']}_ic-{ic}_seed{seed}"
    out_path = CONTROL_CKPT_DIR / f"inverse_{ic}_{group}_{tag}.pkl"
    if out_path.exists():
        print(f"[control] {out_path} already exists, skipping")
        return

    setup = utils_inverse.run_inverse_experiment_setup(
        cfg["agents"], cfg["active_agents"], mode="FaIR",
        CS3=True, DAMIP=False, GeoMIP=False,
        idx_demo=None, seed=seed,
        baseline_save_path=None,
        baseline_K=baseline_cfg["K"], baseline_lr=baseline_cfg["lr"],
        baseline_weight_decay=baseline_cfg["weight_decay"],
    )
    # NOTE: run_inverse_experiment/optimize_emissions_inverse only writes
    # checkpoint_path from *inside* its per-step update loop - with
    # num_updates=0 that loop body never executes, so nothing is ever
    # persisted to checkpoint_dir/checkpoint_path despite passing one in
    # (confirmed directly: the returned dict is correct, but no file lands on
    # disk). Save the result ourselves instead of relying on that mechanism.
    result = utils_inverse.run_inverse_experiment(
        setup,
        group=group,
        checkpoint_dir=str(CONTROL_CKPT_DIR),
        tag=tag,
        num_updates=0,
        step_size=unified_cfg["step_size"],
        momentum=unified_cfg["momentum"],
        nesterov=unified_cfg["nesterov"],
        K_inner=unified_cfg["K_inner"],
        lr_inner=unified_cfg["lr_inner"],
        wd_inner=unified_cfg["wd_inner"],
        smoothness_weight=unified_cfg["smoothness_weight"],
        batch_size=unified_cfg["batch_size"],
        init_cond=ic,
        T=gdef["T"],
        filter_hist=gdef["filter_hist"],
        checkpoint_every=50,
        resume_if_exists=False,  # nothing on disk to resume from - see note above
        preds_every=50,
        key=jax.random.PRNGKey(seed),
    )
    # errors[0] is the OBJECTIVE (NRMSE + w * penalty(U_init)), not the error
    # term. For ic="constant" dU=0 so the two coincide, but for sine/gaussian
    # they do not. Files written before 2026-08-27 stored `errors` alone, so
    # that contamination is unrecoverable from disk after the fact (there is no
    # U_traj and no meta to reconstruct the penalty from). It is small - roughly
    # 1% for CO2 at w=1e-6, negligible for BC at w=3e-5 - so the existing
    # controls were kept rather than re-run. Persist enough to correct it from
    # now on.
    with open(out_path, "wb") as f:
        pickle.dump({
            "errors": np.asarray(result["errors"]),
            "agent": agent, "group": group, "ic": ic, "seed": seed,
            "smoothness_weight": unified_cfg["smoothness_weight"],
            "penalty_form": unified_cfg.get("penalty_form", "legacy"),
            "U_init": jax.tree_util.tree_map(np.asarray, result["U_traj"][0])
                      if result.get("U_traj") else None,
        }, f)
    print(f"[control] {agent}/{group}/{ic}/seed{seed} -> {out_path} "
          f"(iteration0 error={float(result['errors'][0]):.4f})")


def all_control_group_combos():
    """(agent, group, ic) triples, independent of seed."""
    combos = []
    for agent, groups in FIG_GROUPS.items():
        gdefs = group_defs_for(agent)
        for group in groups:
            native_ic = gdefs[group]["init_cond"]
            for ic in ICS:
                if ic != native_ic:
                    combos.append((agent, group, ic))
    return combos


def all_control_combos(seeds=range(N_SEEDS)):
    """(agent, group, ic, seed) - the full seed-swept control set (matches
    Stage 6a/E's 50-seed UQ protocol, per user direction: iteration-0 controls
    must report seed ranges like every other result, not a single point
    estimate)."""
    return [(agent, group, ic, seed)
            for (agent, group, ic) in all_control_group_combos()
            for seed in seeds]


def collect():
    extract_path = OUT_DIR / "extract_results.pkl"
    if not extract_path.exists():
        raise FileNotFoundError(f"{extract_path} missing - run --mode extract first")
    with open(extract_path, "rb") as f:
        results = pickle.load(f)

    # Carry forward any controls already collected. checkpoints/iteration0_
    # controls/ is scratch and gets cleaned between sessions, but the (b)-mode
    # results it produced cost real compute and live only in this output file.
    # Without this, re-running collect purely to refresh the (a)-mode extract
    # half - which is exactly why it is re-run after a migration - would
    # silently replace 20 populated control entries with empty lists.
    out_path = OUT_DIR / "iteration0_ablation_results.pkl"
    prior_controls = {}
    if out_path.exists():
        with open(out_path, "rb") as f:
            prior_controls = (pickle.load(f) or {}).get("controls", {}) or {}

    control_results = {}
    carried = 0
    for agent, group, ic in all_control_group_combos():
        errs, missing_seeds = [], []
        for seed in range(N_SEEDS):
            tag = f"{AGENT_CONFIG[agent]['tag']}_ic-{ic}_seed{seed}"
            p = CONTROL_CKPT_DIR / f"inverse_{ic}_{group}_{tag}.pkl"
            if not p.exists():
                missing_seeds.append(seed)
                continue
            with open(p, "rb") as f:
                ckpt = pickle.load(f)
            errs.append(float(ckpt["errors"][0]))
        key = (agent, group, ic)
        prior = prior_controls.get(key)
        if not errs and prior and prior.get("iteration0_errors"):
            control_results[key] = dict(prior)
            control_results[key]["carried_forward"] = True
            carried += 1
            print(f"[collect] {agent}/{group}/{ic}: no checkpoints on disk - "
                  f"carried forward {len(prior['iteration0_errors'])} prior seeds")
            continue
        control_results[key] = {"iteration0_errors": errs, "missing_seeds": missing_seeds}
        status = f"{len(errs)}/{N_SEEDS} seeds"
        if errs:
            status += f" (mean={sum(errs)/len(errs):.4f})"
        print(f"[collect] {agent}/{group}/{ic}: {status}")

    with open(out_path, "wb") as f:
        pickle.dump({"extract": results, "controls": control_results}, f)
    print(f"[collect] wrote {out_path}"
          + (f" ({carried} control entries carried forward from the previous "
             f"collect, not recomputed)" if carried else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["extract", "control", "collect", "list-combos"], required=True)
    parser.add_argument("--agent", choices=list(AGENT_CONFIG), default=None)
    parser.add_argument("--group", choices=GROUPS, default=None)
    parser.add_argument("--ic", choices=ICS, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "extract":
        extract()
    elif args.mode == "control":
        if not (args.agent and args.group and args.ic):
            raise SystemExit("--mode control requires --agent --group --ic")
        control(args.agent, args.group, args.ic, seed=args.seed)
    elif args.mode == "collect":
        collect()
    elif args.mode == "list-combos":
        for c in all_control_combos():
            print(c)


if __name__ == "__main__":
    main()
