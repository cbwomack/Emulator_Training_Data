#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage 6o, part 2: score each of Stage 6o's 150 (seed, init_cond) optimization
runs and boil each one down to the handful of arrays Figure 7 v2 actually
needs.

Two jobs, both per (seed, init_cond):

  1. Skill. Retrain a fresh emulator on that run's final optimized emissions
     and evaluate it against every eval set (evaluate_optimal_emulator), giving
     the per-scenario NRMSE used to rank seeds for the 25th/50th/75th
     percentile trajectory pick. The ranking statistic is the weighted overall
     NRMSE, Tier 1:7 / Tier 2:5 / DECK:2 / CS3:2 - the same weighting
     convention as 4c_MESM_baseline_hp_search.py, 6l_fig7_scalar_config_swap.py
     and 6m_fig7_seed_selection_variants.ipynb. That seed's own baseline
     emulator is scored alongside it, from the per-seed baseline checkpoint
     Stage 6o wrote.

  2. Extraction. Pull the final optimized CO2 trajectory, the SCM temperature
     response it produces, and the penalty-corrected NRMSE-vs-update curve out
     of the checkpoint. This is what makes the results portable: a full
     2000-update checkpoint is ~30 MB (~4.5 GB across all 150 runs), but the
     figure only needs ~15 KB of it. The collected cache is a couple of MB, so
     the checkpoints can stay on the cluster.

Note on the temperature response: Figure 7's original panels plotted ΔT from
real MESM ensemble runs driven by the optimized emissions. No such runs exist
for these newly-optimized trajectories - commissioning them is out of scope -
so `delT` here is the MESM-calibrated SCM's own response, taken from the
checkpoint's `train_temp_traj` (the objective's `train_temp_raw`, i.e. exactly
the temperature the optimizer itself saw). Panels built from this are SCM
temperature, not MESM temperature; the figure and its caption must say so.

Usage:
    python scripts/6o_fig7_MESM_opt_evaluate.py --mode run-one --seed 0 --init-cond constant
    python scripts/6o_fig7_MESM_opt_evaluate.py --mode collect
"""
import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import jax

import utils_inverse

AGENTS = ['CO2']
ACTIVE_AGENTS = ('CO2',)
MODE = 'MESM_tier1'
GROUP = 'all'
TAG = 'co2_only_MESM'

SEED_SWEEP_DIR = Path('checkpoints/co2_MESM_retuned/seed_sweep')
PARTIAL_DIR = Path('data/SI_results/seed_uncertainty/partial_fig7v2_MESM')
FINAL_PATH = Path('data/SI_results/seed_uncertainty/fig7v2_seed_spread_MESM_tier1.pkl')

UNIFIED_CONFIG_PATH = Path('data/SI_results/hp_retune/best_config_unified.json')
BASELINE_CONFIG_PATH = Path('data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json')

INIT_CONDS = ['constant', 'sine', 'gaussian']
N_SEEDS = 50

# Figure 4 / Figure 7's scenario-group weighting, used for the single
# "average skill over all scenarios" number each seed is ranked by.
WEIGHTS = {"Tier 1": 7, "Tier 2": 5, "DECK": 2, "CS3": 2}


def load_configs():
    unified = json.load(open(UNIFIED_CONFIG_PATH))["config"]
    baseline = json.load(open(BASELINE_CONFIG_PATH))["config"]
    return unified, baseline


def weighted_score(results: dict) -> float:
    """Overall NRMSE for one emulator: Tier 1:7 / Tier 2:5 / DECK:2 / CS3:2."""
    return sum(WEIGHTS[s] * results[s]['mean'] for s in WEIGHTS) / sum(WEIGHTS.values())


def _ckpt_path(seed, init_cond):
    return SEED_SWEEP_DIR / f'inverse_{init_cond}_{GROUP}_{TAG}_seed{seed}.pkl'


def _baseline_path(seed):
    return SEED_SWEEP_DIR / f'baseline_{TAG}_seed{seed}.pkl'


def _partial_path(seed, init_cond):
    return PARTIAL_DIR / f'{init_cond}_seed{seed}.pkl'


def run_one(seed, init_cond, unified_cfg, baseline_cfg):
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _partial_path(seed, init_cond)
    if out_path.exists():
        print(f"[{init_cond} seed={seed}] already done, skipping")
        return

    ckpt_path = _ckpt_path(seed, init_cond)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} missing - run scripts/6o_fig7_MESM_opt_seed_sweep.py "
            f"--seed {seed} --init-cond {init_cond} first")

    # --- Setup: params0 + eval_sets + this seed's own baseline emulator. Same
    # seed drives both sides of the comparison (the Stage 0 fairness property).
    setup = utils_inverse.run_inverse_experiment_setup(
        AGENTS, ACTIVE_AGENTS, mode=MODE,
        CS3=True, DAMIP=False, GeoMIP=False,
        idx_demo=None, seed=seed,
        baseline_save_path=None,  # already written by Stage 6o; do not rewrite
        baseline_K=baseline_cfg["K"], baseline_lr=baseline_cfg["lr"],
        baseline_weight_decay=baseline_cfg["weight_decay"],
    )

    # --- 1. Skill of the optimized emulator.
    # K/lr/weight_decay are the INNER-loop hyperparameters that actually
    # produced this checkpoint, not the baseline's own - see
    # evaluate_optimal_emulator's docstring on why a mismatch here silently
    # invalidates the comparison.
    optimal_results = utils_inverse.evaluate_optimal_emulator(
        training_paths=[str(ckpt_path)],
        train_scenarios=['Opt. All'],
        eval_sets=setup["eval_sets"],
        params0=setup["params0"],
        active_agents=ACTIVE_AGENTS,
        inactive_mode="zeros",
        historical_name="historical",
        key=jax.random.PRNGKey(seed),
        K=unified_cfg["K_inner"],
        lr=unified_cfg["lr_inner"],
        weight_decay=unified_cfg["wd_inner"],
        batch_size=unified_cfg["batch_size"],
        mode=MODE,
    )['Opt. All']

    baseline_results = setup["baseline_results"]

    # --- 2. Compact extraction from the checkpoint.
    raw = utils_inverse.load_inverse_ckpt(str(ckpt_path))

    emissions = np.asarray(raw["U_traj"][-1]["CO2"], dtype=np.float32)

    # Penalty-corrected NRMSE curve, not the raw objective (which is
    # NRMSE + smoothness_weight * sum(dU)^2). smoothness_weight is read from
    # the checkpoint's own recorded meta, so the correction is exact.
    nrmse_traj = np.asarray(utils_inverse.recover_nrmse_trajectory(raw), dtype=np.float32)

    # train_temp_traj is sampled every preds_every updates; [-1] is the final
    # one. Each entry is a list over training scenarios - there is exactly one
    # here (the optimized trajectory itself).
    delT = np.asarray(raw["train_temp_traj"][-1][0], dtype=np.float32).reshape(-1)

    entry = {
        "seed": seed,
        "init_cond": init_cond,
        "emissions": emissions,
        "delT": delT,
        "nrmse_traj": nrmse_traj,
        "updates_done": int(raw["step_count"]),
        "optimal_results": optimal_results,
        "baseline_results": baseline_results,
        "weighted_optimal": weighted_score(optimal_results),
        "weighted_baseline": weighted_score(baseline_results),
        "meta": raw.get("meta"),
    }

    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(entry, f)
    os.replace(tmp, out_path)
    print(f"[{init_cond} seed={seed}] weighted NRMSE: optimized "
          f"{entry['weighted_optimal']:.4f} vs baseline {entry['weighted_baseline']:.4f} "
          f"-> {out_path}")


def collect(n_seeds=N_SEEDS):
    merged = {ic: {} for ic in INIT_CONDS}
    missing = []
    lengths = {ic: set() for ic in INIT_CONDS}
    for ic in INIT_CONDS:
        for seed in range(n_seeds):
            p = _partial_path(seed, ic)
            if not p.exists():
                missing.append((ic, seed))
                continue
            with open(p, "rb") as f:
                entry = pickle.load(f)
            lengths[ic].add(len(entry["nrmse_traj"]))
            merged[ic][seed] = entry

    if missing:
        print(f"missing {len(missing)}/{n_seeds * len(INIT_CONDS)} runs, "
              f"e.g. {missing[:5]} - not writing the final cache yet")
        return

    # Checked PER initial condition, not globally. Different ICs are allowed to
    # run to different num_updates - each gets its own convergence panel, so a
    # non-uniform sweep (e.g. constant extended to 5000, sine/gaussian at 2000)
    # is a deliberate design, not an error. Within one IC it is still a hard
    # failure: mixing lengths there would average different run stages into a
    # single median/IQR band.
    ragged = {ic: sorted(v) for ic, v in lengths.items() if len(v) > 1}
    if ragged:
        raise ValueError(
            f"these initial conditions have seeds at different trajectory lengths: "
            f"{ragged} - some tasks timed out mid-run or were only partially resumed to a "
            f"new num_updates. Finish or re-run them before collecting.")

    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_PATH, "wb") as f:
        pickle.dump(merged, f)
    size_mb = FINAL_PATH.stat().st_size / 1e6
    print(f"collected {n_seeds} seeds x {len(INIT_CONDS)} ICs -> {FINAL_PATH} ({size_mb:.1f} MB)")

    print(f"\n=== Weighted overall NRMSE (Tier1:7/Tier2:5/DECK:2/CS3:2), median [IQR] ===")
    base = np.array([merged[INIT_CONDS[0]][s]["weighted_baseline"] for s in range(n_seeds)])
    print(f"{'baseline':10s} {np.median(base):.4f} "
          f"[{np.percentile(base, 25):.4f}, {np.percentile(base, 75):.4f}]")
    for ic in INIT_CONDS:
        arr = np.array([merged[ic][s]["weighted_optimal"] for s in range(n_seeds)])
        n_better = int((arr < base).sum())
        n_upd = merged[ic][0]["updates_done"]
        print(f"{ic:10s} {np.median(arr):.4f} "
              f"[{np.percentile(arr, 25):.4f}, {np.percentile(arr, 75):.4f}]  "
              f"beats baseline in {n_better}/{n_seeds} seeds   ({n_upd} updates)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["run-one", "collect"], required=True)
    parser.add_argument("--seed", type=int, default=None, help="run-one only")
    parser.add_argument("--init-cond", choices=INIT_CONDS, default=None, help="run-one only")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS, help="collect only")
    args = parser.parse_args()

    if args.mode == "collect":
        collect(args.n_seeds)
        return

    if args.seed is None or args.init_cond is None:
        raise SystemExit("run-one requires both --seed and --init-cond")

    unified_cfg, baseline_cfg = load_configs()
    run_one(args.seed, args.init_cond, unified_cfg, baseline_cfg)


if __name__ == "__main__":
    main()
