#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Reviewer-revision script (Point 1: uncertainty quantification). Companion for
6a_seed_uncertainty_sweep.ipynb.

Reproduces the SCM-side quantitative results behind Figs 3, 4, 6 (Fig 7's
MESM half is cluster-driven and out of scope here) across multiple MLP-init
seeds, so the manuscript can report mean +/- spread instead of a single-seed
point estimate. Every run in this pipeline is otherwise deterministic given
its seed (no minibatching, no stochastic outer loop) - see REVISIONS.md for
why the seed is the only source of run-to-run variance that exists here.

Usage:
    # Pilot: time one cheap emulator-only retrain and one full outer-loop
    # reoptimization for real, before committing to seed counts.
    python 6a_seed_uncertainty_sweep.py --mode pilot

    # Full sweep, expensive axis (independent re-optimizations), CO2-only
    # group first per the user's phased rollout (watch for optimization
    # instability before extending to other agent groups):
    python 6a_seed_uncertainty_sweep.py --mode sweep --axis expensive --script 3a_inverse_CO2_only.py --n-expensive-seeds 5

    # Cheap axis (emulator-only retrains on an already-optimized trajectory):
    python 6a_seed_uncertainty_sweep.py --mode sweep --axis cheap --script 3a_inverse_CO2_only.py --n-cheap-seeds 20
"""
import os
import sys
import csv
import time
import pickle
import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless/unattended run: never let a plt.show() block on a GUI window

import numpy as np
import jax

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import utils_inverse

OUT_DIR = Path("data/SI_results/seed_uncertainty")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cheapest of 3a_inverse_CO2_only.py's experiment groups (T=477 vs. 751 for
# the others) - used as the pilot's representative "full outer-loop
# reoptimization" cost.
PILOT_EXPERIMENT = {
    'group': 'H-ext',
    'tag': 'co2_only',
    'num_updates': 1000,
    'step_size': 1000.0,
    'momentum': 0.9,
    'nesterov': True,
    'K_inner': 400,
    'lr_inner': 0.05,
    'wd_inner': 0.01,
    'init_cond': 'sine',
    'T': 477,
    'filter_hist': True,
    'smoothness_weight': 1e-05,
    'checkpoint_every': 50,
    'resume_if_exists': False,  # pilot: always a fresh timing run
    'preds_every': 50,
}


def run_pilot():
    print("=== Timing pilot: cheap emulator-only retrain ===")
    t0 = time.perf_counter()
    setup = utils_inverse.run_inverse_experiment_setup(
        ['CO2'], ('CO2',), mode='FaIR', CS3=True, DAMIP=False, GeoMIP=False,
        baseline_save_path=None, seed=0, idx_demo=None,
    )
    t_cheap = time.perf_counter() - t0
    print(f"generate baseline + eval_sets (seed=0): {t_cheap:.2f}s")

    t0 = time.perf_counter()
    _ = utils_inverse.generate_and_eval_baseline_emulator(
        setup["eval_sets"]["Tier 1"], setup["eval_sets"], save_path=None,
        verbose=False, hidden_sizes=[16], mode='FaIR', seed=1,
    )
    t_cheap2 = time.perf_counter() - t0
    print(f"re-retrain baseline only, new seed (seed=1): {t_cheap2:.2f}s")

    print("\n=== Timing pilot: one full outer-loop reoptimization ===")
    print(f"Experiment: {PILOT_EXPERIMENT['group']} (T={PILOT_EXPERIMENT['T']}, "
          f"num_updates={PILOT_EXPERIMENT['num_updates']}, K_inner={PILOT_EXPERIMENT['K_inner']})")
    t0 = time.perf_counter()
    utils_inverse.run_inverse_experiment(
        setup, checkpoint_dir=str(OUT_DIR / "pilot_checkpoints"), **PILOT_EXPERIMENT,
    )
    t_expensive = time.perf_counter() - t0
    print(f"full reoptimization (1 seed): {t_expensive:.2f}s ({t_expensive/60:.2f} min)")

    summary = {
        "baseline_setup_s": t_cheap,
        "baseline_retrain_only_s": t_cheap2,
        "full_reoptimization_s": t_expensive,
        "pilot_experiment": PILOT_EXPERIMENT,
    }
    with open(OUT_DIR / "pilot_timing.pkl", "wb") as f:
        pickle.dump(summary, f)

    print("\n=== Summary (also saved to data/SI_results/seed_uncertainty/pilot_timing.pkl) ===")
    for k, v in summary.items():
        if k != "pilot_experiment":
            print(f"  {k}: {v:.2f}s")
    return summary


def load_experiments_module(script_path: Path):
    """
    Load one of the existing 3x/SIx companion scripts (e.g.
    3a_inverse_CO2_only.py) as a module, to reuse its exact EXPERIMENTS dict/
    AGENTS/ACTIVE_AGENTS/MODE/CHECKPOINT_DIR rather than re-transcribing tuned
    hyperparameters by hand (same "extracted, never hand-retyped" principle
    used when those scripts were first generated - see PROGRESS.md Phase 3).
    Uses importlib.util directly since these filenames aren't valid `import`
    identifiers (leading digit).
    """
    spec = importlib.util.spec_from_file_location("_experiments_module", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_instability(result: dict, label: str) -> bool:
    """
    Flag genuine numerical divergence (NaN/Inf in the loss trajectory or the
    final emissions trajectory) - distinct from a large-but-finite failure
    value, which is an expected, legitimate outcome in some configs (that's
    what reviewer Point 6's transparency table is about, not a bug to catch
    here). Prints a one-line summary; returns True if the run looks stable.
    """
    errors = np.asarray(result["errors"], dtype=float)
    loss_nan = bool(np.isnan(errors).any() or np.isinf(errors).any())

    U_final = result["U_traj"][-1]
    u_nan = any(
        bool(np.isnan(np.asarray(v)).any()) or bool(np.isinf(np.asarray(v)).any())
        for v in U_final.values()
    )
    stable = not (loss_nan or u_nan)
    status = "ok" if stable else "UNSTABLE (NaN/Inf detected)"
    print(f"[{label}] loss0={errors[0]:.4g} lossF={errors[-1]:.4g} status={status}", flush=True)
    return stable


def run_expensive_sweep(script_path: Path, seeds: list[int], group_filter: str | None = None) -> list[dict]:
    """
    Independent full outer-loop re-optimizations across `seeds`, for every
    experiment group in the given companion script (or just `group_filter`
    if given). Each seed gets its own setup (so the baseline and the
    optimized-emissions emulator share init within that seed, per Stage 0)
    and its own checkpoint path (tag suffixed with the seed, so seeds never
    collide/resume into each other). Logs a CSV of per-run stability/timing
    to data/SI_results/seed_uncertainty/.
    """
    mod = load_experiments_module(script_path)
    checkpoint_subdir = f"{mod.CHECKPOINT_DIR}/seed_sweep"
    groups = [group_filter] if group_filter else list(mod.EXPERIMENTS)

    log_path = OUT_DIR / f"expensive_sweep_{script_path.stem}.csv"
    log_rows = []

    for seed in seeds:
        print(f"\n=== seed {seed}: building setup ({script_path.name}) ===", flush=True)
        setup = utils_inverse.run_inverse_experiment_setup(
            mod.AGENTS, mod.ACTIVE_AGENTS, mode=mod.MODE,
            CS3=True, DAMIP=False, GeoMIP=False, seed=seed, idx_demo=None,
        )
        for name in groups:
            cfg = dict(mod.EXPERIMENTS[name])
            cfg["tag"] = f"{cfg['tag']}_seed{seed}"
            t0 = time.perf_counter()
            result = utils_inverse.run_inverse_experiment(
                setup, checkpoint_dir=checkpoint_subdir, **cfg,
            )
            dt = time.perf_counter() - t0
            stable = check_instability(result, f"seed={seed} group={name}")
            log_rows.append({
                "seed": seed, "group": name, "seconds": round(dt, 2),
                "loss0": float(np.asarray(result["errors"])[0]),
                "lossF": float(np.asarray(result["errors"])[-1]),
                "stable": stable,
            })
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(log_rows[0]))
                writer.writeheader()
                writer.writerows(log_rows)

    n_unstable = sum(1 for r in log_rows if not r["stable"])
    print(f"\n=== Expensive sweep done: {len(log_rows)} runs, {n_unstable} flagged unstable. Log: {log_path} ===")
    return log_rows


def run_cheap_sweep(script_path: Path, seeds: list[int], group_filter: str | None = None) -> list[dict]:
    """
    Cheap axis: re-train just the final emulator (fresh MLP init per seed) on
    each *existing, already-optimized* checkpoint's final U_traj[-1] - no
    re-optimization. Note: evaluate_optimal_emulator's own `key` argument is
    unused internally (a pre-existing quirk - it trains from whatever
    `params0` it's given, not from `key`), so the seed is varied here by
    building a fresh `params0` via init_mlp_params per seed instead.
    """
    mod = load_experiments_module(script_path)
    groups = [group_filter] if group_filter else list(mod.EXPERIMENTS)

    setup0 = utils_inverse.run_inverse_experiment_setup(
        mod.AGENTS, mod.ACTIVE_AGENTS, mode=mod.MODE,
        CS3=True, DAMIP=False, GeoMIP=False, seed=0, idx_demo=None,
    )
    input_dim = int(setup0["params0"][0]["W"].shape[0])
    hidden_sizes = [int(setup0["params0"][0]["W"].shape[1])]

    log_path = OUT_DIR / f"cheap_sweep_{script_path.stem}.csv"
    log_rows = []

    for name in groups:
        cfg = mod.EXPERIMENTS[name]
        checkpoint_path = f"{mod.CHECKPOINT_DIR}/inverse_{cfg['init_cond']}_{cfg['group']}_{cfg['tag']}.pkl"
        if not Path(checkpoint_path).exists():
            print(f"[skip] {checkpoint_path} not found on disk - run the expensive axis "
                  f"(or the original 3a_inverse_CO2_only.py) first.", flush=True)
            continue

        for seed in seeds:
            t0 = time.perf_counter()
            params0_seed = utils_inverse.init_mlp_params(
                jax.random.PRNGKey(seed), input_dim=input_dim, hidden_sizes=hidden_sizes,
            )
            results_out = utils_inverse.evaluate_optimal_emulator(
                training_paths=[checkpoint_path],
                train_scenarios=[f"{name}_seed{seed}"],
                eval_sets=setup0["eval_sets"],
                params0=params0_seed,
                active_agents=mod.ACTIVE_AGENTS,
                mode=mod.MODE,
            )
            dt = time.perf_counter() - t0
            mean_err = results_out[f"{name}_seed{seed}"]["All"]["mean"]
            print(f"[seed={seed} group={name}] All-set mean NRMSE={mean_err:.4f} ({dt:.1f}s)", flush=True)
            log_rows.append({"seed": seed, "group": name, "seconds": round(dt, 2), "all_mean_nrmse": mean_err})
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(log_rows[0]))
                writer.writeheader()
                writer.writerows(log_rows)

    print(f"\n=== Cheap sweep done: {len(log_rows)} runs. Log: {log_path} ===")
    return log_rows


FIG4_CO2_GROUPS = ["tier1", "tier2", "DECK", "CS3", "all"]
FIG4_CO2_TRAIN_LABELS = ["Opt. Tier 1", "Opt. Tier 2", "Opt. DECK", "Opt. CS3", "Opt. All"]


def compute_fig4_eval_spread(script_path: Path, seeds: list[int]) -> dict:
    """
    For the Figure-4-style "performance summary" panel: evaluate each seed's
    already-completed expensive-axis checkpoints (tier1/tier2/DECK/CS3/all -
    H-ext is Figure 2's single-scenario example, not part of Figure 4) against
    eval_sets, using *that seed's own* params0 (same init the outer-loop
    optimization for that seed actually used, per the Stage 0 fairness
    property) - not a fresh/different seed for the evaluation step. This
    mirrors exactly how data/plotting/optimal_co2_only.pkl was originally
    built (same evaluate_optimal_emulator call, same train_scenarios/
    checkpoint naming), just repeated per seed.

    Returns {seed: results_out} where results_out matches
    evaluate_optimal_emulator's normal return schema
    (results_out[train_label][eval_set]['mean']). Cached to
    data/SI_results/seed_uncertainty/fig4_eval_spread_<script_stem>.pkl.
    """
    mod = load_experiments_module(script_path)
    checkpoint_subdir = f"{mod.CHECKPOINT_DIR}/seed_sweep"

    all_results = {}
    for seed in seeds:
        print(f"\n=== Fig4 eval, seed {seed} ===", flush=True)
        setup = utils_inverse.run_inverse_experiment_setup(
            mod.AGENTS, mod.ACTIVE_AGENTS, mode=mod.MODE,
            CS3=True, DAMIP=False, GeoMIP=False, seed=seed, idx_demo=None,
        )
        training_paths = []
        for name in FIG4_CO2_GROUPS:
            cfg = dict(mod.EXPERIMENTS[name])
            path = f"{checkpoint_subdir}/inverse_{cfg['init_cond']}_{cfg['group']}_{cfg['tag']}_seed{seed}.pkl"
            if not Path(path).exists():
                raise FileNotFoundError(f"{path} missing - run the expensive-axis sweep for seed {seed} first")
            training_paths.append(path)

        t0 = time.perf_counter()
        results_out = utils_inverse.evaluate_optimal_emulator(
            training_paths=training_paths,
            train_scenarios=FIG4_CO2_TRAIN_LABELS,
            eval_sets=setup["eval_sets"],
            params0=setup["params0"],
            active_agents=mod.ACTIVE_AGENTS,
            mode=mod.MODE,
        )
        dt = time.perf_counter() - t0
        print(f"seed {seed}: evaluated {len(training_paths)} checkpoints in {dt:.1f}s", flush=True)
        all_results[seed] = results_out

    out_path = OUT_DIR / f"fig4_eval_spread_{script_path.stem}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\n=== Fig4 eval spread done: {len(seeds)} seeds. Saved: {out_path} ===")
    return all_results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["pilot", "sweep", "fig4-eval"], default="pilot")
    parser.add_argument("--axis", choices=["expensive", "cheap"], default="expensive")
    parser.add_argument("--script", default="3a_inverse_CO2_only.py",
                         help="Which existing 3x/SIx companion script's EXPERIMENTS dict to sweep")
    parser.add_argument("--group", default=None, help="Only this experiment group (default: all in the script)")
    parser.add_argument("--n-cheap-seeds", type=int, default=20)
    parser.add_argument("--n-expensive-seeds", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "pilot":
        run_pilot()
        return

    script_path = PROJECT_ROOT / "scripts" / args.script
    if args.mode == "fig4-eval":
        compute_fig4_eval_spread(script_path, seeds=list(range(args.n_expensive_seeds)))
        return

    if args.axis == "expensive":
        run_expensive_sweep(script_path, seeds=list(range(args.n_expensive_seeds)), group_filter=args.group)
    else:
        run_cheap_sweep(script_path, seeds=list(range(args.n_cheap_seeds)), group_filter=args.group)


if __name__ == "__main__":
    main()
