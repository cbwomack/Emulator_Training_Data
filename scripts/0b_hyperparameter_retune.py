#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Phase 0, Stage 0b: hyperparameter re-tuning for the CO2-only bilevel
outer-loop optimizer.

Design (revised 2026-08-04, per user direction): ONE shared hyperparameter
configuration - step_size, momentum, nesterov, K_inner, lr_inner, wd_inner,
smoothness_weight, batch_size - used identically across all 6 groups defined
in scripts/3a_inverse_CO2_only.py's EXPERIMENTS dict (H-ext, tier1, tier2,
DECK, CS3, all). This matches the manuscript's own framing (SI sec:
sensitivity: hyperparameters are tuned once "to the architecture", not
per scenario) and the fact that every hyperparameter except step_size was
already identical across groups in the original methodology.

(Earlier version of this script searched each group independently and
picked 6 separate per-group winners; that produced a fragmented, hard-to-
describe set of configs and doesn't match how the manuscript presents its
methodology - superseded by this design. It also had a real bug, fixed
here: get_setup_and_groups() used to hardcode seed=0 regardless of the
requested seed, making params0 - and, for batch_size=None, everything -
identical across "different seeds". Confirmed directly: bit-identical
scores across seeds for the old 'all' config. Fixed by keying the setup
cache on the actual requested seed.)

Method: two-phase "cheap search, then expensive validation", scaled up
2026-08-04 now that everything is cluster-parallelized (100 cheap-search
configs x 6 groups = 600 runs; top-5 candidates x 6 groups x 10 seeds = 300
validate runs; 900 runs total):
  1. `cheap-search`: sample N=100 shared candidate configs; evaluate each
     one on every group (reduced num_updates=300, single seed=0).
     Parallelizable per (config_idx, group) via `cheap-search-one`.
  2. `select`: combine each candidate's 6 per-group scores into one
     weighted mean (weights match Fig 4's existing scenario-count
     convention - tier1:7, tier2:5, DECK:2, CS3:2 - extended with H-ext:1
     since it's a single evaluation target, and all:17 = sum of the other
     5, since 'all' evaluates against their combined/union scenario set);
     keep the top-K.
  3. `validate`: re-run the top-K candidates at full num_updates=1000,
     across 10 seeds, on every group, to confirm the winner holds up at
     full scale; pick the single overall winner by weighted mean score
     across groups and seeds. Parallelizable per (cand_idx, group, seed)
     via `validate-one`.
A candidate only counts as "stable" (usable by select/finalize) if its
error trajectory is *also* well-behaved throughout - not just finite at
the end. This directly targets the spike-instability failure mode found in
the pre-Stage-0 checkpoints (runaway trajectories that spike 5-30x and
never recover): disqualified if it has more than MAX_SPIKES=5 single-step
increases exceeding SPIKE_RATIO_THRESHOLD=1.5x, or if its final error is
more than MAX_REGRET=15% worse than the best point it ever reached.
The final winning single config is written to
data/SI_results/hp_retune/best_config_unified.json, ready to feed Stage
0c's checkpoint regeneration (used identically for every group).

Every phase is resumable: each run's result is written to its own file
immediately (never a shared per-process-unsafe append), and anything
already present is skipped on restart. Everything is meant to run on the
cluster via SLURM arrays (see REVISIONS.md for the job scripts), not
locally - both phases have per-combo CLI entry points for exactly that.

Usage:
    python 0b_hyperparameter_retune.py --mode cheap-search
    python 0b_hyperparameter_retune.py --mode select
    python 0b_hyperparameter_retune.py --mode validate-one --cand-idx 0 --group tier1 --validate-seeds 0
    python 0b_hyperparameter_retune.py --mode validate    # single-machine convenience wrapper
    python 0b_hyperparameter_retune.py --mode collect
    python 0b_hyperparameter_retune.py --mode finalize
    python 0b_hyperparameter_retune.py --mode report
"""
import os
import sys
import csv
import json
import time
import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import jax

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import utils_inverse

OUT_DIR = Path("data/SI_results/hp_retune")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGENTS = ["CO2"]
ACTIVE_AGENTS = ("CO2",)
MODE = "FaIR"

SEARCH_NUM_UPDATES = 300
VALIDATE_NUM_UPDATES = 1000
TOP_K = 5

# Monotonicity/stability criteria (per user direction, 2026-08-04): a config
# is only "stable" (usable by select/finalize) if its error trajectory is
# also well-behaved throughout, not just finite at the end - directly
# targets the spike-instability failure mode found in the pre-Stage-0
# checkpoints (runaway trajectories that spike 5-30x and never recover).
SPIKE_RATIO_THRESHOLD = 1.5  # single-step error[t+1]/error[t] above this counts as a spike
MAX_SPIKES = 5                # tolerate a handful of minor spikes (numerical/minibatch noise)
MAX_REGRET = 0.15             # final error must be within 15% of the trajectory's own best point

TUNABLE_KEYS = ["step_size", "momentum", "nesterov", "K_inner", "lr_inner",
                "wd_inner", "smoothness_weight", "batch_size"]

# Fig 4's existing scenario-count weighting (see 6a_seed_uncertainty_sweep.ipynb's
# WEIGHTS = [7,5,2,2] for tier1/tier2/DECK/CS3), extended with H-ext and 'all' at
# weight 1 each since they're each a single evaluation target, not part of that
# established per-scenario-count convention.
GROUP_WEIGHTS = {"H-ext": 1, "tier1": 7, "tier2": 5, "DECK": 2, "CS3": 2, "all": 17}
# 'all' = 17 = sum of the other 5 (1+7+5+2+2), since it evaluates against
# the combined/union scenario set the other 5 groups each cover individually.


def load_experiments_module():
    spec = importlib.util.spec_from_file_location(
        "_exp_mod_3a", PROJECT_ROOT / "scripts" / "3a_inverse_CO2_only.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bs_label(bs):
    return "full" if bs is None else str(bs)


def sample_config(rng):
    # Absolute log-uniform range (not centered on any one group's old default,
    # since step_size is now shared across all 6 groups) - chosen to comfortably
    # span the per-group winners the old per-group search found (848-5886).
    return {
        "step_size": float(np.exp(rng.uniform(np.log(300), np.log(6000)))),
        "momentum": float(rng.choice([0.7, 0.85, 0.9, 0.95, 0.99])),
        "nesterov": bool(rng.choice([True, False])),
        "K_inner": 400,  # locked to the manuscript's methodology, not searched
        "lr_inner": float(np.exp(rng.uniform(np.log(0.01), np.log(0.15)))),
        "wd_inner": float(rng.choice([0.0, 1e-3, 3e-3, 1e-2, 3e-2])),
        "smoothness_weight": float(rng.choice([0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4])),
        "batch_size": [None, 16, 32, 64][int(rng.integers(0, 4))],
    }


_SETUP_CACHE = {}


def get_setup_and_groups(seed=0):
    """
    Builds (and caches, per-seed) the experiment setup - critically,
    `params0` (the MLP's initial weights) is seeded from this call's `seed`,
    NOT hardcoded. Bug history: this used to hardcode seed=0 unconditionally
    (ignoring the caller's requested seed) and cache a single global setup -
    meaning every "different seed" validate run actually reused identical
    params0, and for batch_size=None (no minibatch key usage either) that
    made the run 100% seed-invariant - confirmed directly (bit-identical
    scores across seeds for the 'all' group). Fixed 2026-08-04.
    """
    if seed not in _SETUP_CACHE:
        setup = utils_inverse.run_inverse_experiment_setup(
            AGENTS, ACTIVE_AGENTS, mode=MODE, CS3=True, DAMIP=False, GeoMIP=False,
            seed=seed, idx_demo=None,
        )
        group_emis_dicts = utils_inverse.build_group_emis_dicts(
            setup["emis_dict_train_JAX"], setup["eval_sets"],
        )
        _SETUP_CACHE[seed] = (setup, group_emis_dicts)
    return _SETUP_CACHE[seed]


def run_one(base_exp, group_emis_dict, params0, cfg, num_updates, seed,
            checkpoint_path=None):
    batch_size = cfg.get("batch_size", None)
    t0 = time.perf_counter()
    try:
        result = utils_inverse.optimize_emissions_inverse(
            emis_dict=group_emis_dict,
            params0=params0,
            num_updates=num_updates,
            step_size=cfg["step_size"], momentum=cfg["momentum"], nesterov=cfg["nesterov"],
            K_inner=cfg["K_inner"], lr_inner=cfg["lr_inner"], wd_inner=cfg["wd_inner"],
            smoothness_weight=cfg["smoothness_weight"],
            active_agents=ACTIVE_AGENTS,
            init_cond=base_exp["init_cond"],
            T=base_exp["T"],
            filter_hist=base_exp["filter_hist"],
            mode=MODE,
            checkpoint_path=checkpoint_path,
            checkpoint_every=max(num_updates, 1),
            resume_if_exists=checkpoint_path is not None,
            preds_every=max(num_updates, 1),
            batch_size=batch_size,
            key=jax.random.PRNGKey(1000 + seed),
        )
        errors = np.asarray(result["errors"])
        finite = bool(np.isfinite(errors).all())
        if finite:
            ratio = errors[1:] / np.maximum(errors[:-1], 1e-8)
            n_spikes = int((ratio > SPIKE_RATIO_THRESHOLD).sum())
            regret = float((errors[-1] - errors.min()) / max(float(errors.min()), 1e-8))
            monotonic_ok = (n_spikes <= MAX_SPIKES) and (regret <= MAX_REGRET)
        else:
            n_spikes, regret, monotonic_ok = -1, float("inf"), False
        stable = finite and monotonic_ok
        tail = errors[-min(20, len(errors)):]
        score = float(np.mean(tail)) if finite else float("inf")
        lossF = float(errors[-1]) if finite else float("nan")
    except Exception as e:
        print(f"  !! run failed: {type(e).__name__}: {e}", flush=True)
        stable = False
        score = float("inf")
        lossF = float("nan")
        n_spikes, regret = -1, float("inf")
    dt = time.perf_counter() - t0
    return {"score": score, "lossF": lossF, "stable": stable, "n_spikes": n_spikes,
            "regret": round(regret, 4) if np.isfinite(regret) else regret, "seconds": round(dt, 2)}


def _weighted_mean(group_scores: dict):
    """group_scores: {group: score}. Returns None if any of the 6 groups is missing."""
    if set(group_scores) != set(GROUP_WEIGHTS):
        return None
    num = sum(GROUP_WEIGHTS[g] * group_scores[g] for g in GROUP_WEIGHTS)
    den = sum(GROUP_WEIGHTS.values())
    return num / den


# ----------------------------------------------------------------------
# Phase 1: cheap-search - each candidate config evaluated on every group
# ----------------------------------------------------------------------

def _cheap_result_path(config_idx, group):
    return OUT_DIR / f"cheap_unified_cfg{config_idx}_{group}_result.json"


def _get_config_by_index(config_idx, search_seed):
    """Deterministically regenerates the config at `config_idx` from a fresh
    RNG seeded by `search_seed` - lets independent processes (e.g. separate
    SLURM array tasks) agree on "config idx X" without needing a shared
    pre-generated file. Cheap: sampling itself costs nothing, only the
    training run that follows is expensive."""
    rng = np.random.default_rng(search_seed)
    cfg = None
    for _ in range(config_idx + 1):
        cfg = sample_config(rng)
    return cfg


def cheap_search_one(config_idx, group, search_seed):
    """Run exactly one (config_idx, group) cheap-search combo - the unit of
    work a SLURM array task calls. Idempotent: skips if its result file
    already exists."""
    rp = _cheap_result_path(config_idx, group)
    if rp.is_file():
        print(f"[cheap-search cfg={config_idx} group={group}] already done, skipping")
        return

    exp_mod = load_experiments_module()
    if group not in exp_mod.EXPERIMENTS:
        raise SystemExit(f"unknown group {group!r}; available: {sorted(exp_mod.EXPERIMENTS)}")
    setup, group_emis_dicts = get_setup_and_groups(seed=0)
    cfg = _get_config_by_index(config_idx, search_seed)
    base_exp = exp_mod.EXPERIMENTS[group]

    res = run_one(base_exp, group_emis_dicts[group], setup["params0"], cfg,
                  SEARCH_NUM_UPDATES, seed=0, checkpoint_path=None)
    row = {"config_idx": config_idx, "group": group, **cfg,
           "batch_size": _bs_label(cfg["batch_size"]), **res}
    print(f"[cheap-search cfg={config_idx} group={group}] score={res['score']:.4g} "
          f"stable={res['stable']} ({res['seconds']}s)", flush=True)
    tmp = rp.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, rp)


def cheap_search(exp_mod, n_configs, search_seed):
    """Single-machine convenience wrapper: run every (config_idx, group)
    combo sequentially. Prefer cheap_search_one + a SLURM array for anything
    parallelizable."""
    groups = list(exp_mod.EXPERIMENTS)
    for idx in range(n_configs):
        for group in groups:
            cheap_search_one(idx, group, search_seed)


def _load_cheap_results():
    rows = [json.load(open(p)) for p in sorted(OUT_DIR.glob("cheap_unified_cfg*_result.json"))]
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r["config_idx"], {})[r["group"]] = r
    return by_cfg


def select_candidates(top_k):
    by_cfg = _load_cheap_results()
    if not by_cfg:
        print("no cheap-search results yet")
        return
    scored = []
    for config_idx, by_group in by_cfg.items():
        if not all(by_group.get(g, {}).get("stable") for g in GROUP_WEIGHTS):
            continue
        group_scores = {g: by_group[g]["score"] for g in GROUP_WEIGHTS}
        wmean = _weighted_mean(group_scores)
        scored.append((wmean, config_idx, by_group))
    scored.sort(key=lambda t: t[0])
    top = scored[:top_k]

    candidates = []
    for wmean, config_idx, by_group in top:
        any_row = next(iter(by_group.values()))
        cfg = {
            "step_size": any_row["step_size"], "momentum": any_row["momentum"],
            "nesterov": any_row["nesterov"], "K_inner": any_row["K_inner"],
            "lr_inner": any_row["lr_inner"], "wd_inner": any_row["wd_inner"],
            "smoothness_weight": any_row["smoothness_weight"],
            "batch_size": None if any_row["batch_size"] == "full" else int(any_row["batch_size"]),
        }
        candidates.append({"config_idx": config_idx, "cheap_weighted_score": wmean, "config": cfg})

    out_path = OUT_DIR / "top_candidates_unified.json"
    with open(out_path, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"wrote {len(candidates)} top candidates -> {out_path}")
    for c in candidates:
        print(f"  cfg{c['config_idx']}: weighted_score={c['cheap_weighted_score']:.4g}")


def _load_candidates():
    p = OUT_DIR / "top_candidates_unified.json"
    if not p.is_file():
        return None
    return json.load(open(p))


# ----------------------------------------------------------------------
# Phase 2: validate - top-K candidates, all 6 groups, several seeds
# ----------------------------------------------------------------------

def _validate_result_path(cand_idx, group, seed, num_updates):
    return OUT_DIR / f"validate_unified_cand{cand_idx}_{group}_seed{seed}_n{num_updates}_result.json"


def validate_one(cand_idx, group, seed, num_updates):
    candidates = _load_candidates()
    if candidates is None:
        print("no top-candidates file yet, run --mode select first")
        return
    if cand_idx >= len(candidates):
        print(f"cand_idx={cand_idx} out of range ({len(candidates)} candidates)")
        return

    result_path = _validate_result_path(cand_idx, group, seed, num_updates)
    if result_path.is_file():
        print(f"[validate cand={cand_idx} group={group} seed={seed}] already done, skipping")
        return

    setup, group_emis_dicts = get_setup_and_groups(seed=seed)
    exp_mod = load_experiments_module()
    base_exp = exp_mod.EXPERIMENTS[group]
    cfg = candidates[cand_idx]["config"]

    ckpt = OUT_DIR / f"validate_unified_cand{cand_idx}_{group}_seed{seed}.pkl"
    res = run_one(base_exp, group_emis_dicts[group], setup["params0"], cfg,
                  num_updates, seed=seed, checkpoint_path=str(ckpt))
    row = {"cand_idx": cand_idx, "group": group, "seed": seed, "num_updates": num_updates,
           **cfg, "batch_size": _bs_label(cfg["batch_size"]), **res}
    print(f"[validate cand={cand_idx} group={group} seed={seed}] score={res['score']:.4g} "
          f"stable={res['stable']} ({res['seconds']}s)", flush=True)

    tmp_path = result_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(row, f)
    os.replace(tmp_path, result_path)


def validate(seeds, num_updates):
    """Single-machine convenience wrapper: run every (cand_idx, group, seed)
    combo sequentially, then collect+finalize. Prefer validate_one + a SLURM
    array for anything parallelizable."""
    candidates = _load_candidates()
    if candidates is None:
        print("no top-candidates file yet, run --mode select first")
        return
    for cand_idx in range(len(candidates)):
        for group in GROUP_WEIGHTS:
            for seed in seeds:
                validate_one(cand_idx, group, seed, num_updates)
    collect(num_updates)
    finalize(seeds, num_updates)


def collect(num_updates):
    pattern = f"validate_unified_cand*_*_seed*_n{num_updates}_result.json"
    result_files = sorted(OUT_DIR.glob(pattern))
    if not result_files:
        print(f"no per-combo result files found for num_updates={num_updates}")
        return
    rows = [json.load(open(p)) for p in result_files]

    csv_path = OUT_DIR / "validate_unified.csv"
    existing = []
    if csv_path.is_file():
        existing = [r for r in csv.DictReader(open(csv_path)) if int(r["num_updates"]) != num_updates]

    fieldnames = ["cand_idx", "group", "seed", "num_updates"] + TUNABLE_KEYS + \
                 ["score", "lossF", "stable", "n_spikes", "regret", "seconds"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing:
            writer.writerow(r)
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"collected {len(rows)} result files -> {csv_path} "
          f"({len(existing)} pre-existing rows for other num_updates preserved)")


def finalize(seeds, num_updates):
    """Pick the single overall winner: for each candidate, average its score
    across seeds within each group, then combine across groups via the
    weighted mean - require every group present and stable on every
    requested seed."""
    csv_path = OUT_DIR / "validate_unified.csv"
    if not csv_path.is_file():
        print("no validate results yet")
        return
    candidates = _load_candidates()
    rows = [r for r in csv.DictReader(open(csv_path)) if int(r["num_updates"]) == num_updates]

    by_cand_group = {}
    for r in rows:
        key = (r["cand_idx"], r["group"])
        by_cand_group.setdefault(key, []).append(r)

    cand_ids = sorted({r["cand_idx"] for r in rows}, key=int)
    best_idx, best_score, best_breakdown = None, float("inf"), None
    for cand_idx in cand_ids:
        group_means = {}
        ok = True
        for group in GROUP_WEIGHTS:
            rs = by_cand_group.get((cand_idx, group), [])
            if len(rs) < len(seeds) or not all(r["stable"] == "True" for r in rs):
                ok = False
                break
            group_means[group] = float(np.mean([float(r["score"]) for r in rs]))
        if not ok:
            continue
        wmean = _weighted_mean(group_means)
        if wmean < best_score:
            best_score = wmean
            best_idx = cand_idx
            best_breakdown = group_means

    if best_idx is None:
        print("no candidate is fully validated (all 6 groups x all requested seeds, all stable) yet")
        return

    winner = candidates[int(best_idx)]["config"]
    out_path = OUT_DIR / "best_config_unified.json"
    with open(out_path, "w") as f:
        json.dump({"cand_idx": best_idx, "weighted_validate_score": best_score,
                   "per_group_mean_score": best_breakdown, "config": winner}, f, indent=2)
    print(f"winner = candidate {best_idx} (weighted score {best_score:.4g}) -> {out_path}")
    for g, s in best_breakdown.items():
        print(f"  {g:6s}: {s:.4g}")


def report():
    for kind, path in [("cheap-search results", None),
                        ("top-candidates", OUT_DIR / "top_candidates_unified.json"),
                        ("validate results", OUT_DIR / "validate_unified.csv"),
                        ("best-config", OUT_DIR / "best_config_unified.json")]:
        if path is None:
            n = len(list(OUT_DIR.glob("cheap_unified_cfg*_result.json")))
            print(f"cheap-search results: {n} files")
            continue
        status = "done" if path.is_file() else "missing"
        print(f"{kind}: {status} ({path})")
    best_path = OUT_DIR / "best_config_unified.json"
    if best_path.is_file():
        print(json.dumps(json.load(open(best_path)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["cheap-search", "cheap-search-one", "select", "validate",
                                            "validate-one", "collect", "finalize", "report"], required=True)
    parser.add_argument("--group", default=None, help="validate-one/cheap-search-one: one of H-ext/tier1/tier2/DECK/CS3/all")
    parser.add_argument("--n-configs", type=int, default=100, help="cheap-search only")
    parser.add_argument("--search-seed", type=int, default=0, help="cheap-search/cheap-search-one")
    parser.add_argument("--config-idx", type=int, default=None, help="cheap-search-one only")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="select only")
    parser.add_argument("--validate-seeds", type=str, default="0,1,2,3,4,5,6,7,8,9",
                         help="validate/finalize: comma list. validate-one: single seed value")
    parser.add_argument("--validate-num-updates", type=int, default=VALIDATE_NUM_UPDATES,
                         help="validate/validate-one/finalize only")
    parser.add_argument("--cand-idx", type=int, default=None, help="validate-one only")
    args = parser.parse_args()

    seeds = [int(s) for s in args.validate_seeds.split(",")]

    if args.mode == "cheap-search":
        cheap_search(load_experiments_module(), args.n_configs, args.search_seed)
    elif args.mode == "cheap-search-one":
        if args.config_idx is None or args.group is None:
            raise SystemExit("cheap-search-one requires --config-idx and --group")
        cheap_search_one(args.config_idx, args.group, args.search_seed)
    elif args.mode == "select":
        select_candidates(args.top_k)
    elif args.mode == "validate":
        validate(seeds, args.validate_num_updates)
    elif args.mode == "validate-one":
        if args.cand_idx is None or args.group is None or len(seeds) != 1:
            raise SystemExit("validate-one requires --cand-idx, --group, and a single --validate-seeds value")
        validate_one(args.cand_idx, args.group, seeds[0], args.validate_num_updates)
    elif args.mode == "collect":
        collect(args.validate_num_updates)
    elif args.mode == "finalize":
        finalize(seeds, args.validate_num_updates)
    elif args.mode == "report":
        report()


if __name__ == "__main__":
    main()
