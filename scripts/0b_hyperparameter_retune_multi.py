#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Phase 0, Stage 0b (multi-agent): hyperparameter re-tuning for the all-5-
agents-active (CO2/CH4/N2O/Sulfur/BC) bilevel outer-loop optimizer.

checkpoints/multi/ - used by Figs 4 (multi-agent panel)/5/6/7 - was never
regenerated after the same undocumented pipeline change that made every
single-forcing checkpoint stale (see REVISIONS.md, Jan26-Mar9 2026), exactly
like CO2/CH4/N2O/Sulfur/BC before their own Phase 0 passes. This is that
same treatment applied to the multi-agent case. Per user direction
(2026-08-13): "the learning rate for each agent needs to be optimized" -
unlike every single-forcing search (ONE shared step_size), the multi-agent
optimizer's step_size is a DICT, one independent value per active agent
(matching 3b_inverse_all_agents.py's own EXPERIMENTS structure, e.g.
{'CO2': 100.0, 'CH4': 10000.0, 'N2O': 100.0, 'Sulfur': 100.0, 'BC': 10.0}),
so this is a genuinely higher-dimensional search than any single-forcing
agent's, not just "agent=multi" bolted onto the existing generalized script.

Per-agent step_size log-uniform ranges reuse each agent's OWN already-tuned
single-forcing Phase 0 range as a sensible, evidence-based prior (not a
guess) - CO2: (300, 6000) from 0b_hyperparameter_retune.py; CH4: (3000,
150000), N2O: (15, 3000), Sulfur: (1500, 15000), BC: (30, 400), all four
from 0b_hyperparameter_retune_agent.py's AGENT_CONFIG.

Groups: 3b_inverse_all_agents.py's own EXPERIMENTS dict only ever defined
H-ext/DAMIP/GeoMIP/all (never tier1/tier2/DECK/CS3 individually) - but
Figure 5 loads a 'tier1'-group checkpoint that isn't even in that dict
(inverse_constant_tier1_all_agents_subset2.pkl, a separately-run,
undocumented variant - see 6c_ood_scenario_ssp370_lowntcf.py's docstring on
this same quirk). This search covers H-ext/tier1/DAMIP/GeoMIP/all - the
documented groups plus 'tier1' for Figure 5 - rather than silently carrying
forward the undocumented "subset2" one-off. GROUP_WEIGHTS are equal (1 each)
for this search's own candidate-selection metric only; this is a distinct,
simpler weighting than the single-forcing GROUP_WEIGHTS (which mirror Fig
4/SI Fig 6's real scenario-count weighting) since DAMIP/GeoMIP don't have a
natural scenario-count analog and this weighting only decides which
hyperparameter config wins, not any headline result number.

Usage:
    python 0b_hyperparameter_retune_multi.py --mode cheap-search
    python 0b_hyperparameter_retune_multi.py --mode select
    python 0b_hyperparameter_retune_multi.py --mode validate-one --cand-idx 0 --group tier1 --validate-seeds 0
    python 0b_hyperparameter_retune_multi.py --mode validate    # single-machine convenience wrapper
    python 0b_hyperparameter_retune_multi.py --mode collect
    python 0b_hyperparameter_retune_multi.py --mode finalize
    python 0b_hyperparameter_retune_multi.py --mode report
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

MODE = "FaIR"
AGENTS = ["CO2", "CH4", "N2O", "Sulfur", "BC"]
ACTIVE_AGENTS = ("CO2", "CH4", "N2O", "Sulfur", "BC")

SEARCH_NUM_UPDATES = 300
VALIDATE_NUM_UPDATES = 1000
TOP_K = 5

SPIKE_RATIO_THRESHOLD = 1.5
MAX_SPIKES = 5
MAX_REGRET = 0.15

# Per user direction (2026-08-13): prioritize run stability (regret, spikes)
# in candidate SELECTION, not just as the existing binary MAX_SPIKES/
# MAX_REGRET pass/fail gate above. A candidate right at the regret gate
# (regret=MAX_REGRET) gets its ranking score inflated by
# STABILITY_PENALTY_WEIGHT (=100%) relative to a perfectly-monotonic one
# (regret=0); spikes are penalized the same way, scaled to MAX_SPIKES. Both
# candidates still have to clear the hard gate to be considered at all -
# this only breaks ties among already-stable candidates toward the more
# stable ones, rather than picking whichever happens to have the single
# lowest final loss.
STABILITY_PENALTY_WEIGHT = 1.0

TUNABLE_SCALAR_KEYS = ["momentum", "nesterov", "K_inner", "lr_inner",
                       "wd_inner", "smoothness_weight", "batch_size"]

STEP_SIZE_RANGES = {
    "CO2": (300.0, 6000.0),
    "CH4": (3000.0, 150000.0),
    "N2O": (15.0, 3000.0),
    "Sulfur": (1500.0, 15000.0),
    "BC": (30.0, 400.0),
}

GROUPS = ["H-ext", "tier1", "DAMIP", "GeoMIP", "all"]
GROUP_WEIGHTS = {g: 1 for g in GROUPS}

OUT_DIR = Path("data/SI_results/hp_retune/multi")


def load_3b_experiments_module():
    spec = importlib.util.spec_from_file_location(
        "_exp_mod_3b", PROJECT_ROOT / "scripts" / "3b_inverse_all_agents.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bs_label(bs):
    return "full" if bs is None else str(bs)


def sample_config(rng):
    step_size = {a: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                 for a, (lo, hi) in STEP_SIZE_RANGES.items()}
    return {
        "step_size": step_size,
        "momentum": float(rng.choice([0.7, 0.85, 0.9, 0.95, 0.99])),
        "nesterov": bool(rng.choice([True, False])),
        "K_inner": 400,
        "lr_inner": float(np.exp(rng.uniform(np.log(0.01), np.log(0.15)))),
        "wd_inner": float(rng.choice([0.0, 1e-3, 3e-3, 1e-2, 3e-2])),
        "smoothness_weight": float(rng.choice([0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4])),
        "batch_size": [None, 16, 32, 64][int(rng.integers(0, 4))],
    }


_SETUP_CACHE = {}


def get_setup_and_groups(seed=0):
    if seed not in _SETUP_CACHE:
        setup = utils_inverse.run_inverse_experiment_setup(
            AGENTS, ACTIVE_AGENTS, mode=MODE, CS3=True, DAMIP=True, GeoMIP=True,
            seed=seed, idx_demo=None,
        )
        group_emis_dicts = utils_inverse.build_group_emis_dicts(
            setup["emis_dict_train_JAX"], setup["eval_sets"],
        )
        _SETUP_CACHE[seed] = (setup, group_emis_dicts)
    return _SETUP_CACHE[seed]


def run_one(base_exp, group_emis_dict, params0, cfg, num_updates, seed, checkpoint_path=None):
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
    if set(group_scores) != set(GROUP_WEIGHTS):
        return None
    num = sum(GROUP_WEIGHTS[g] * group_scores[g] for g in GROUP_WEIGHTS)
    den = sum(GROUP_WEIGHTS.values())
    return num / den


def _ranking_score(group_scores: dict, group_regrets: dict, group_spikes: dict):
    """Weighted mean score, inflated by a stability penalty from weighted mean
    regret/spikes (see STABILITY_PENALTY_WEIGHT) - used to RANK already-gate-
    passing candidates, so two candidates with similar predictive score are
    broken toward the more stable (lower regret/fewer spikes) one."""
    wmean_score = _weighted_mean(group_scores)
    if wmean_score is None:
        return None
    wmean_regret = _weighted_mean(group_regrets)
    wmean_spikes = _weighted_mean(group_spikes)
    regret_frac = min(wmean_regret / MAX_REGRET, 1.0) if MAX_REGRET > 0 else 0.0
    spike_frac = min(wmean_spikes / MAX_SPIKES, 1.0) if MAX_SPIKES > 0 else 0.0
    penalty = 1.0 + STABILITY_PENALTY_WEIGHT * 0.5 * (regret_frac + spike_frac)
    return wmean_score * penalty


# ----------------------------------------------------------------------
# Phase 1: cheap-search
# ----------------------------------------------------------------------

def _cheap_result_path(config_idx, group):
    return OUT_DIR / f"cheap_unified_cfg{config_idx}_{group}_result.json"


def _get_config_by_index(config_idx, search_seed):
    rng = np.random.default_rng(search_seed)
    cfg = None
    for _ in range(config_idx + 1):
        cfg = sample_config(rng)
    return cfg


def cheap_search_one(config_idx, group, search_seed):
    rp = _cheap_result_path(config_idx, group)
    if rp.is_file():
        print(f"[multi cheap-search cfg={config_idx} group={group}] already done, skipping")
        return

    exp_mod = load_3b_experiments_module()
    if group == "tier1":
        base_exp = {"init_cond": "constant", "T": 751, "filter_hist": False}
    elif group in exp_mod.EXPERIMENTS:
        base_exp = exp_mod.EXPERIMENTS[group]
    else:
        raise SystemExit(f"unknown group {group!r}; available: {GROUPS}")

    setup, group_emis_dicts = get_setup_and_groups(seed=0)
    cfg = _get_config_by_index(config_idx, search_seed)

    res = run_one(base_exp, group_emis_dicts[group], setup["params0"], cfg,
                  SEARCH_NUM_UPDATES, seed=0, checkpoint_path=None)
    row = {"config_idx": config_idx, "group": group,
           "step_size": cfg["step_size"], **{k: cfg[k] for k in TUNABLE_SCALAR_KEYS},
           "batch_size": _bs_label(cfg["batch_size"]), **res}
    print(f"[multi cheap-search cfg={config_idx} group={group}] score={res['score']:.4g} "
          f"stable={res['stable']} ({res['seconds']}s)", flush=True)
    tmp = rp.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, rp)


def cheap_search(n_configs, search_seed):
    for idx in range(n_configs):
        for group in GROUPS:
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
        group_regrets = {g: by_group[g]["regret"] for g in GROUP_WEIGHTS}
        group_spikes = {g: by_group[g]["n_spikes"] for g in GROUP_WEIGHTS}
        rank = _ranking_score(group_scores, group_regrets, group_spikes)
        scored.append((rank, config_idx, by_group))
    scored.sort(key=lambda t: t[0])
    top = scored[:top_k]

    candidates = []
    for rank, config_idx, by_group in top:
        any_row = next(iter(by_group.values()))
        cfg = {"step_size": any_row["step_size"],
               **{k: any_row[k] for k in TUNABLE_SCALAR_KEYS if k != "batch_size"},
               "batch_size": None if any_row["batch_size"] == "full" else int(any_row["batch_size"])}
        candidates.append({"config_idx": config_idx, "cheap_ranking_score": rank, "config": cfg})

    out_path = OUT_DIR / "top_candidates_unified.json"
    with open(out_path, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"wrote {len(candidates)} top candidates -> {out_path}")
    for c in candidates:
        print(f"  cfg{c['config_idx']}: ranking_score={c['cheap_ranking_score']:.4g} "
              f"(score, penalized for regret/spikes)")


def _load_candidates():
    p = OUT_DIR / "top_candidates_unified.json"
    if not p.is_file():
        return None
    return json.load(open(p))


# ----------------------------------------------------------------------
# Phase 2: validate
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
        print(f"[multi validate cand={cand_idx} group={group} seed={seed}] already done, skipping")
        return

    setup, group_emis_dicts = get_setup_and_groups(seed=seed)
    exp_mod = load_3b_experiments_module()
    if group == "tier1":
        base_exp = {"init_cond": "constant", "T": 751, "filter_hist": False}
    else:
        base_exp = exp_mod.EXPERIMENTS[group]
    cfg = candidates[cand_idx]["config"]

    ckpt = OUT_DIR / f"validate_unified_cand{cand_idx}_{group}_seed{seed}.pkl"
    res = run_one(base_exp, group_emis_dicts[group], setup["params0"], cfg,
                  num_updates, seed=seed, checkpoint_path=str(ckpt))
    row = {"cand_idx": cand_idx, "group": group, "seed": seed, "num_updates": num_updates,
           "step_size": json.dumps(cfg["step_size"]),
           **{k: cfg[k] for k in TUNABLE_SCALAR_KEYS if k != "batch_size"},
           "batch_size": _bs_label(cfg["batch_size"]), **res}
    print(f"[multi validate cand={cand_idx} group={group} seed={seed}] score={res['score']:.4g} "
          f"stable={res['stable']} ({res['seconds']}s)", flush=True)

    tmp_path = result_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(row, f)
    os.replace(tmp_path, result_path)


def validate(seeds, num_updates):
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

    fieldnames = ["cand_idx", "group", "seed", "num_updates", "step_size"] + \
                 [k for k in TUNABLE_SCALAR_KEYS if k != "batch_size"] + \
                 ["batch_size", "score", "lossF", "stable", "n_spikes", "regret", "seconds"]
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
    best_idx, best_rank, best_breakdown, best_score = None, float("inf"), None, None
    for cand_idx in cand_ids:
        group_means, group_regret_means, group_spike_means = {}, {}, {}
        ok = True
        for group in GROUP_WEIGHTS:
            rs = by_cand_group.get((cand_idx, group), [])
            if len(rs) < len(seeds) or not all(r["stable"] == "True" for r in rs):
                ok = False
                break
            group_means[group] = float(np.mean([float(r["score"]) for r in rs]))
            group_regret_means[group] = float(np.mean([float(r["regret"]) for r in rs]))
            group_spike_means[group] = float(np.mean([float(r["n_spikes"]) for r in rs]))
        if not ok:
            continue
        rank = _ranking_score(group_means, group_regret_means, group_spike_means)
        if rank < best_rank:
            best_rank = rank
            best_idx = cand_idx
            best_breakdown = group_means
            best_score = _weighted_mean(group_means)

    if best_idx is None:
        print(f"no candidate is fully validated (all {len(GROUPS)} groups x all requested seeds, all stable) yet")
        return

    winner = candidates[int(best_idx)]["config"]
    out_path = OUT_DIR / "best_config_unified.json"
    with open(out_path, "w") as f:
        json.dump({"cand_idx": best_idx, "weighted_ranking_score": best_rank,
                   "weighted_validate_score": best_score,
                   "per_group_mean_score": best_breakdown, "config": winner}, f, indent=2)
    print(f"winner = candidate {best_idx} (ranking score {best_rank:.4g}, raw score {best_score:.4g}) -> {out_path}")
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
    parser.add_argument("--group", default=None, help=f"one of {GROUPS}")
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.validate_seeds.split(",")]

    if args.mode == "cheap-search":
        cheap_search(args.n_configs, args.search_seed)
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
