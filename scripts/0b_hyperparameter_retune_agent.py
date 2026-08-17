#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Phase 0, Stage 0b (agent-generalized): hyperparameter re-tuning for the
CH4/N2O/Sulfur/BC single-forcing bilevel outer-loop optimizer.

Generalizes scripts/0b_hyperparameter_retune.py (CO2-only, left untouched -
its already-validated best_config_unified.json is not to be disturbed) to
take a `--agent {CH4,N2O,Sulfur,BC}` argument. Same design in every other
respect: ONE shared hyperparameter configuration - step_size, momentum,
nesterov, K_inner, lr_inner, wd_inner, smoothness_weight, batch_size - used
identically across all 6 groups (H-ext, tier1, tier2, DECK, CS3, all) for
the given agent, same two-phase cheap-search/validate design, same
GROUP_WEIGHTS scheme, same MAX_SPIKES/MAX_REGRET stability criteria.

The one thing that must NOT be reused as-is from CO2 (per the standing
caution already in REVISIONS.md): the step_size search range. Each agent's
own EXPERIMENTS defaults span wildly different scales (CH4: 10,000-100,000;
N2O: 50-1,000; Sulfur: 5,000-10,000; BC: flat 100 - vs. CO2's 848-5,886), so
AGENT_CONFIG below re-centers each agent's log-uniform range on its own
known defaults (the *originally* extracted per-group values, before Stage A
filled in placeholder step_sizes for CH4/Sulfur's newly-added groups -
those placeholders are irrelevant to this range choice).

Usage (identical mode set to the CO2 script, all now requiring --agent):
    python 0b_hyperparameter_retune_agent.py --agent CH4 --mode cheap-search
    python 0b_hyperparameter_retune_agent.py --agent CH4 --mode select
    python 0b_hyperparameter_retune_agent.py --agent CH4 --mode validate-one --cand-idx 0 --group tier1 --validate-seeds 0
    python 0b_hyperparameter_retune_agent.py --agent CH4 --mode validate    # single-machine convenience wrapper
    python 0b_hyperparameter_retune_agent.py --agent CH4 --mode collect
    python 0b_hyperparameter_retune_agent.py --agent CH4 --mode finalize
    python 0b_hyperparameter_retune_agent.py --agent CH4 --mode report
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

SEARCH_NUM_UPDATES = 300
VALIDATE_NUM_UPDATES = 1000
TOP_K = 5

# Same stability criteria as the CO2 script - agent-agnostic.
SPIKE_RATIO_THRESHOLD = 1.5
MAX_SPIKES = 5
MAX_REGRET = 0.15

TUNABLE_KEYS = ["step_size", "momentum", "nesterov", "K_inner", "lr_inner",
                "wd_inner", "smoothness_weight", "batch_size"]

# Same weighting convention as the CO2 script (matches Fig 4/SI Fig 6's
# existing scenario-count weighting) - structural, not agent-specific.
GROUP_WEIGHTS = {"H-ext": 1, "tier1": 7, "tier2": 5, "DECK": 2, "CS3": 2, "all": 17}

# Per-agent: which companion script's EXPERIMENTS dict to load, the
# agents/active_agents tuple for run_inverse_experiment_setup, and the
# step_size log-uniform search range re-centered on that agent's own
# originally-extracted per-group defaults (see module docstring).
AGENT_CONFIG = {
    "CH4": {
        "module": "SIa_inverse_CH4_only.py",
        "agents": ["CH4"], "active_agents": ("CH4",),
        "step_size_range": (3000.0, 150000.0),  # known H-ext=10000, tier2=100000
    },
    "N2O": {
        "module": "SIb_inverse_N2O_only.py",
        "agents": ["N2O"], "active_agents": ("N2O",),
        "step_size_range": (15.0, 3000.0),  # known H-ext=1000, rest=50
    },
    "Sulfur": {
        "module": "SIc_inverse_Sulfur_only.py",
        "agents": ["Sulfur"], "active_agents": ("Sulfur",),
        "step_size_range": (1500.0, 15000.0),  # known H-ext=5000, tier1=10000
    },
    "BC": {
        "module": "SId_inverse_BC_only.py",
        "agents": ["BC"], "active_agents": ("BC",),
        "step_size_range": (30.0, 400.0),  # known: flat 100 across every group
    },
}


def load_experiments_module(agent):
    spec = importlib.util.spec_from_file_location(
        f"_exp_mod_{agent}",
        PROJECT_ROOT / "scripts" / "supplementary_notebooks" / AGENT_CONFIG[agent]["module"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bs_label(bs):
    return "full" if bs is None else str(bs)


def sample_config(rng, step_size_range):
    lo, hi = step_size_range
    return {
        "step_size": float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
        "momentum": float(rng.choice([0.7, 0.85, 0.9, 0.95, 0.99])),
        "nesterov": bool(rng.choice([True, False])),
        "K_inner": 400,  # locked to the manuscript's methodology, not searched
        "lr_inner": float(np.exp(rng.uniform(np.log(0.01), np.log(0.15)))),
        "wd_inner": float(rng.choice([0.0, 1e-3, 3e-3, 1e-2, 3e-2])),
        "smoothness_weight": float(rng.choice([0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4])),
        "batch_size": [None, 16, 32, 64][int(rng.integers(0, 4))],
    }


_SETUP_CACHE = {}


def get_setup_and_groups(agent, seed=0):
    key = (agent, seed)
    if key not in _SETUP_CACHE:
        cfg = AGENT_CONFIG[agent]
        setup = utils_inverse.run_inverse_experiment_setup(
            cfg["agents"], cfg["active_agents"], mode=MODE, CS3=True, DAMIP=False, GeoMIP=False,
            seed=seed, idx_demo=None,
        )
        group_emis_dicts = utils_inverse.build_group_emis_dicts(
            setup["emis_dict_train_JAX"], setup["eval_sets"],
        )
        _SETUP_CACHE[key] = (setup, group_emis_dicts)
    return _SETUP_CACHE[key]


def run_one(base_exp, group_emis_dict, params0, cfg, num_updates, seed, active_agents,
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
            active_agents=active_agents,
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


# ----------------------------------------------------------------------
# Phase 1: cheap-search - each candidate config evaluated on every group
# ----------------------------------------------------------------------

def _cheap_result_path(out_dir, config_idx, group):
    return out_dir / f"cheap_unified_cfg{config_idx}_{group}_result.json"


def _get_config_by_index(config_idx, search_seed, step_size_range):
    rng = np.random.default_rng(search_seed)
    cfg = None
    for _ in range(config_idx + 1):
        cfg = sample_config(rng, step_size_range)
    return cfg


def cheap_search_one(agent, out_dir, config_idx, group, search_seed):
    rp = _cheap_result_path(out_dir, config_idx, group)
    if rp.is_file():
        print(f"[{agent} cheap-search cfg={config_idx} group={group}] already done, skipping")
        return

    exp_mod = load_experiments_module(agent)
    if group not in exp_mod.EXPERIMENTS:
        raise SystemExit(f"unknown group {group!r}; available: {sorted(exp_mod.EXPERIMENTS)}")
    setup, group_emis_dicts = get_setup_and_groups(agent, seed=0)
    step_size_range = AGENT_CONFIG[agent]["step_size_range"]
    cfg = _get_config_by_index(config_idx, search_seed, step_size_range)
    base_exp = exp_mod.EXPERIMENTS[group]
    active_agents = AGENT_CONFIG[agent]["active_agents"]

    res = run_one(base_exp, group_emis_dicts[group], setup["params0"], cfg,
                  SEARCH_NUM_UPDATES, seed=0, active_agents=active_agents, checkpoint_path=None)
    row = {"config_idx": config_idx, "group": group, **cfg,
           "batch_size": _bs_label(cfg["batch_size"]), **res}
    print(f"[{agent} cheap-search cfg={config_idx} group={group}] score={res['score']:.4g} "
          f"stable={res['stable']} ({res['seconds']}s)", flush=True)
    tmp = rp.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, rp)


def cheap_search(agent, out_dir, n_configs, search_seed):
    exp_mod = load_experiments_module(agent)
    groups = list(exp_mod.EXPERIMENTS)
    for idx in range(n_configs):
        for group in groups:
            cheap_search_one(agent, out_dir, idx, group, search_seed)


def _load_cheap_results(out_dir):
    rows = [json.load(open(p)) for p in sorted(out_dir.glob("cheap_unified_cfg*_result.json"))]
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r["config_idx"], {})[r["group"]] = r
    return by_cfg


def select_candidates(out_dir, top_k):
    by_cfg = _load_cheap_results(out_dir)
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

    out_path = out_dir / "top_candidates_unified.json"
    with open(out_path, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"wrote {len(candidates)} top candidates -> {out_path}")
    for c in candidates:
        print(f"  cfg{c['config_idx']}: weighted_score={c['cheap_weighted_score']:.4g}")


def _load_candidates(out_dir):
    p = out_dir / "top_candidates_unified.json"
    if not p.is_file():
        return None
    return json.load(open(p))


# ----------------------------------------------------------------------
# Phase 2: validate - top-K candidates, all 6 groups, several seeds
# ----------------------------------------------------------------------

def _validate_result_path(out_dir, cand_idx, group, seed, num_updates):
    return out_dir / f"validate_unified_cand{cand_idx}_{group}_seed{seed}_n{num_updates}_result.json"


def validate_one(agent, out_dir, cand_idx, group, seed, num_updates):
    candidates = _load_candidates(out_dir)
    if candidates is None:
        print("no top-candidates file yet, run --mode select first")
        return
    if cand_idx >= len(candidates):
        print(f"cand_idx={cand_idx} out of range ({len(candidates)} candidates)")
        return

    result_path = _validate_result_path(out_dir, cand_idx, group, seed, num_updates)
    if result_path.is_file():
        print(f"[{agent} validate cand={cand_idx} group={group} seed={seed}] already done, skipping")
        return

    setup, group_emis_dicts = get_setup_and_groups(agent, seed=seed)
    exp_mod = load_experiments_module(agent)
    base_exp = exp_mod.EXPERIMENTS[group]
    cfg = candidates[cand_idx]["config"]
    active_agents = AGENT_CONFIG[agent]["active_agents"]

    ckpt = out_dir / f"validate_unified_cand{cand_idx}_{group}_seed{seed}.pkl"
    res = run_one(base_exp, group_emis_dicts[group], setup["params0"], cfg,
                  num_updates, seed=seed, active_agents=active_agents, checkpoint_path=str(ckpt))
    row = {"cand_idx": cand_idx, "group": group, "seed": seed, "num_updates": num_updates,
           **cfg, "batch_size": _bs_label(cfg["batch_size"]), **res}
    print(f"[{agent} validate cand={cand_idx} group={group} seed={seed}] score={res['score']:.4g} "
          f"stable={res['stable']} ({res['seconds']}s)", flush=True)

    tmp_path = result_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(row, f)
    os.replace(tmp_path, result_path)


def validate(agent, out_dir, seeds, num_updates):
    candidates = _load_candidates(out_dir)
    if candidates is None:
        print("no top-candidates file yet, run --mode select first")
        return
    for cand_idx in range(len(candidates)):
        for group in GROUP_WEIGHTS:
            for seed in seeds:
                validate_one(agent, out_dir, cand_idx, group, seed, num_updates)
    collect(out_dir, num_updates)
    finalize(out_dir, seeds, num_updates)


def collect(out_dir, num_updates):
    pattern = f"validate_unified_cand*_*_seed*_n{num_updates}_result.json"
    result_files = sorted(out_dir.glob(pattern))
    if not result_files:
        print(f"no per-combo result files found for num_updates={num_updates}")
        return
    rows = [json.load(open(p)) for p in result_files]

    csv_path = out_dir / "validate_unified.csv"
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


def finalize(out_dir, seeds, num_updates):
    csv_path = out_dir / "validate_unified.csv"
    if not csv_path.is_file():
        print("no validate results yet")
        return
    candidates = _load_candidates(out_dir)
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
    out_path = out_dir / "best_config_unified.json"
    with open(out_path, "w") as f:
        json.dump({"cand_idx": best_idx, "weighted_validate_score": best_score,
                   "per_group_mean_score": best_breakdown, "config": winner}, f, indent=2)
    print(f"winner = candidate {best_idx} (weighted score {best_score:.4g}) -> {out_path}")
    for g, s in best_breakdown.items():
        print(f"  {g:6s}: {s:.4g}")


def report(out_dir):
    for kind, path in [("cheap-search results", None),
                        ("top-candidates", out_dir / "top_candidates_unified.json"),
                        ("validate results", out_dir / "validate_unified.csv"),
                        ("best-config", out_dir / "best_config_unified.json")]:
        if path is None:
            n = len(list(out_dir.glob("cheap_unified_cfg*_result.json")))
            print(f"cheap-search results: {n} files")
            continue
        status = "done" if path.is_file() else "missing"
        print(f"{kind}: {status} ({path})")
    best_path = out_dir / "best_config_unified.json"
    if best_path.is_file():
        print(json.dumps(json.load(open(best_path)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=list(AGENT_CONFIG), required=True)
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

    agent = args.agent
    out_dir = Path(f"data/SI_results/hp_retune/{agent}")
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.validate_seeds.split(",")]

    if args.mode == "cheap-search":
        cheap_search(agent, out_dir, args.n_configs, args.search_seed)
    elif args.mode == "cheap-search-one":
        if args.config_idx is None or args.group is None:
            raise SystemExit("cheap-search-one requires --config-idx and --group")
        cheap_search_one(agent, out_dir, args.config_idx, args.group, args.search_seed)
    elif args.mode == "select":
        select_candidates(out_dir, args.top_k)
    elif args.mode == "validate":
        validate(agent, out_dir, seeds, args.validate_num_updates)
    elif args.mode == "validate-one":
        if args.cand_idx is None or args.group is None or len(seeds) != 1:
            raise SystemExit("validate-one requires --cand-idx, --group, and a single --validate-seeds value")
        validate_one(agent, out_dir, args.cand_idx, args.group, seeds[0], args.validate_num_updates)
    elif args.mode == "collect":
        collect(out_dir, args.validate_num_updates)
    elif args.mode == "finalize":
        finalize(out_dir, seeds, args.validate_num_updates)
    elif args.mode == "report":
        report(out_dir)


if __name__ == "__main__":
    main()
