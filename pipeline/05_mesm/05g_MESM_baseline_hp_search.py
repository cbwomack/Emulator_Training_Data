#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
MESM vector-emulator hyperparameter retuning, baseline emulator.

Usage:
    python 05g_MESM_baseline_hp_search.py --mode search-one --config-idx 0 --seed 0
    python 05g_MESM_baseline_hp_search.py --mode search --n-configs 100 --seeds 0,1,2,3,4  # single-machine convenience
    python 05g_MESM_baseline_hp_search.py --mode collect
    python 05g_MESM_baseline_hp_search.py --mode finalize
    python 05g_MESM_baseline_hp_search.py --mode report
"""
import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import utils_inverse

OUT_DIR = Path("data/SI_results/hp_retune/MESM_vector/baseline_search")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_DIR = "data/MESM/emis_driven/zonal_data_mean/"
HIDDEN_SIZES = [16]
K_FIXED = 400

EVAL_WEIGHTS = {"Tier 1": 7, "Tier 2": 5, "DECK": 2, "CS3": 2}
CURRENT_DEFAULT = {"K": 400, "lr": 0.1, "weight_decay": 0.01}

N_CONFIGS_DEFAULT = 100
SEEDS_DEFAULT = [0, 1, 2, 3, 4]


def sample_config(rng):
    return {
        "K": K_FIXED,
        "lr": float(np.exp(rng.uniform(np.log(0.005), np.log(0.3)))),
        "weight_decay": float(rng.choice([0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 2e-1])),
    }


def _get_config_by_index(config_idx, search_seed=0):
    rng = np.random.default_rng(search_seed)
    cfg = None
    for _ in range(config_idx + 1):
        cfg = sample_config(rng)
    return cfg


_SETUP_CACHE = {}


def get_setup():
    if "setup" not in _SETUP_CACHE:
        _SETUP_CACHE["setup"] = utils_inverse.build_MESM_baseline_eval_sets(eval_dir=EVAL_DIR)
    return _SETUP_CACHE["setup"]


def _weighted_mean(eval_scores: dict):
    if not set(EVAL_WEIGHTS).issubset(eval_scores):
        return None
    num = sum(EVAL_WEIGHTS[k] * eval_scores[k] for k in EVAL_WEIGHTS)
    den = sum(EVAL_WEIGHTS.values())
    return num / den


def run_one(cfg, seed):
    setup = get_setup()
    t0 = time.perf_counter()
    try:
        results, preds, truths, paramsK, *_ = utils_inverse.generate_and_eval_emulator_vector(
            emis_dict_train=setup["emis_dict_tier1_JAX"],
            targets_dict_train=setup["targets_dict_tier1"],
            eval_emis_sets=setup["eval_emis_sets"],
            eval_targets_sets=setup["eval_targets_sets"],
            output_dim=setup["output_dim"],
            lat_coords=setup["lat_coords"],
            hidden_sizes=HIDDEN_SIZES,
            K=cfg["K"], lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            key_seed=seed, verbose=False,
        )
        eval_scores = {k: float(results[k]["mean"]["global"]) for k in EVAL_WEIGHTS if k in results}
        finite = all(np.isfinite(v) for v in eval_scores.values())
        score = _weighted_mean(eval_scores) if finite else float("inf")
        stable = finite and score is not None
    except Exception as e:
        print(f"  !! run failed: {type(e).__name__}: {e}", flush=True)
        eval_scores, score, stable = {}, float("inf"), False
    dt = time.perf_counter() - t0
    return {"score": score, "eval_scores": eval_scores, "stable": stable, "seconds": round(dt, 2)}


def _result_path(config_idx, seed):
    return OUT_DIR / f"search_cfg{config_idx}_seed{seed}_result.json"


def search_one(config_idx, seed, search_seed=0):
    rp = _result_path(config_idx, seed)
    if rp.is_file():
        print(f"[cfg={config_idx} seed={seed}] already done, skipping")
        return
    cfg = _get_config_by_index(config_idx, search_seed)
    res = run_one(cfg, seed)
    row = {"config_idx": config_idx, "seed": seed, **cfg, **res}
    print(f"[cfg={config_idx} seed={seed}] K={cfg['K']} lr={cfg['lr']:.4g} wd={cfg['weight_decay']:.4g} "
          f"score={res['score']:.4g} stable={res['stable']} ({res['seconds']}s)", flush=True)
    tmp = rp.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, rp)


def search(n_configs, seeds, search_seed=0):
    for idx in range(n_configs):
        for seed in seeds:
            search_one(idx, seed, search_seed)


def collect():
    result_files = sorted(OUT_DIR.glob("search_cfg*_seed*_result.json"))
    if not result_files:
        print("no results yet")
        return
    rows = [json.load(open(p)) for p in result_files]
    csv_path = OUT_DIR / "search_results.csv"
    fieldnames = ["config_idx", "seed", "K", "lr", "weight_decay", "score", "stable", "seconds"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"collected {len(rows)} results -> {csv_path}")


def finalize():
    csv_path = OUT_DIR / "search_results.csv"
    if not csv_path.is_file():
        print("no collected results yet, run --mode collect first")
        return
    rows = list(csv.DictReader(open(csv_path)))
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r["config_idx"], []).append(r)

    best_idx, best_mean, best_cfg = None, float("inf"), None
    for config_idx, rs in by_cfg.items():
        if not all(r["stable"] == "True" for r in rs):
            continue
        mean_score = float(np.mean([float(r["score"]) for r in rs]))
        if mean_score < best_mean:
            best_mean = mean_score
            best_idx = config_idx
            best_cfg = {"K": int(rs[0]["K"]), "lr": float(rs[0]["lr"]), "weight_decay": float(rs[0]["weight_decay"])}

    if best_idx is None:
        print("no fully-stable candidate yet")
        return

    out_path = OUT_DIR / "best_config_baseline.json"
    with open(out_path, "w") as f:
        json.dump({"config_idx": best_idx, "mean_score": best_mean, "config": best_cfg,
                   "n_seeds": len(by_cfg[best_idx])}, f, indent=2)
    print(f"winner = candidate {best_idx} (mean score {best_mean:.4g}, n_seeds={len(by_cfg[best_idx])}) -> {out_path}")
    print(f"  config: {best_cfg}")
    print(f"  (current/original baseline: {CURRENT_DEFAULT})")

    scored = []
    for config_idx, rs in by_cfg.items():
        if not all(r["stable"] == "True" for r in rs):
            continue
        mean_score = float(np.mean([float(r["score"]) for r in rs]))
        scored.append((mean_score, config_idx))
    scored.sort()
    print(f"\n  n_stable_configs={len(scored)} / {len(by_cfg)}")
    print(f"  top 5: {scored[:5]}")


def report():
    for kind, path in [("search results", OUT_DIR / "search_results.csv"),
                        ("best config", OUT_DIR / "best_config_baseline.json")]:
        status = "done" if path.is_file() else "missing"
        print(f"{kind}: {status} ({path})")
    n = len(list(OUT_DIR.glob("search_cfg*_seed*_result.json")))
    print(f"raw per-combo result files: {n}")
    best_path = OUT_DIR / "best_config_baseline.json"
    if best_path.is_file():
        print(json.dumps(json.load(open(best_path)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["search", "search-one", "collect", "finalize", "report"], required=True)
    parser.add_argument("--n-configs", type=int, default=N_CONFIGS_DEFAULT)
    parser.add_argument("--search-seed", type=int, default=0, help="RNG seed for sampling configs")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4", help="comma list, search mode only")
    parser.add_argument("--config-idx", type=int, default=None, help="search-one only")
    parser.add_argument("--seed", type=int, default=None, help="search-one only")
    args = parser.parse_args()

    if args.mode == "search":
        seeds = [int(s) for s in args.seeds.split(",")]
        search(args.n_configs, seeds, args.search_seed)
    elif args.mode == "search-one":
        if args.config_idx is None or args.seed is None:
            raise SystemExit("search-one requires --config-idx and --seed")
        search_one(args.config_idx, args.seed, args.search_seed)
    elif args.mode == "collect":
        collect()
    elif args.mode == "finalize":
        finalize()
    elif args.mode == "report":
        report()


if __name__ == "__main__":
    main()
