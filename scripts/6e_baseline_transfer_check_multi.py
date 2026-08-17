#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Multi-agent analog of scripts/6e_baseline_transfer_check.py: does CO2's
tuned baseline emulator config (K=400, lr=0.0480, weight_decay=0.001 - see
data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json)
transfer to the all-5-agents-active (CO2/CH4/N2O/Sulfur/BC) case, or does
checkpoints/multi/'s baseline need its own independent dense K=400 search
(same pattern as scripts/6e_baseline_hp_search_k400_agent.py)?

Same cheap A/B as the single-forcing check: CO2's tuned config vs. the
original copied-over default (K=400, lr=0.05, weight_decay=0.01), 5 seeds
each, baseline training only (no bilevel outer loop) - cheap.

Usage:
    python 6e_baseline_transfer_check_multi.py --mode run-one --config co2_tuned --seed 0
    python 6e_baseline_transfer_check_multi.py --mode collect
    python 6e_baseline_transfer_check_multi.py --mode report
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import jax

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import utils_inverse

OUT_DIR = Path("data/SI_results/baseline_hp/transfer_check_multi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODE = "FaIR"
HIDDEN_SIZES = [16]
SEEDS = [0, 1, 2, 3, 4]

AGENTS = ["CO2", "CH4", "N2O", "Sulfur", "BC"]
ACTIVE_AGENTS = ("CO2", "CH4", "N2O", "Sulfur", "BC")

CO2_TUNED_PATH = Path("data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json")
EVAL_WEIGHTS = {"Tier 1": 7, "Tier 2": 5, "DECK": 2, "CS3": 2}

_SETUP_CACHE = {}


def _load_baseline_configs():
    co2_tuned = json.load(open(CO2_TUNED_PATH))["config"]
    return {
        "original": {"K": 400, "lr": 0.05, "weight_decay": 0.01},
        "co2_tuned": {"K": co2_tuned["K"], "lr": co2_tuned["lr"], "weight_decay": co2_tuned["weight_decay"]},
    }


def get_setup():
    if "multi" not in _SETUP_CACHE:
        setup = utils_inverse.run_inverse_experiment_setup(
            AGENTS, ACTIVE_AGENTS, mode=MODE, CS3=True, DAMIP=True, GeoMIP=True,
            seed=0, idx_demo=None,
        )
        _SETUP_CACHE["multi"] = setup
    return _SETUP_CACHE["multi"]


def _weighted_mean(eval_scores: dict):
    if not set(EVAL_WEIGHTS).issubset(eval_scores):
        return None
    num = sum(EVAL_WEIGHTS[k] * eval_scores[k] for k in EVAL_WEIGHTS)
    den = sum(EVAL_WEIGHTS.values())
    return num / den


def _result_path(config_name, seed):
    return OUT_DIR / f"transfer_multi_{config_name}_seed{seed}_result.json"


def run_one(config_name, seed):
    rp = _result_path(config_name, seed)
    if rp.is_file():
        print(f"[multi {config_name} seed={seed}] already done, skipping")
        return
    cfg = _load_baseline_configs()[config_name]
    setup = get_setup()
    t0 = time.perf_counter()
    paramsK, results, preds, gt = utils_inverse.evaluate_baseline_over_multiple_tests(
        emis_dict_train=setup["emis_dict_train_JAX"], eval_sets=setup["eval_sets"],
        key=jax.random.PRNGKey(seed), hidden_sizes=HIDDEN_SIZES,
        K=cfg["K"], lr=cfg["lr"], weight_decay=cfg["weight_decay"], mode=MODE,
    )
    eval_scores = {k: float(v["mean"]) for k, v in results.items()}
    wmean = _weighted_mean(eval_scores)
    dt = time.perf_counter() - t0
    row = {"config_name": config_name, "seed": seed, **cfg,
           "eval_scores": eval_scores, "weighted_score": wmean, "seconds": round(dt, 2)}
    print(f"[multi {config_name} seed={seed}] weighted_score={wmean:.4g} ({dt:.1f}s)", flush=True)
    tmp = rp.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, rp)


def collect():
    result_files = sorted(OUT_DIR.glob("transfer_multi_*_seed*_result.json"))
    if not result_files:
        print("no results yet")
        return
    rows = [json.load(open(p)) for p in result_files]
    out_path = OUT_DIR / "transfer_results.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"collected {len(rows)} results -> {out_path}")


def report():
    results_path = OUT_DIR / "transfer_results.json"
    if not results_path.is_file():
        print("no collected results yet, run --mode collect first")
        return
    rows = json.load(open(results_path))

    by_config = {}
    for r in rows:
        by_config.setdefault(r["config_name"], []).append(r["weighted_score"])

    orig = by_config.get("original")
    tuned = by_config.get("co2_tuned")
    if not orig or not tuned:
        print(f"missing results - orig={bool(orig)} tuned={bool(tuned)}")
        return
    orig_mean = float(np.mean(orig))
    tuned_mean = float(np.mean(tuned))
    pct_change = 100.0 * (orig_mean - tuned_mean) / orig_mean
    label = "transfers (reuse)" if pct_change >= -2.0 else "DOES NOT transfer (needs own K=400 search)"
    verdict = {"original_mean": orig_mean, "co2_tuned_mean": tuned_mean,
               "pct_change": pct_change, "verdict": label}
    print(f"{'original (mean)':>18} {'co2_tuned (mean)':>18} {'co2_tuned vs original':>24}")
    print(f"{orig_mean:>18.4g} {tuned_mean:>18.4g} {pct_change:>+22.1f}%  {label}")

    out_path = OUT_DIR / "transfer_verdict.json"
    with open(out_path, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nsaved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["run-one", "collect", "report"], required=True)
    parser.add_argument("--config", choices=["original", "co2_tuned"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "run-one":
        if args.config is None or args.seed is None:
            raise SystemExit("run-one requires --config and --seed")
        run_one(args.config, args.seed)
    elif args.mode == "collect":
        collect()
    elif args.mode == "report":
        report()


if __name__ == "__main__":
    main()
