#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Phase 0, Stage C (new, per user direction 2026-08-11): does CO2's tuned
baseline emulator config (K=400, lr=0.0480, weight_decay=0.001 - see
data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json)
transfer to the other single-forcing agents (CH4/N2O/Sulfur/BC), or does the
baseline need its own independent Stage-6e-style search per agent?

Motivation: the baseline is architecturally agent-agnostic (same MLP, same
K/lr/weight_decay knobs), but the learning rate that works well plausibly
depends on gradient magnitude, which is NOT agent-agnostic - different
agents' emissions/feature scales differ by orders of magnitude (e.g. the
optimizer's own per-agent step_size defaults span 50 (N2O) to 100,000 (CH4)
in scripts/SIa-SId's EXPERIMENTS dicts). Rather than assume transfer (risky)
or run a full independent 500-run search for all four agents up front
(expensive, and unnecessary if transfer holds), this runs a cheap A/B: CO2's
tuned config vs. the original copied-over default (K=400, lr=0.05,
weight_decay=0.01), 5 seeds each, on each agent's own eval_sets - baseline
training only, no bilevel outer loop, so this is fast (comparable to
6e_gap_persistence.py's per-run cost).

Decision rule (applied at --mode report time, not automated here): if CO2's
tuned config is comparable-or-better for an agent, Stage D can reuse
best_baseline_config_K400.json as-is for that agent. If it's clearly worse,
that agent needs its own dense K=400-only search (same pattern as
scripts/6e_baseline_hp_search_k400.py, generalized) before Stage D.

Usage:
    python 6e_baseline_transfer_check.py --mode run-one --agent CH4 --config co2_tuned --seed 0
    python 6e_baseline_transfer_check.py --mode collect
    python 6e_baseline_transfer_check.py --mode report
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

OUT_DIR = Path("data/SI_results/baseline_hp/transfer_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODE = "FaIR"
HIDDEN_SIZES = [16]
SEEDS = [0, 1, 2, 3, 4]

AGENTS = ["CH4", "N2O", "Sulfur", "BC"]

CO2_TUNED_PATH = Path("data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json")


def _load_baseline_configs():
    co2_tuned = json.load(open(CO2_TUNED_PATH))["config"]
    return {
        "original": {"K": 400, "lr": 0.05, "weight_decay": 0.01},
        "co2_tuned": {"K": co2_tuned["K"], "lr": co2_tuned["lr"], "weight_decay": co2_tuned["weight_decay"]},
    }


EVAL_WEIGHTS = {"Tier 1": 7, "Tier 2": 5, "DECK": 2, "CS3": 2}

_SETUP_CACHE = {}


def get_setup(agent):
    if agent not in _SETUP_CACHE:
        setup = utils_inverse.run_inverse_experiment_setup(
            [agent], (agent,), mode=MODE, CS3=True, DAMIP=False, GeoMIP=False,
            seed=0, idx_demo=None,
        )
        _SETUP_CACHE[agent] = setup
    return _SETUP_CACHE[agent]


def _weighted_mean(eval_scores: dict):
    if not set(EVAL_WEIGHTS).issubset(eval_scores):
        return None
    num = sum(EVAL_WEIGHTS[k] * eval_scores[k] for k in EVAL_WEIGHTS)
    den = sum(EVAL_WEIGHTS.values())
    return num / den


def _result_path(agent, config_name, seed):
    return OUT_DIR / f"transfer_{agent}_{config_name}_seed{seed}_result.json"


def run_one(agent, config_name, seed):
    rp = _result_path(agent, config_name, seed)
    if rp.is_file():
        print(f"[{agent} {config_name} seed={seed}] already done, skipping")
        return
    cfg = _load_baseline_configs()[config_name]
    setup = get_setup(agent)
    t0 = time.perf_counter()
    paramsK, results, preds, gt = utils_inverse.evaluate_baseline_over_multiple_tests(
        emis_dict_train=setup["emis_dict_train_JAX"], eval_sets=setup["eval_sets"],
        key=jax.random.PRNGKey(seed), hidden_sizes=HIDDEN_SIZES,
        K=cfg["K"], lr=cfg["lr"], weight_decay=cfg["weight_decay"], mode=MODE,
    )
    eval_scores = {k: float(v["mean"]) for k, v in results.items()}
    wmean = _weighted_mean(eval_scores)
    dt = time.perf_counter() - t0
    row = {"agent": agent, "config_name": config_name, "seed": seed, **cfg,
           "eval_scores": eval_scores, "weighted_score": wmean, "seconds": round(dt, 2)}
    print(f"[{agent} {config_name} seed={seed}] weighted_score={wmean:.4g} ({dt:.1f}s)", flush=True)
    tmp = rp.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, rp)


def collect():
    result_files = sorted(OUT_DIR.glob("transfer_*_seed*_result.json"))
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

    by_agent_config = {}
    for r in rows:
        by_agent_config.setdefault((r["agent"], r["config_name"]), []).append(r["weighted_score"])

    print(f"{'agent':>8} {'original (mean)':>18} {'co2_tuned (mean)':>18} {'co2_tuned vs original':>24}")
    verdict = {}
    for agent in AGENTS:
        orig = by_agent_config.get((agent, "original"))
        tuned = by_agent_config.get((agent, "co2_tuned"))
        if not orig or not tuned:
            print(f"{agent:>8}   (missing results - orig={bool(orig)} tuned={bool(tuned)})")
            continue
        orig_mean = float(np.mean(orig))
        tuned_mean = float(np.mean(tuned))
        pct_change = 100.0 * (orig_mean - tuned_mean) / orig_mean  # positive = tuned is better (lower NRMSE)
        label = "transfers (reuse)" if pct_change >= -2.0 else "DOES NOT transfer (needs own search)"
        verdict[agent] = {"original_mean": orig_mean, "co2_tuned_mean": tuned_mean,
                           "pct_change": pct_change, "verdict": label}
        print(f"{agent:>8} {orig_mean:>18.4g} {tuned_mean:>18.4g} {pct_change:>+22.1f}%  {label}")

    out_path = OUT_DIR / "transfer_verdict.json"
    with open(out_path, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nsaved -> {out_path}")
    print("\npct_change = 100*(original_mean - co2_tuned_mean)/original_mean : positive means CO2's "
          "tuned config improves on the original default for that agent too (transfer holds).")
    print("Threshold: co2_tuned must be within 2% of (or better than) the original default to count "
          "as 'transfers' - anything worse than that triggers an independent per-agent search.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["run-one", "collect", "report"], required=True)
    parser.add_argument("--agent", choices=AGENTS, default=None)
    parser.add_argument("--config", choices=["original", "co2_tuned"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "run-one":
        if args.agent is None or args.config is None or args.seed is None:
            raise SystemExit("run-one requires --agent, --config, and --seed")
        run_one(args.agent, args.config, args.seed)
    elif args.mode == "collect":
        collect()
    elif args.mode == "report":
        report()


if __name__ == "__main__":
    main()
