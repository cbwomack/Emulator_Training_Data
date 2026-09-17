#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
One-off ablation, not part of the reviewer-response pipeline: what happens to
Figure 7 (the MESM vector/zonal-temperature emulator) if it is trained with
the scalar (non-vector) CO2 GMST emulator's own tuned hyperparameters instead
of Figure 7's own dedicated vector-tuned configs (4c_MESM_baseline_hp_search.py
/ 4c_MESM_optimized_hp_search.py)?

Mirrors 4c_evaluate_MESM_emulator_seed_sweep.py's design exactly (same
build_MESM_baseline_eval_sets/build_MESM_opt_eval_sets/
generate_and_eval_emulator_vector calls, same 50-seed loop), swapping only
the two config sources:

- baseline: data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json
  (lr=0.048, weight_decay=0.001, K=400) - Stage 6e's dense re-tune of the
  scalar CO2 baseline emulator.
- optimized: data/SI_results/hp_retune/best_config_unified.json (lr_inner=
  0.0252, wd_inner=0.03, K_inner=400) - Stage 0b's original bilevel-search
  winner for the scalar CO2 optimized emulator, predating this MESM/Fig7
  work. Only the "both" (constant+sine combined) optimized training variant
  is run, matching the main-text comparison.

Only 2 of the 4 usual variants run (baseline, both) - not constant/sine
individually, per the user's own scoping choice.

Usage:
    python 6l_fig7_scalar_config_swap.py --seed 0 --variant baseline
    python 6l_fig7_scalar_config_swap.py --seed 0 1 2 --variant both
    python 6l_fig7_scalar_config_swap.py  # both variants x seeds 0-49
    python 6l_fig7_scalar_config_swap.py --mode collect
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
import utils_inverse

OPT_GROUP = "all"
EVAL_DIR = "data/MESM/emis_driven/zonal_data_mean/"
HIDDEN_SIZES = [16]
OUT_DIR = Path("data/plotting/MESM_seed_sweep_scalarcfg")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALAR_BASELINE_CONFIG_PATH = "data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json"
SCALAR_OPTIMIZED_CONFIG_PATH = "data/SI_results/hp_retune/best_config_unified.json"

FINAL_PATH = Path("data/SI_results/seed_uncertainty/fig7_seed_spread_scalarcfg_MESM.pkl")

VARIANTS = ["baseline", "both"]
WEIGHTS = {"Tier 1": 7, "Tier 2": 5, "DECK": 2, "CS3": 2}

_BASE_SETUP_CACHE = {}


def load_configs():
    baseline_cfg = json.load(open(SCALAR_BASELINE_CONFIG_PATH))["config"]
    opt_raw = json.load(open(SCALAR_OPTIMIZED_CONFIG_PATH))["config"]
    optimized_cfg = {"K": opt_raw["K_inner"], "lr": opt_raw["lr_inner"], "weight_decay": opt_raw["wd_inner"]}
    return baseline_cfg, optimized_cfg


def get_base_setup():
    if "setup" not in _BASE_SETUP_CACHE:
        _BASE_SETUP_CACHE["setup"] = utils_inverse.build_MESM_baseline_eval_sets(eval_dir=EVAL_DIR)
    return _BASE_SETUP_CACHE["setup"]


def _out_path(variant, seed):
    return OUT_DIR / f"{variant}_seed{seed}.pkl"


def run_one(seed, variant, baseline_cfg, optimized_cfg):
    out_path = _out_path(variant, seed)
    if out_path.exists():
        print(f"[{variant} seed={seed}] already done, skipping")
        return

    base = get_base_setup()

    if variant == "baseline":
        results, preds, truths, paramsK, *_ = utils_inverse.generate_and_eval_emulator_vector(
            emis_dict_train=base["emis_dict_tier1_JAX"],
            targets_dict_train=base["targets_dict_tier1"],
            eval_emis_sets=base["eval_emis_sets"],
            eval_targets_sets=base["eval_targets_sets"],
            output_dim=base["output_dim"],
            lat_coords=base["lat_coords"],
            hidden_sizes=HIDDEN_SIZES,
            K=baseline_cfg["K"], lr=baseline_cfg["lr"], weight_decay=baseline_cfg["weight_decay"],
            key_seed=seed, verbose=True,
        )
    else:
        opt = utils_inverse.build_MESM_opt_eval_sets(
            base["eval_emis_sets"], base["eval_targets_sets"],
            ic_list=["constant", "sine"], group=OPT_GROUP, eval_dir=EVAL_DIR,
        )
        results, preds, truths, paramsK, *_ = utils_inverse.generate_and_eval_emulator_vector(
            emis_dict_train=opt["emis_dict_opt"],
            targets_dict_train=opt["targets_dict_opt"],
            eval_emis_sets=opt["eval_emis_opt_sets"],
            eval_targets_sets=opt["eval_targets_opt_sets"],
            output_dim=base["output_dim"],
            lat_coords=base["lat_coords"],
            hidden_sizes=HIDDEN_SIZES,
            K=optimized_cfg["K"], lr=optimized_cfg["lr"], weight_decay=optimized_cfg["weight_decay"],
            key_seed=seed, verbose=True,
        )

    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(results, f)
    os.replace(tmp, out_path)
    print(f"[{variant} seed={seed}] saved {out_path}")


def collect():
    merged = {}
    missing = []
    for seed in range(50):
        entry = {}
        for variant in VARIANTS:
            p = _out_path(variant, seed)
            if not p.exists():
                missing.append((variant, seed))
                continue
            with open(p, "rb") as f:
                entry[variant] = pickle.load(f)
        if len(entry) == len(VARIANTS):
            merged[seed] = entry

    if missing:
        print(f"WARNING: {len(missing)} (variant, seed) results missing, e.g. {missing[:5]}")

    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_PATH, "wb") as f:
        pickle.dump(merged, f)
    print(f"Saved {FINAL_PATH} ({len(merged)}/50 seeds complete)")

    print("\n=== Summary (median [IQR], Fig-4/Fig-7 weighting Tier1:7/Tier2:5/DECK:2/CS3:2) ===")
    for variant in VARIANTS:
        per_set = {s: [] for s in WEIGHTS}
        for seed in sorted(merged):
            for s in WEIGHTS:
                per_set[s].append(merged[seed][variant][s]["mean"]["global"])
        n = len(per_set["Tier 1"])
        overall = np.array([
            sum(WEIGHTS[s] * per_set[s][i] for s in WEIGHTS) / sum(WEIGHTS.values())
            for i in range(n)
        ])
        med, q1, q3 = np.median(overall), *np.percentile(overall, [25, 75])
        print(f"{variant:10s} overall  median={med:.4f}  IQR=[{q1:.4f},{q3:.4f}]  (n={n})")
        for s in WEIGHTS:
            arr = np.array(per_set[s])
            m, lo, hi = np.median(arr), *np.percentile(arr, [25, 75])
            print(f"    {s:8s} median={m:.4f} IQR=[{lo:.4f},{hi:.4f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, nargs="+", default=list(range(50)))
    parser.add_argument("--variant", choices=VARIANTS, default=None,
                         help="one of baseline/both; default runs both")
    parser.add_argument("--mode", choices=["run", "collect"], default="run")
    args = parser.parse_args()

    if args.mode == "collect":
        collect()
        return

    baseline_cfg, optimized_cfg = load_configs()
    variants = [args.variant] if args.variant else VARIANTS

    for seed in args.seed:
        for variant in variants:
            run_one(seed, variant, baseline_cfg, optimized_cfg)


if __name__ == "__main__":
    main()
