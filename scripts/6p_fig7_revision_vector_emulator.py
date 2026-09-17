#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Redo of Figure 7 v1 (the MESM zonal/vector-output emulator comparison,
utils_inverse.generate_and_eval_emulator_vector) with the revision initial
conditions - "constant", "sine", "gaussian", and "both" (constant+sine
trained jointly, matching the original pipeline's "both") - in place of the
original checkpoints/co2/inverse_{constant,sine}_all_co2_only_MESM.pkl data.
This is the vector-emulator figure (scripts/4c_evaluate_MESM_emulator*.py,
utils_plotting.plot_scenario_difference_bars2), not the Stage 6o scalar-GMST
"Figure 7 v2" pipeline (scripts/6o_fig7_MESM_opt_*.py), which already covers
constant/sine/gaussian for a different (scalar) emulator.

Baseline is unchanged from the existing Figure 7: same [16]-hidden vector
MLP, trained on the Tier 1 ScenarioMIP scenarios via utils_inverse.
build_MESM_baseline_eval_sets(), same tuned hyperparameters (data/SI_results/
hp_retune/MESM_vector/baseline_search/best_config_baseline.json). Reused
via utils_inverse.build_MESM_baseline_eval_sets - not reimplemented here.

The optimized side reimplements (rather than calls) utils_inverse.
build_MESM_opt_eval_sets, because that function is hardcoded to the frozen,
published checkpoints/co2/inverse_{IC}_all_co2_only_MESM.pkl trajectories -
it is on the existing Figure 7's path and is deliberately left untouched.
This script instead reads the revision emissions directly from
data/MESM/emis_driven/MESM_inputs/opt_all_revision_{ic}.txt - the literal
file MESM was driven with - and asserts it matches the cached best-seed
trajectory in the Stage 6o seed-spread cache (fig7v2_seed_spread_MESM_tier1.
pkl) within 1e-5, so the provenance of these numbers stays checkable:

    IC        seed (of 50, argmin weighted_optimal)
    constant  38
    sine      32
    gaussian  9

MESM ground truth for the optimized side is data/MESM/emis_driven/
zonal_data_mean/optimized_revision/opt_all_revision_{ic}_mean.pkl, written
by scripts/4f_process_MESM_revision_data.py.

All three optimized variants use the tuned "optimized" hyperparameters
(data/SI_results/hp_retune/MESM_vector/optimized_search/best_config_
optimized.json, K=400 lr=0.149 wd=0.003) - the SAME config the existing
Figure 7 uses for its "both" variant, so these numbers are directly
comparable to it. That config was tuned on "both", not on any of these three
ICs individually; it is reused rather than re-tuned per the user's direction.

Usage:
    python 6p_fig7_revision_vector_emulator.py --mode run --seed 0
    python 6p_fig7_revision_vector_emulator.py --mode run                 # seeds 0-49, all variants
    python 6p_fig7_revision_vector_emulator.py --mode collect
    python 6p_fig7_revision_vector_emulator.py --mode report
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
from paths import DATA_DIR

REV_ICS = ["constant", "sine", "gaussian"]
# "both" trains jointly on constant+sine revision ground truth, matching the
# original pipeline's IC_VARIANTS["both"] = ["constant", "sine"]
# (scripts/4c_evaluate_MESM_emulator.py) - no gaussian counterpart, since the
# original "both" never existed for a third IC either.
IC_GROUPS = {"constant": ["constant"], "sine": ["sine"], "gaussian": ["gaussian"],
             "both": ["constant", "sine"]}
OPT_VARIANTS = ["constant", "sine", "gaussian", "both"]
ALL_VARIANTS = ["baseline"] + OPT_VARIANTS

EVAL_DIR = "data/MESM/emis_driven/zonal_data_mean/"
HIDDEN_SIZES = [16]
OUT_DIR = Path("data/plotting/MESM_revision_seed_sweep")
FINAL_CACHE_PATH = Path("data/SI_results/seed_uncertainty/fig7_revision_seed_spread_MESM.pkl")
SUMMARY_PATH = Path("data/SI_results/seed_uncertainty/fig7_revision_summary.json")

BASELINE_CONFIG_PATH = "data/SI_results/hp_retune/MESM_vector/baseline_search/best_config_baseline.json"
OPTIMIZED_CONFIG_PATH = "data/SI_results/hp_retune/MESM_vector/optimized_search/best_config_optimized.json"

# Provenance check: the revision .txt files are the best-seed (argmin
# weighted_optimal, of 50) trajectories from the Stage 6o seed sweep cache.
SEED_SPREAD_CACHE_PATH = "data/SI_results/seed_uncertainty/fig7v2_seed_spread_MESM_tier1.pkl"
EXPECTED_BEST_SEED = {"constant": 38, "sine": 32, "gaussian": 9}

# Ranking weights used everywhere else in this codebase for the single
# "overall" number (4c_MESM_*_hp_search.py, 6l_fig7_scalar_config_swap.py,
# 6o_fig7_MESM_opt_evaluate.py, 6n_supplement_table_export.py).
SET_WEIGHTS = {"Tier 1": 7, "Tier 2": 5, "DECK": 2, "CS3": 2}

_BASE_SETUP_CACHE = {}


def load_configs():
    baseline_cfg = json.load(open(BASELINE_CONFIG_PATH))["config"]
    optimized_cfg = json.load(open(OPTIMIZED_CONFIG_PATH))["config"]
    return baseline_cfg, optimized_cfg


def get_base_setup():
    if "setup" not in _BASE_SETUP_CACHE:
        _BASE_SETUP_CACHE["setup"] = utils_inverse.build_MESM_baseline_eval_sets(eval_dir=EVAL_DIR)
    return _BASE_SETUP_CACHE["setup"]


def _check_provenance(ic: str, co2_array: np.ndarray, tol: float = 1e-5) -> None:
    """Confirm the revision .txt still matches the cached best-seed trajectory
    documented in the module docstring, so a silently-replaced input file
    would be caught rather than quietly producing different numbers."""
    if not Path(SEED_SPREAD_CACHE_PATH).exists():
        print(f"[{ic}] provenance check skipped: {SEED_SPREAD_CACHE_PATH} not found")
        return
    with open(SEED_SPREAD_CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    expected_seed = EXPECTED_BEST_SEED[ic]
    cached_emis = np.asarray(cache[ic][expected_seed]["emissions"]).ravel()
    n = min(len(cached_emis), len(co2_array))
    err = float(np.abs(cached_emis[:n] - co2_array[:n]).max())
    if err > tol:
        raise SystemExit(
            f"[{ic}] emissions .txt no longer matches the documented best seed "
            f"{expected_seed} (max abs diff {err:.3e} > {tol}). The revision "
            f"input file may have changed - update EXPECTED_BEST_SEED / re-verify "
            f"provenance before trusting these results."
        )
    print(f"[{ic}] provenance OK: matches seed {expected_seed} of {SEED_SPREAD_CACHE_PATH} "
          f"(max abs diff {err:.2e})")


def build_revision_opt_sets(base: dict, ic_list: list[str]) -> dict:
    """Optimized-side train/eval sets for one or more revision ICs trained
    jointly (ic_list=["constant","sine"] for the "both" variant). Mirrors
    utils_inverse.build_MESM_opt_eval_sets's shape/contract but reads
    emissions from the revision .txt files (not frozen checkpoints) and
    ground truth from the optimized_revision/ pickles (not optimized/)."""
    emis_dict_opt = {}
    for ic in ic_list:
        emis_path = f"{DATA_DIR}/MESM/emis_driven/MESM_inputs/opt_all_revision_{ic}.txt"
        co2_array = np.loadtxt(emis_path, usecols=(2,), skiprows=2)
        _check_provenance(ic, co2_array)
        key = f"revision_{ic}"
        emis_dict_opt[key] = np.zeros((5, len(co2_array)))
        emis_dict_opt[key][0, :] = co2_array

    eval_emis_opt_sets = {"optimized": emis_dict_opt.copy()}
    scenarios_train = {"optimized_revision": [f"all_revision_{ic}" for ic in ic_list]}

    _, targets_dict_opt, _, _ = utils_inverse.generate_target_data(
        scenarios_train, data_dir=EVAL_DIR, opt=True
    )
    # generate_target_data keys targets by scenario name ("all_revision_{ic}"),
    # but build_dataset_vector_targets/prepare_data_vector match emis_dict keys
    # against target dict keys - rename to match emis_dict_opt's keys ("revision_{ic}").
    targets_dict_opt = {f"revision_{ic}": targets_dict_opt[f"all_revision_{ic}"] for ic in ic_list}
    eval_targets_opt_sets = {"optimized": dict(targets_dict_opt)}

    for eval_key in base["eval_targets_sets"]:
        eval_targets_opt_sets[eval_key] = base["eval_targets_sets"][eval_key]
        eval_emis_opt_sets[eval_key] = base["eval_emis_sets"][eval_key]

    return {
        "eval_emis_opt_sets": eval_emis_opt_sets,
        "eval_targets_opt_sets": eval_targets_opt_sets,
        "emis_dict_opt": emis_dict_opt,
        "targets_dict_opt": targets_dict_opt,
    }


def _out_path(variant: str, seed: int) -> Path:
    return OUT_DIR / f"{variant}_seed{seed}.pkl"


def run_one(seed: int, variant: str, baseline_cfg: dict, optimized_cfg: dict) -> None:
    out_path = _out_path(variant, seed)
    if out_path.exists():
        print(f"[{variant} seed={seed}] already done, skipping")
        return

    base = get_base_setup()

    if variant == "baseline":
        results, *_ = utils_inverse.generate_and_eval_emulator_vector(
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
        opt = build_revision_opt_sets(base, IC_GROUPS[variant])
        results, *_ = utils_inverse.generate_and_eval_emulator_vector(
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(results, f)
    os.replace(tmp, out_path)
    print(f"[{variant} seed={seed}] saved {out_path}")


def collect(n_seeds: int, seeds: list[int] | None = None,
           variants: list[str] | None = None, out_path: Path | None = None) -> None:
    seed_list = seeds if seeds is not None else list(range(n_seeds))
    variant_list = variants if variants is not None else ALL_VARIANTS
    out = out_path if out_path is not None else FINAL_CACHE_PATH
    all_results = {}
    missing = []
    for seed in seed_list:
        entry = {}
        for variant in variant_list:
            p = _out_path(variant, seed)
            if not p.exists():
                missing.append((seed, variant))
                continue
            key = "baseline" if variant == "baseline" else f"optimal_{variant}"
            with open(p, "rb") as f:
                entry[key] = pickle.load(f)
        if len(entry) == len(variant_list):
            all_results[seed] = entry
    if missing:
        print(f"missing {len(missing)} (seed, variant) pairs, e.g. {missing[:5]} - "
              f"collected {len(all_results)}/{len(seed_list)} complete seeds anyway")
    if not all_results:
        print("nothing to collect")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(all_results, f)
    print(f"collected {len(all_results)} seeds -> {out}")


def _weighted_overall(results: dict) -> float:
    num, den = 0.0, 0.0
    for set_name, w in SET_WEIGHTS.items():
        if set_name in results and "mean" in results[set_name]:
            num += w * results[set_name]["mean"]["global"]
            den += w
    return num / den if den > 0 else float("nan")


def report(cache_path: Path | None = None, summary_path: Path | None = None) -> None:
    cache_path = cache_path if cache_path is not None else FINAL_CACHE_PATH
    summary_path = summary_path if summary_path is not None else SUMMARY_PATH
    if not cache_path.exists():
        raise SystemExit(f"{cache_path} not found - run --mode collect first")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    seeds = sorted(cache.keys())
    print(f"{len(seeds)} seeds in cache\n")

    summary = {"n_seeds": len(seeds), "set_weights": SET_WEIGHTS, "by_ic": {}}

    base_overall = np.array([_weighted_overall(cache[s]["baseline"]) for s in seeds])
    print(f"baseline weighted-overall NRMSE: median {np.median(base_overall):.4f} "
          f"[{np.percentile(base_overall,25):.4f}, {np.percentile(base_overall,75):.4f}]")
    summary["baseline_weighted_overall"] = {
        "median": float(np.median(base_overall)),
        "p25": float(np.percentile(base_overall, 25)),
        "p75": float(np.percentile(base_overall, 75)),
    }

    available = [v for v in OPT_VARIANTS if f"optimal_{v}" in cache[seeds[0]]]
    for ic in available:
        key = f"optimal_{ic}"
        opt_overall = np.array([_weighted_overall(cache[s][key]) for s in seeds])
        beats = int(np.sum(opt_overall < base_overall))
        print(f"\n=== {ic} ===")
        print(f"  weighted-overall NRMSE: median {np.median(opt_overall):.4f} "
              f"[{np.percentile(opt_overall,25):.4f}, {np.percentile(opt_overall,75):.4f}]")
        print(f"  seeds where {ic} beats baseline (paired, same seed): {beats}/{len(seeds)}")

        set_rows = {}
        for set_name in SET_WEIGHTS:
            scenarios = set()
            for s in seeds:
                if set_name in cache[s]["baseline"]:
                    scenarios.update(k for k in cache[s]["baseline"][set_name] if k != "mean")
            for scen in sorted(scenarios):
                base_vals, opt_vals = [], []
                for s in seeds:
                    b = cache[s]["baseline"].get(set_name, {}).get(scen, {}).get("global")
                    o = cache[s][key].get(set_name, {}).get(scen, {}).get("global")
                    if b is not None and o is not None:
                        base_vals.append(b)
                        opt_vals.append(o)
                if not base_vals:
                    continue
                base_vals, opt_vals = np.array(base_vals), np.array(opt_vals)
                pct_change = 100 * (base_vals - opt_vals) / base_vals
                set_rows[f"{set_name}/{scen}"] = {
                    "nrmse_base_median": float(np.median(base_vals)),
                    "nrmse_opt_median": float(np.median(opt_vals)),
                    "pct_change_p25_p50_p75": [float(np.percentile(pct_change, q)) for q in (25, 50, 75)],
                }
        print(f"  {'scenario':<22}{'base':>9}{'opt':>9}{'%chg(p25,p50,p75)':>28}")
        for name, row in set_rows.items():
            p25, p50, p75 = row["pct_change_p25_p50_p75"]
            print(f"  {name:<22}{row['nrmse_base_median']:>9.4f}{row['nrmse_opt_median']:>9.4f}"
                  f"{f'({p25:+.1f}, {p50:+.1f}, {p75:+.1f})':>28}")

        summary["by_ic"][ic] = {
            "weighted_overall": {
                "median": float(np.median(opt_overall)),
                "p25": float(np.percentile(opt_overall, 25)),
                "p75": float(np.percentile(opt_overall, 75)),
            },
            "seeds_beating_baseline": beats,
            "n_seeds": len(seeds),
            "per_scenario": set_rows,
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["run", "collect", "report"], required=True)
    parser.add_argument("--seed", type=int, nargs="+", default=list(range(50)))
    parser.add_argument("--variant", choices=ALL_VARIANTS, default=None,
                        help="one of baseline/constant/sine/gaussian/both; default runs all five")
    parser.add_argument("--n-seeds", type=int, default=50, help="collect only")
    parser.add_argument("--variants", nargs="+", choices=ALL_VARIANTS, default=None,
                        help="collect/report: which variants to require/read per seed "
                             "(default: all five). E.g. --variants baseline constant sine both "
                             "to skip gaussian.")
    parser.add_argument("--out-path", type=Path, default=None,
                        help="collect: cache file to write (default: fig7_revision_seed_spread_MESM.pkl)")
    parser.add_argument("--cache-path", type=Path, default=None,
                        help="report: cache file to read (default: fig7_revision_seed_spread_MESM.pkl)")
    args = parser.parse_args()

    if args.mode == "run":
        baseline_cfg, optimized_cfg = load_configs()
        variants = [args.variant] if args.variant else ALL_VARIANTS
        for seed in args.seed:
            for variant in variants:
                run_one(seed, variant, baseline_cfg, optimized_cfg)
    elif args.mode == "collect":
        collect(args.n_seeds, seeds=args.seed if args.seed != list(range(50)) else None,
               variants=args.variants, out_path=args.out_path)
    elif args.mode == "report":
        report(cache_path=args.cache_path)


if __name__ == "__main__":
    main()
