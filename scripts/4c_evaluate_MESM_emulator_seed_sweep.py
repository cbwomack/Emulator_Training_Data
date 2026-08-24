#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
50-seed regeneration of the MESM vector emulator's baseline and three
optimized-side (constant/sine/both) result summaries, using the tuned
hyperparameters from 4c_MESM_baseline_hp_search.py / 4c_MESM_optimized_hp_
search.py - mirrors 0c_regenerate_checkpoints_co2.py's seed-sweep design for
the scalar CO2-only case, adapted for the vector (zonal-temperature)
emulator and its four training variants instead of six scenario groups.

Each (seed, variant) task is independent - unlike 0c_regenerate_checkpoints_
co2.py's "only one array task writes the shared baseline" carve-out (needed
there to avoid N parallel writers racing on one file), every task here
writes its own distinct output file, so no such carve-out is needed.

Only the small NRMSE-summary `results` dict is saved per (seed, variant) -
not the full preds/truths diagnostics 4c_evaluate_MESM_emulator.py's
single-run outputs include, since Figure 7's seed-spread plot only ever
reads results[eval_set][scenario]['global'] (see utils_inverse.
regenerate_fig7_MESM_cache_seed_sweep) and 200 files (50 seeds x 4 variants)
of full zonal prediction arrays would be needless storage for no plotting
benefit.

Usage:
    python 4c_evaluate_MESM_emulator_seed_sweep.py --seed 0 --variant baseline
    python 4c_evaluate_MESM_emulator_seed_sweep.py --seed 0 1 2 --variant both
    python 4c_evaluate_MESM_emulator_seed_sweep.py  # all 4 variants x seeds 0-49 (local convenience; use the submit_* SLURM array for the real 50-seed sweep)
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

import utils_inverse

OPT_GROUP = "all"
EVAL_DIR = "data/MESM/emis_driven/zonal_data_mean/"
HIDDEN_SIZES = [16]
OUT_DIR = Path("data/plotting/MESM_seed_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_CONFIG_PATH = "data/SI_results/hp_retune/MESM_vector/baseline_search/best_config_baseline.json"
OPTIMIZED_CONFIG_PATH = "data/SI_results/hp_retune/MESM_vector/optimized_search/best_config_optimized.json"

VARIANT_IC_MAP = {"constant": ["constant"], "sine": ["sine"], "both": ["constant", "sine"]}
ALL_VARIANTS = ["baseline", "constant", "sine", "both"]

_BASE_SETUP_CACHE = {}


def load_configs():
    baseline_cfg = json.load(open(BASELINE_CONFIG_PATH))["config"]
    optimized_cfg = json.load(open(OPTIMIZED_CONFIG_PATH))["config"]
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
            ic_list=VARIANT_IC_MAP[variant], group=OPT_GROUP, eval_dir=EVAL_DIR,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, nargs="+", default=list(range(50)))
    parser.add_argument("--variant", choices=ALL_VARIANTS, default=None,
                         help="one of baseline/constant/sine/both; default runs all four (SLURM array tasks should pass one)")
    args = parser.parse_args()

    baseline_cfg, optimized_cfg = load_configs()
    variants = [args.variant] if args.variant else ALL_VARIANTS

    for seed in args.seed:
        for variant in variants:
            run_one(seed, variant, baseline_cfg, optimized_cfg)


if __name__ == "__main__":
    main()
