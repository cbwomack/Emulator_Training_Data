#!/usr/bin/env python
"""
Companion script for 4c_evaluate_MESM_emulator.ipynb: trains and evaluates the
vector-output (zonal-temperature) MLP emulator against MESM output, once on
the baseline (ScenarioMIP tier1) training set and once on the CO2-only
inverse-optimized ("all", constant+sine initial conditions) training set from
checkpoints/co2/. Writes the pickles the notebook needs to reload and plot.
Runs standalone so it can be scheduled.

Fixes one portability bug and preserves two other pre-existing quirks as-is,
flagged rather than silently fixed:

1. Fixed: the notebook hardcoded an absolute
   /Users/chriswomack/Documents/PhD/Project 2/data/MESM/emis_driven path for
   the DECK ensemble text files - same class of bug as 2c_calibrate_MESM, now
   uses paths.DATA_DIR.
2. Fixed (execution order): the notebook's cell order doesn't match its real
   dependency order - the cell that injects an 'optimized' eval set into
   eval_emis_sets/eval_targets_sets (originally cell 6, "# Run after
   optimization") reads eval_emis_opt_sets/eval_targets_opt_sets, which are
   only defined two cells *later* (originally cell 11). The notebook only
   ever ran correctly because a human executed the cells out of visual order
   (3, 4, 5, 11, 6, 7, 8, 9, 12, 13, 14, 15 - confirmed by cell 7's own
   printed "optimized Mean Global NRMSE" line, which only appears if
   eval_emis_sets already contains 'optimized' by the time cell 7 runs).
   This script uses that real dependency order, not the notebook's visual
   top-to-bottom order.
3. NOT fixed (preserved as-is, flagged for Phase 5): the saved optimal-
   emulator-results filename bakes in a loop-variable leak. The notebook
   loops `for IC in ['constant', 'sine']: ...` to build the combined
   constant+sine optimized training set, then later saves the *combined*
   result to a filename built from the *last* value IC happened to hold
   (`optimal_co2_only_MESM_{IC}.pkl` -> always "..._sine.pkl", regardless of
   the fact the result is the combined training run, not a sine-only one).
   This script reproduces that exact (mislabeled-but-functional) filename
   rather than "fixing" it to something like "..._combined.pkl", since that
   would be a silent, unrequested rename of an artifact other code may
   already reference.

Also saves a *_diagnostics.pkl pair (lat_coords + preds/truths) alongside the
literal baseline/optimal result pickles the original notebook cells produced,
since the notebook's own plot_zonal_predictions calls need preds/truths that
the original notebook cells never persisted to disk (they only existed as
in-memory variables in the same kernel session) - same pattern used for
2b_optimize_GeoMIP's emis_G6sulfur_diagnostics.pickle.

Usage:
    python 4c_evaluate_MESM_emulator.py
"""
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import utils_inverse
from paths import DATA_DIR

EVAL_DIR = "data/MESM/emis_driven/zonal_data_mean/"
BASELINE_PATH = "data/plotting/baseline_co2_only_MESM.pkl"
BASELINE_DIAGNOSTICS_PATH = "data/plotting/baseline_co2_only_MESM_diagnostics.pkl"
OPT_GROUP = "all"
OPT_ICS = ["constant", "sine"]
# Reproduces the notebook's loop-variable-leak filename (see module docstring,
# point 3) - always the *last* IC in OPT_ICS, regardless of what the saved
# result actually contains (the combined constant+sine training run).
OPT_PATH = f"data/plotting/optimal_co2_only_MESM_{OPT_ICS[-1]}.pkl"
OPT_DIAGNOSTICS_PATH = f"data/plotting/optimal_co2_only_MESM_{OPT_ICS[-1]}_diagnostics.pkl"


def build_eval_sets():
    agents = ["CO2"]
    scenarios_eval = {
        "Tier 1": ["historical", "H-ext", "L", "M", "ML", "VLHO", "VLLO-ext"],
        "Tier 2": ["H-ext-OS", "M-ext", "ML-ext", "L-ext", "VLHO-ext"],
        "DECK": ["1pctCO2", "2xCO2"],
        "CS3": ["AA", "CT", "historical"],
    }

    (eval_emis_sets, emis_dict_tier1_JAX, emis_dict_tier2_JAX, emis_dict_CS3_JAX, emis_dict_all_JAX) = (
        utils_inverse.generate_eval_data(agents, DECK=False, CS3=True, DAMIP=False, GeoMIP=False)
    )

    (eval_targets_sets, targets_dict_tier1, targets_dict_tier2, targets_dict_DECK, targets_dict_CS3, output_dim, lat_coords) = (
        utils_inverse.generate_target_data(scenarios_eval, data_dir=EVAL_DIR)
    )

    emis_path = str(DATA_DIR / "MESM" / "emis_driven")
    emis_1pct_path = f"{emis_path}/1PRCO2/carbemiss.txt"
    emis_1pct = np.loadtxt(emis_1pct_path, usecols=(2,), skiprows=2)
    emis_mat_1pct = np.zeros((5, len(emis_1pct)))
    emis_mat_1pct[0, :] = emis_1pct

    emis_2xCO2_path = f"{emis_path}/2xCO2/implco2emiss.3100.25.txt"
    emis_2xCO2 = np.loadtxt(emis_2xCO2_path, usecols=(2,))
    emis_mat_2xCO2 = np.zeros((5, len(emis_2xCO2)))
    emis_mat_2xCO2[0, :] = emis_2xCO2

    eval_emis_sets["DECK"] = {"1pctCO2": emis_mat_1pct, "2xCO2": emis_mat_2xCO2}

    return eval_emis_sets, eval_targets_sets, emis_dict_tier1_JAX, targets_dict_tier1, output_dim, lat_coords


def build_opt_sets(eval_emis_sets, eval_targets_sets):
    emis_dict_opt = {}
    for IC in OPT_ICS:
        opt_path = f"checkpoints/co2/inverse_{IC}_{OPT_GROUP}_co2_only_MESM.pkl"
        with open(opt_path, "rb") as f:
            res = pickle.load(f)
        co2_array = res["U_traj"][-1]["CO2"]
        emis_dict_opt[OPT_GROUP + "_" + IC] = np.zeros((5, len(co2_array)))
        emis_dict_opt[OPT_GROUP + "_" + IC][0, :] = co2_array.copy()

    eval_emis_opt_sets = {"optimized": emis_dict_opt.copy()}
    scenarios_train = {"optimized": [OPT_GROUP + "_" + IC for IC in OPT_ICS]}

    eval_targets_opt_sets, targets_dict_opt, _, _ = utils_inverse.generate_target_data(
        scenarios_train, data_dir=EVAL_DIR, opt=True
    )

    for key in eval_targets_sets:
        eval_targets_opt_sets[key] = eval_targets_sets[key].copy()
        eval_emis_opt_sets[key] = eval_emis_sets[key].copy()

    return eval_emis_opt_sets, eval_targets_opt_sets, emis_dict_opt, targets_dict_opt


def main():
    eval_emis_sets, eval_targets_sets, emis_dict_tier1_JAX, targets_dict_tier1, output_dim, lat_coords = build_eval_sets()

    eval_emis_opt_sets, eval_targets_opt_sets, emis_dict_opt, targets_dict_opt = build_opt_sets(
        eval_emis_sets, eval_targets_sets
    )

    # Inject the 'optimized' eval set into eval_emis_sets/eval_targets_sets
    # *before* the baseline emulator is trained/evaluated below - this is the
    # real dependency order (see module docstring, point 2).
    eval_emis_sets["optimized"] = eval_emis_opt_sets["optimized"].copy()
    eval_targets_sets["optimized"] = eval_targets_opt_sets["optimized"].copy()

    results_baseline, preds_baseline, truths_baseline, paramsk_baseline, stats_X_baseline = (
        utils_inverse.generate_and_eval_emulator_vector(
            emis_dict_train=emis_dict_tier1_JAX,
            targets_dict_train=targets_dict_tier1,
            eval_emis_sets=eval_emis_sets,
            eval_targets_sets=eval_targets_sets,
            output_dim=output_dim,
            lat_coords=lat_coords,
            hidden_sizes=[16],
            K=400,
            lr=1e-1,
            weight_decay=1e-2,
            verbose=True,
        )
    )

    os.makedirs("data/plotting", exist_ok=True)
    with open(BASELINE_PATH, "wb") as f:
        pickle.dump(results_baseline, f)
    print(f"Saved {BASELINE_PATH}")
    with open(BASELINE_DIAGNOSTICS_PATH, "wb") as f:
        pickle.dump({"preds": preds_baseline, "truths": truths_baseline, "lat_coords": lat_coords}, f)
    print(f"Saved {BASELINE_DIAGNOSTICS_PATH}")

    results_opt, preds_opt, truths_opt, params_opt, stats_X_opt = utils_inverse.generate_and_eval_emulator_vector(
        emis_dict_train=emis_dict_opt,
        targets_dict_train=targets_dict_opt,
        eval_emis_sets=eval_emis_opt_sets,
        eval_targets_sets=eval_targets_opt_sets,
        output_dim=output_dim,
        lat_coords=lat_coords,
        hidden_sizes=[16],
        lr=1e-1,
        weight_decay=1e-2,
        K=400,
        verbose=True,
    )

    with open(OPT_PATH, "wb") as f:
        pickle.dump(results_opt, f)
    print(f"Saved {OPT_PATH}")
    with open(OPT_DIAGNOSTICS_PATH, "wb") as f:
        pickle.dump({"preds": preds_opt, "truths": truths_opt, "lat_coords": lat_coords}, f)
    print(f"Saved {OPT_DIAGNOSTICS_PATH}")

    for key in results_baseline.keys():
        print("Eval set: ", key)
        for scen in results_baseline[key].keys():
            old = results_baseline[key][scen]["global"]
            new = results_opt[key][scen]["global"]
            pct_change = 100 * (old - new) / old
            print("\tScen:", scen, "Change:", pct_change)
            print("\t\tRaw base:", old, "Raw opt:", new)


if __name__ == "__main__":
    main()
