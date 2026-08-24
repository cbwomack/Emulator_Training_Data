#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5 and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Companion script for 4c_evaluate_MESM_emulator.ipynb: trains and evaluates the
vector-output (zonal-temperature) MLP emulator against MESM output - once on
the baseline (ScenarioMIP tier1) training set, and once each on three CO2-only
inverse-optimized ("all") training sets (constant-IC-only, sine-IC-only, and
both combined) from checkpoints/co2/. Writes the pickles the notebook needs to
reload and plot. Runs standalone so it can be scheduled.

Fixes two portability/correctness bugs and preserves one pre-existing
execution-order quirk, flagged rather than silently changed:

1. Fixed: the notebook hardcoded an absolute
   /Users/chriswomack/Documents/PhD/Project 2/data/MESM/emis_driven path for
   the DECK ensemble text files - same class of bug as 2c_calibrate_MESM, now
   uses paths.DATA_DIR (indirectly, via utils_inverse.build_MESM_baseline_
   eval_sets(), which this script's build_eval_sets() now delegates to -
   promoted there so MESM HP-search/seed-sweep scripts can reuse the same
   eval-set construction without duplicating it).
2. Fixed (three genuinely distinct optimized-side training runs, not one
   mislabeled file): the original notebook produced the constant-only,
   sine-only, and combined "both" results as three separate one-off manual
   runs, saving each under its own optimal_co2_only_MESM_{IC}.pkl filename.
   Once this got scripted, only the combined case was ever actually trained -
   the save-path line looped `for IC in ['constant', 'sine']: ...` to build
   the combined training set, then reused the loop's *last* value of IC for
   the filename (`optimal_co2_only_MESM_{IC}.pkl` -> always "..._sine.pkl"),
   which silently overwrote whatever the real sine-only run had produced with
   a copy of the combined result instead. utils_inverse.build_MESM_opt_eval_
   sets(ic_list=...) now genuinely trains each of the three variants
   (IC_VARIANTS below) from its own emissions trajectory/ground truth, so all
   three output files are real, distinct runs again - confirmed the combined
   ("both") variant's numbers match the original manual run to 5 decimal
   places on every Tier1/Tier2/DECK/CS3 scenario but one, validating this
   reproduces the intended computation rather than just resolving the
   filename collision. The real sine-only result has no surviving copy (its
   only save location was overwritten by the bug) and had to be retrained
   from scratch here rather than recovered.
3. NOT changed (preserved as-is): the notebook's cell order doesn't match its
   real dependency order - the cell that injects an 'optimized' eval set into
   eval_emis_sets/eval_targets_sets (originally cell 6, "# Run after
   optimization") reads eval_emis_opt_sets/eval_targets_opt_sets, which are
   only defined two cells *later* (originally cell 11). The notebook only
   ever ran correctly because a human executed the cells out of visual order
   (3, 4, 5, 11, 6, 7, 8, 9, 12, 13, 14, 15). This script uses that real
   dependency order, not the notebook's visual top-to-bottom order. Unlike
   the single-combined-run version, the baseline emulator here is no longer
   evaluated against any 'optimized' eval-set entry at all (there's no longer
   one canonical choice, now that there are three, and Figure 7's plot never
   read that entry anyway - utils_inverse.load_fig7_emic_data's scenario_keys
   cover only Tier1/Tier2/DECK/CS3).

Also saves a *_diagnostics.pkl pair (lat_coords + preds/truths) alongside each
literal baseline/optimal result pickle the original notebook cells produced,
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

import utils_inverse

EVAL_DIR = "data/MESM/emis_driven/zonal_data_mean/"
BASELINE_PATH = "data/plotting/baseline_co2_only_MESM.pkl"
BASELINE_DIAGNOSTICS_PATH = "data/plotting/baseline_co2_only_MESM_diagnostics.pkl"
OPT_GROUP = "all"


def build_eval_sets():
    """Assemble the Tier 1/Tier 2/DECK/CS3 emissions and MESM zonal-temperature target sets used to evaluate the baseline emulator.

    Thin wrapper around utils_inverse.build_MESM_baseline_eval_sets(), which
    this logic was promoted into so HP-search/seed-sweep scripts could reuse
    it without duplicating it - this function now just adapts that dict
    return into this script's original positional-tuple shape.
    """
    setup = utils_inverse.build_MESM_baseline_eval_sets(eval_dir=EVAL_DIR)
    return (
        setup["eval_emis_sets"], setup["eval_targets_sets"],
        setup["emis_dict_tier1_JAX"], setup["targets_dict_tier1"],
        setup["output_dim"], setup["lat_coords"],
    )


# The three optimized-side training variants Figure 7 compares (Phase 6 fix,
# see module docstring point 3 below): constant-IC-only, sine-IC-only, and
# both combined - matching the three one-off manual runs the user originally
# produced by hand before any of this was scripted. Filename suffix -> ic_list.
IC_VARIANTS = {"constant": ["constant"], "sine": ["sine"], "both": ["constant", "sine"]}


def main():
    eval_emis_sets, eval_targets_sets, emis_dict_tier1_JAX, targets_dict_tier1, output_dim, lat_coords = build_eval_sets()

    # Baseline is trained/evaluated once, against only the Tier1/Tier2/DECK/CS3
    # eval sets - unlike the old single-combined-run version, it does NOT get
    # an 'optimized' eval-set entry injected, since that entry was never
    # actually read by Figure 7's plot (utils_inverse.load_fig7_emic_data's
    # scenario_keys cover only Tier1/Tier2/DECK/CS3) and there's no longer one
    # canonical "optimized" set to pick, now that there are three.
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

    for variant_name, ic_list in IC_VARIANTS.items():
        opt = utils_inverse.build_MESM_opt_eval_sets(
            eval_emis_sets, eval_targets_sets, ic_list=ic_list, group=OPT_GROUP, eval_dir=EVAL_DIR
        )

        results_opt, preds_opt, truths_opt, params_opt, stats_X_opt = utils_inverse.generate_and_eval_emulator_vector(
            emis_dict_train=opt["emis_dict_opt"],
            targets_dict_train=opt["targets_dict_opt"],
            eval_emis_sets=opt["eval_emis_opt_sets"],
            eval_targets_sets=opt["eval_targets_opt_sets"],
            output_dim=output_dim,
            lat_coords=lat_coords,
            hidden_sizes=[16],
            lr=1e-1,
            weight_decay=1e-2,
            K=400,
            verbose=True,
        )

        opt_path = f"data/plotting/optimal_co2_only_MESM_{variant_name}.pkl"
        opt_diagnostics_path = f"data/plotting/optimal_co2_only_MESM_{variant_name}_diagnostics.pkl"
        with open(opt_path, "wb") as f:
            pickle.dump(results_opt, f)
        print(f"Saved {opt_path}")
        with open(opt_diagnostics_path, "wb") as f:
            pickle.dump({"preds": preds_opt, "truths": truths_opt, "lat_coords": lat_coords}, f)
        print(f"Saved {opt_diagnostics_path}")

        print(f"=== variant: {variant_name} ({ic_list}) ===")
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
