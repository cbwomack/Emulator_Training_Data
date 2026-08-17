#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Phase 1, Stage 6c(a): leakage-status inventory (reviewer Point 3).

Documentation only - no new compute. For every quantitative result reported
in the manuscript (Figs 3-7, SI Fig 6), classifies each (training-target,
evaluation-set) combination as one of:

  IN-OBJECTIVE  - the evaluation set (or a superset containing it) was part
                  of what the model's outer-loop optimization directly
                  minimized error on. Not a test of generalization.
  OOD           - the evaluation set was never part of that model's own
                  optimization target. A genuine generalization test.
  BASELINE      - the baseline emulator evaluated on its own training set
                  (ScenarioMIP Priority 1) - explicitly framed in the
                  manuscript as an error *lower bound*, not a generalization
                  claim, so not a leakage concern in the reviewer's sense.
  TRAJECTORY    - an NRMSE-vs-iteration curve showing optimization progress
                  on the model's own target set (Figs 3, 5) - by
                  construction in-objective, and already explicitly framed
                  that way in the manuscript (not a hidden problem).

Every entry below is derived directly from the code (`build_group_emis_dicts`,
`generate_eval_data`, and the `training_paths`/`train_scenarios` lists in each
`load_fig*_data`/`regenerate_fig*_cache` function in utils_inverse.py) and
the manuscript text (main.tex/supplement.tex, both gitignored - not shipped
with the repo) - see the `basis` field on each entry for exactly what was
checked. Two findings are flagged as HEADLINE - see the printed summary and
data/SI_results/leakage_inventory/findings.md for the full writeup.

Usage:
    python 6c_leakage_inventory.py
"""
import csv
from pathlib import Path

OUT_DIR = Path("data/SI_results/leakage_inventory")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Figure 3 / Figure 5: NRMSE-vs-iteration trajectories
# ----------------------------------------------------------------------
# utils_inverse.load_fig3_single_forcing_data reads each agent's `tier1`-group
# checkpoint's own `errors` array; utils_inverse.load_fig5_multi_forcing_data
# does the same for the multi-agent `tier1` checkpoint. Both plot the
# in-objective optimization trajectory on the model's OWN target set -
# explicitly framed in the caption as "Evolution of evaluation loss...when
# reproducing...anomalies for [the training target]", not claimed as a
# generalization test. Baseline dashed line is likewise labeled "the
# baseline emulator's error lower bound (evaluated on its own training
# data)". Correctly framed as-is; included here for completeness.
TRAJECTORY_ENTRIES = [
    {"figure": "Fig 3", "panel": f"({chr(97+i)}) {agent}-only", "train_target": "Tier 1 (own)",
     "eval_target": "Tier 1 (own)", "status": "TRAJECTORY",
     "basis": "load_fig3_single_forcing_data: checkpoints/{a}/inverse_constant_tier1_{a}_only.pkl "
              "'errors' array = the tier1-optimization's own per-step objective loss."}
    for i, agent in enumerate(["CO2", "CH4", "N2O", "Sulfur", "BC"])
] + [
    {"figure": "Fig 3", "panel": f"({chr(97+i)}) {agent}-only", "train_target": "baseline (own)",
     "eval_target": "Tier 1 (own)", "status": "BASELINE",
     "basis": "baseline_{a}_only.pkl['Tier 1']['mean'] - baseline evaluated on its own training set, "
              "explicitly labeled 'error lower bound' in the caption."}
    for i, agent in enumerate(["CO2", "CH4", "N2O", "Sulfur", "BC"])
] + [
    {"figure": "Fig 5", "panel": "(a) All agents", "train_target": "Tier 1 (own)",
     "eval_target": "Tier 1 (own)", "status": "TRAJECTORY",
     "basis": "load_fig5_multi_forcing_data: checkpoints/multi/inverse_constant_tier1_all_agents_subset2.pkl"},
    {"figure": "Fig 5", "panel": "(a) All agents", "train_target": "baseline (own)",
     "eval_target": "Tier 1 (own)", "status": "BASELINE",
     "basis": "checkpoints/multi/baseline_all_agents_subset.pkl['Tier 1']['mean']"},
]

# ----------------------------------------------------------------------
# Figure 4 (CO2-only + multi-agent) and SI Fig 6 (per single-forcing agent)
# ----------------------------------------------------------------------
# Both use utils_plotting.plot_vertical_stacked_bars/plot_grouped_improvement_bars
# with train_scenarios=['Opt. Tier 1','Opt. Tier 2','Opt. DECK','Opt. CS3','Opt. All']
# and test_scenarios=['Tier 1','Tier 2','DECK','CS3'] (load_fig4_data,
# load_SI_extended_results_data). Each 'Opt. X' group's own training-target
# emis_dict comes from build_group_emis_dicts, which for every non-'all' group
# is exactly eval_sets[X] (the SAME data evaluate_optimal_emulator later scores
# 'X' against) - so 'Opt. X' evaluated on test set X is always IN-OBJECTIVE,
# every other cell is OOD. 'Opt. All' is built from eval_sets['All'], which
# generate_eval_data sets to the union of every requested eval_data dict - for
# these two figures that's Tier 1 + Tier 2 + DECK + CS3 (single-agent scripts
# never pass DAMIP/GeoMIP=True, confirmed: DAMIP/GeoMIP are "multi-forcing
# only" per supplement.tex sec:evaluation) - so 'Opt. All' is IN-OBJECTIVE on
# ALL FOUR test columns. This matches the manuscript's own explicit caveat
# ("optimizing over all scenario sets at once inherently introduces
# information leakage", main.tex, immediately after Table 1) - i.e. this
# particular leakage is disclosed, not hidden; included for completeness.
_GRID_TRAIN_TO_TEST = {"Opt. Tier 1": "Tier 1", "Opt. Tier 2": "Tier 2",
                        "Opt. DECK": "DECK", "Opt. CS3": "CS3"}
_TEST_SETS = ["Tier 1", "Tier 2", "DECK", "CS3"]


def _grid_entries(figure, panel):
    rows = []
    for train, own_test in _GRID_TRAIN_TO_TEST.items():
        for test in _TEST_SETS:
            status = "IN-OBJECTIVE" if test == own_test else "OOD"
            rows.append({"figure": figure, "panel": panel, "train_target": train,
                         "eval_target": test, "status": status,
                         "basis": "build_group_emis_dicts: group X's emis_dict == eval_sets[X] "
                                  "(utils_inverse.py:1915-1939); load_fig4_data/"
                                  "load_SI_extended_results_data test_scenarios."})
    for test in _TEST_SETS:
        rows.append({"figure": figure, "panel": panel, "train_target": "Opt. All",
                     "eval_target": test, "status": "IN-OBJECTIVE (disclosed)",
                     "basis": "eval_sets['All'] = union of every requested eval_data dict "
                              "(generate_eval_data, utils_inverse.py:1790-1794) = Tier1+Tier2+DECK+CS3 "
                              "for single-agent scripts. Manuscript explicitly states this leakage "
                              "(main.tex, para after Table 1: 'optimizing over all scenario sets at "
                              "once inherently introduces information leakage')."})
    return rows


GRID_ENTRIES = (
    _grid_entries("Fig 4", "(a) CO2-only")
    + _grid_entries("Fig 4", "(b) Multi-agent")
    + [e for agent in ["CO2", "CH4", "N2O", "Sulfur", "BC"]
       for e in _grid_entries("SI Fig 6", f"{agent}-only")]
)

# ----------------------------------------------------------------------
# Figure 6: individual effects (DAMIP/GeoMIP) - HEADLINE FINDING
# ----------------------------------------------------------------------
# regenerate_fig6_individual_effects_cache (utils_inverse.py:2992-3037):
# train_scenarios_ind_effects = ['Opt. Tier 1', 'Opt. DAMIP', 'Opt. GeoMIP', 'Opt. All'],
# evaluated against eval_sets_ind_effects built with DAMIP=True, GeoMIP=True.
# Critically, the 'Opt. All' checkpoint here comes from
# checkpoints/multi/inverse_constant_all_all_agents.pkl, produced by
# scripts/3b_inverse_all_agents.py's setup call:
#     run_inverse_experiment_setup(..., CS3=True, DAMIP=True, GeoMIP=True)
# - confirmed directly in the script (line 129) - so for the MULTI-AGENT
# case (unlike Fig 4/SI Fig 6's single-agent case above), eval_sets['All']
# DOES include DAMIP and GeoMIP. This means the 'Opt. All' emulator's own
# optimization objective directly included the DAMIP/GeoMIP scenarios -
# yet main.tex explicitly states: "The emulator optimized over the combined
# dataset achieves high accuracy when evaluated on the out-of-distribution
# isolated forcing and climate intervention subsets; emulating DAMIP and
# GeoMIP yields R^2=0.97" (main.tex, Results, para after Fig 5).
# **The R^2=0.97 headline figure is computed entirely from IN-OBJECTIVE
# evaluations, directly contradicting its own "out-of-distribution" label.**
FIG6_ENTRIES = [
    {"figure": "Fig 6", "panel": "(a)-(d)", "train_target": "Opt. Tier 1", "eval_target": "DAMIP (M-GHG, M-aer)",
     "status": "OOD", "basis": "Tier 1 optimization never includes DAMIP scenarios."},
    {"figure": "Fig 6", "panel": "(a)-(d)", "train_target": "Opt. Tier 1", "eval_target": "GeoMIP (G6sulfur)",
     "status": "OOD", "basis": "Tier 1 optimization never includes GeoMIP scenarios."},
    {"figure": "Fig 6", "panel": "(a)-(d)", "train_target": "Opt. DAMIP", "eval_target": "DAMIP (M-GHG, M-aer)",
     "status": "IN-OBJECTIVE", "basis": "DAMIP is itself one of the 7 optimization targets "
              "(supplement.tex sec:evaluation) - directly optimized against."},
    {"figure": "Fig 6", "panel": "(a)-(d)", "train_target": "Opt. DAMIP", "eval_target": "GeoMIP (G6sulfur)",
     "status": "OOD", "basis": "DAMIP optimization never includes the GeoMIP scenario."},
    {"figure": "Fig 6", "panel": "(a)-(d)", "train_target": "Opt. GeoMIP", "eval_target": "GeoMIP (G6sulfur)",
     "status": "IN-OBJECTIVE", "basis": "GeoMIP is itself one of the 7 optimization targets - "
              "directly optimized against."},
    {"figure": "Fig 6", "panel": "(a)-(d)", "train_target": "Opt. GeoMIP", "eval_target": "DAMIP (M-GHG, M-aer)",
     "status": "OOD", "basis": "GeoMIP optimization never includes the DAMIP scenarios."},
    {"figure": "Fig 6", "panel": "(a)-(d)", "train_target": "Opt. All", "eval_target": "DAMIP (M-GHG, M-aer)",
     "status": "IN-OBJECTIVE (HEADLINE - contradicts manuscript's 'out-of-distribution' claim)",
     "basis": "scripts/3b_inverse_all_agents.py:129 run_inverse_experiment_setup(..., DAMIP=True, "
              "GeoMIP=True) -> eval_sets['All'] includes DAMIP. Manuscript's R^2=0.97 sentence "
              "(main.tex, Results) calls this same evaluation 'out-of-distribution'."},
    {"figure": "Fig 6", "panel": "(a)-(d)", "train_target": "Opt. All", "eval_target": "GeoMIP (G6sulfur)",
     "status": "IN-OBJECTIVE (HEADLINE - contradicts manuscript's 'out-of-distribution' claim)",
     "basis": "Same as above, GeoMIP=True in the same setup call."},
]

# ----------------------------------------------------------------------
# Figure 7 / MESM - second, subtler finding (indirect/upstream leakage)
# ----------------------------------------------------------------------
# main.tex Results: "...we perform an independent evaluation using an
# intermediate-complexity climate model...we both verify our optimized
# scenarios are useful...and prevent any information leakage during
# training." This is accurate in the DIRECT sense: MESM's own training loop
# never backpropagates against MESM's Tier 2/DECK/CS3 evaluation sets - no
# code path optimizes MESM against its own eval data.
# BUT: the CO2 emissions trajectory MESM trains on is NOT independent of
# those eval sets - it comes from checkpoints/co2/inverse_{constant,sine}_
# all_co2_only_MESM.pkl (group='all'), i.e. an SCM-side optimization that
# itself targeted eval_sets['All'] = Tier 1 + Tier 2 + DECK + CS3 (CO2-only,
# no DAMIP/GeoMIP - those are multi-forcing only). So MESM's reported
# "extrapolative improvements" on Priority 2/DECK could be partly inherited
# from this upstream SCM-side exposure to those same scenarios, via the
# emissions trajectory itself - a channel the manuscript's leakage claim
# does not address, since it only reasons about MESM's own training loop.
FIG7_ENTRIES = [
    {"figure": "Fig 7", "panel": "MESM summary", "train_target": "MESM baseline (own, Priority 1)",
     "eval_target": "Priority 1 (own)", "status": "BASELINE",
     "basis": "'the baseline emulator inherently retains the highest skill on its own training data "
              "(Priority 1)' (main.tex) - explicitly framed as expected, not a leakage claim."},
    {"figure": "Fig 7", "panel": "MESM summary", "train_target": "MESM-on-SCM-optimized-CO2 (constant/sine)",
     "eval_target": "Priority 2 / DECK (direct, MESM's own loop)", "status": "OOD",
     "basis": "MESM's own training loop never optimizes against its Priority 2/DECK eval sets - "
              "confirmed no such code path exists."},
    {"figure": "Fig 7", "panel": "MESM summary", "train_target": "MESM-on-SCM-optimized-CO2 (constant/sine)",
     "eval_target": "Priority 2 / DECK (indirect, via upstream SCM optimization)",
     "status": "SUBTLE/INDIRECT IN-OBJECTIVE - not addressed by the manuscript's leakage claim",
     "basis": "checkpoints/co2/inverse_{constant,sine}_all_co2_only_MESM.pkl use group='all', i.e. "
              "the SCM-side trajectory MESM trains on was itself optimized against eval_sets['All'] "
              "= Tier1+Tier2+DECK+CS3 (CO2-only). The manuscript's leakage-prevention claim only "
              "reasons about MESM's own training loop, not this upstream channel."},
]

ALL_ENTRIES = TRAJECTORY_ENTRIES + GRID_ENTRIES + FIG6_ENTRIES + FIG7_ENTRIES


def main():
    csv_path = OUT_DIR / "leakage_inventory.csv"
    fieldnames = ["figure", "panel", "train_target", "eval_target", "status", "basis"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ALL_ENTRIES:
            writer.writerow(row)
    print(f"wrote {len(ALL_ENTRIES)} entries -> {csv_path}")

    by_status = {}
    for e in ALL_ENTRIES:
        by_status.setdefault(e["status"], 0)
        by_status[e["status"]] += 1
    print("\nCounts by status:")
    for status, n in sorted(by_status.items()):
        print(f"  {status}: {n}")

    print("\n=== HEADLINE FINDINGS ===")
    print("1. Fig 6's reported R^2=0.97 'out-of-distribution' result (Opt. All evaluated on")
    print("   DAMIP/GeoMIP) is computed entirely from IN-OBJECTIVE evaluations - the 'All'")
    print("   optimization target directly includes DAMIP and GeoMIP for the multi-agent case")
    print("   (scripts/3b_inverse_all_agents.py:129), contradicting the manuscript's own")
    print("   'out-of-distribution' label for this specific number.")
    print("2. Fig 7/MESM's 'prevent any information leakage during training' claim is accurate")
    print("   for MESM's own training loop, but doesn't address the upstream SCM-side")
    print("   optimization (group='all') that generated MESM's training trajectory - a subtler,")
    print("   second-order leakage channel not addressed by the manuscript's framing.")


if __name__ == "__main__":
    main()
