#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage 6d - Point 4 (SCM->MESM fidelity): runs the recalibrated SCM
(mode='MESM') forward on Tier1/Tier2/DECK/CS3 CO2-only emissions and compares
its GMST output against the real MESM zonal-temperature output already on
disk (data/MESM/emis_driven/zonal_data_mean/), area-weighted down to a global
mean. Reports NRMSE and R^2 per scenario, per scenario set, and overall.

**Time-axis alignment (the open question this stage was blocked on):**
MESM's raw zonal_data_mean files (e.g. 'Tier 1/H-ext_mean.pkl', shape
(639, 46)) run on MESM's own internal year numbering and don't share the
SCM's own calendar-year emissions arrays' length (H-ext: T=477) at face
value. The resolution turned out to already exist in this codebase, just
never applied to a direct SCM-vs-MESM comparison before: utils_inverse.
build_dataset_vector_targets (used by the existing MLP-vs-MESM comparison in
4c_evaluate_MESM_emulator.py) crops the first `target_crop_future=162` years
off every non-historical MESM target array before use - 639 - 162 = 477,
exactly matching H-ext's true future-only length, confirming MESM's raw
files carry a 162-year historical prefix ahead of where the named future
scenario actually starts. This script reuses that exact, already-in-
production cropping function rather than re-deriving the alignment, and
pairs it with utils_inverse.build_dataset_from_runfair_dict's own historical-
context handling (simulate_targets_gmst: concatenate historical + current,
run once, return only the current-scenario tail) on the SCM side - the two
were already mutually consistent, just never combined for this comparison.

Scope: CO2-only (agents=['CO2'], mode='MESM'), matching the only existing
MESM-calibrated checkpoints (checkpoints/co2/*_MESM.pkl) and the existing
MLP-vs-MESM comparison's own scope (4c_evaluate_MESM_emulator.py:86). Per
user direction, uses those existing (pre-Phase-0, unregenerated) MESM
checkpoints/calibration as-is rather than waiting on a regeneration pass.
DAMIP/GeoMIP excluded (no MESM ground truth exists for them), matching
REVISIONS.md's stated scope for this stage.

Usage:
    python 6d_scm_mesm_fidelity.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import numpy as np

import utils_inverse
from paths import DATA_DIR

EVAL_DIR = "data/MESM/emis_driven/zonal_data_mean/"
AGENTS = ["CO2"]
MODE = "MESM"
OUT_DIR = Path("data/SI_results/scm_mesm_fidelity")

SCENARIOS_EVAL = {
    "Tier 1": ["historical", "H-ext", "L", "M", "ML", "VLHO", "VLLO-ext"],
    "Tier 2": ["H-ext-OS", "M-ext", "ML-ext", "L-ext", "VLHO-ext"],
    "DECK": ["1pctCO2", "2xCO2"],
    "CS3": ["AA", "CT", "historical"],
}


def build_eval_sets():
    """Same construction as scripts/4c_evaluate_MESM_emulator.py:build_eval_sets
    (CO2-only emissions + MESM zonal targets for Tier1/Tier2/DECK/CS3),
    duplicated rather than imported since that module's main() has side
    effects (trains an MLP) this script doesn't want."""
    eval_emis_sets, emis_dict_tier1_JAX, emis_dict_tier2_JAX, emis_dict_CS3_JAX, emis_dict_all_JAX = (
        utils_inverse.generate_eval_data(AGENTS, DECK=False, CS3=True, DAMIP=False, GeoMIP=False)
    )
    eval_targets_sets, *_rest, lat_coords = utils_inverse.generate_target_data(SCENARIOS_EVAL, data_dir=EVAL_DIR)

    emis_path = str(DATA_DIR / "MESM" / "emis_driven")
    emis_1pct = np.loadtxt(f"{emis_path}/1PRCO2/carbemiss.txt", usecols=(2,), skiprows=2)
    emis_mat_1pct = np.zeros((5, len(emis_1pct)))
    emis_mat_1pct[0, :] = emis_1pct

    emis_2xCO2 = np.loadtxt(f"{emis_path}/2xCO2/implco2emiss.3100.25.txt", usecols=(2,))
    emis_mat_2xCO2 = np.zeros((5, len(emis_2xCO2)))
    emis_mat_2xCO2[0, :] = emis_2xCO2

    eval_emis_sets["DECK"] = {"1pctCO2": emis_mat_1pct, "2xCO2": emis_mat_2xCO2}

    return eval_emis_sets, eval_targets_sets, lat_coords


def scm_gmst_by_scenario(emis_dict):
    """{scenario: GMST array}, SCM-simulated (mode='MESM'), historical-context
    handling identical to the existing baseline-training path
    (utils_inverse.build_dataset_from_runfair_dict -> simulate_targets_gmst)."""
    raw = utils_inverse.build_dataset_from_runfair_dict(emis_dict, agents=AGENTS, mode=MODE)
    return {scen: np.asarray(y) for (_X, y, scen) in raw}


def mesm_global_truth_by_scenario(emis_dict, targets_dict, lat_coords):
    """{scenario: area-weighted global-mean MESM truth array}, using the
    existing, already-in-production crop/align logic
    (utils_inverse.build_dataset_vector_targets)."""
    weights = np.cos(np.deg2rad(lat_coords))
    weights = np.maximum(weights, 1e-6)
    weights = weights / np.sum(weights)

    scens = list(emis_dict.keys())
    raw = utils_inverse.build_dataset_vector_targets(emis_dict, targets_dict, scens, agents=AGENTS)
    out = {}
    for (_X, y_target, scen) in raw:
        y_target = np.asarray(y_target)
        out[scen] = np.sum(y_target * weights[None, :], axis=1)
    return out


def nrmse_r2(pred, truth):
    n = min(len(pred), len(truth))
    pred, truth = pred[:n], truth[:n]
    rmse = np.sqrt(np.mean((pred - truth) ** 2))
    nrmse = rmse / (np.max(np.abs(truth)) + 1e-8)
    ss_res = np.sum((truth - pred) ** 2)
    ss_tot = np.sum((truth - np.mean(truth)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return float(nrmse), float(r2), n


def main():
    eval_emis_sets, eval_targets_sets, lat_coords = build_eval_sets()

    results = {}
    for set_name, emis_d in eval_emis_sets.items():
        targets_d = eval_targets_sets.get(set_name)
        if targets_d is None:
            print(f"[{set_name}] no MESM targets found, skipping")
            continue

        scm_gmst = scm_gmst_by_scenario(emis_d)
        mesm_truth = mesm_global_truth_by_scenario(emis_d, targets_d, lat_coords)

        set_results = {}
        for scen in emis_d:
            if set_name not in ("Tier 1", "All") and scen == "historical":
                continue  # matches evaluate_emulator_vector_over_multiple_tests's own convention
            if scen not in scm_gmst or scen not in mesm_truth:
                print(f"[{set_name}/{scen}] missing SCM or MESM data, skipping")
                continue
            nrmse, r2, n = nrmse_r2(scm_gmst[scen], mesm_truth[scen])
            set_results[scen] = {"nrmse": nrmse, "r2": r2, "n_years": n}
            print(f"[{set_name}/{scen}] NRMSE={nrmse:.4f} R2={r2:.4f} (n={n} years)")

        if set_results:
            set_results["mean"] = {
                "nrmse": float(np.mean([v["nrmse"] for v in set_results.values()])),
                "r2": float(np.mean([v["r2"] for v in set_results.values()])),
            }
            print(f"[{set_name}] mean NRMSE={set_results['mean']['nrmse']:.4f} "
                  f"mean R2={set_results['mean']['r2']:.4f}")
        results[set_name] = set_results

    all_nrmse = [v["nrmse"] for s in results.values() for k, v in s.items() if k != "mean"]
    all_r2 = [v["r2"] for s in results.values() for k, v in s.items() if k != "mean"]
    overall = {"nrmse": float(np.mean(all_nrmse)), "r2": float(np.mean(all_r2))} if all_nrmse else None
    if overall:
        print(f"[overall] mean NRMSE={overall['nrmse']:.4f} mean R2={overall['r2']:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "scm_mesm_fidelity_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"per_scenario_set": results, "overall": overall}, f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
