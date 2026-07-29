#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5 and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Companion script for supplementary_notebooks/gen_MESM_data.ipynb: writes CO2
emissions input files (MESM's plain-text ascii format) for every scenario the
external MESM model needs to be driven with, plus the optimized ("inverse")
CO2 trajectory found by the sine-initial-condition 'all' inverse experiment.
This is a pure file-writing utility (no analysis output kept in memory beyond
what's written to disk), so the whole notebook's generation logic migrates
here; the notebook keeps only a diagnostic reload-and-replot of the written
files. Runs standalone so it can be scheduled.

Usage:
    python gen_MESM_data.py
"""
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import utils_FaIR_JAX

AGENTS = ["CO2"]
NEEDS_HIST = [
    "H-ext", "M", "ML", "L", "VLLO-ext", "VLHO", "H-ext-OS", "M-ext",
    "ML-ext", "L-ext", "VLHO-ext", "AA", "CT",
]
OUT_DIR = Path("data/MESM/emis_driven/MESM_inputs")


def write_emissions_file(co2_array: np.ndarray, filepath: str, hist: np.ndarray | None = None) -> None:
    """Write one scenario's CO2 trajectory to MESM's plain-text ascii input format."""
    if hist is not None:
        co2_array = np.concatenate([hist, co2_array])

    with open(filepath, "w") as f:
        # Required MESM headers
        f.write("UNITS:  GtC/yr  GtCO2/yr\n")
        f.write("YEARS   FossilCO2\n")

        # Iterate through the array with a 1-based index for the year
        for year, co2 in enumerate(co2_array, 1):
            # Convert GtCO2 to GtC (ratio of atomic mass 12/44)
            carbon = co2 * (12.0 / 44.0)

            # Write year, calculated Carbon, and original CO2 with source-matched spacing
            f.write(f"{year:4d}{carbon:18.6f}{co2:18.6f}\n")


def write_scenario_inputs() -> None:
    """Write every ScenarioMIP/DECK/CS3 scenario's CO2 emissions to a MESM input file, prepending historical data where MESM needs it."""
    emis_dict_tier1_JAX, emis_dict_tier2_JAX, emis_dict_DECK_JAX, emis_dict_CS3_JAX = utils_FaIR_JAX.generate_JAX_data(
        AGENTS, CS3=True
    )
    emis_dict_all_JAX = emis_dict_tier1_JAX | emis_dict_tier2_JAX | emis_dict_DECK_JAX | emis_dict_CS3_JAX

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # MESM files require historical data padding, amount of padding required
    # varies by scenario
    for scen in emis_dict_all_JAX.keys():
        filepath = OUT_DIR / f"{scen}_emis.txt"
        co2_array = emis_dict_all_JAX[scen][0, :]
        if scen in NEEDS_HIST:
            if scen in ["AA", "CT"]:
                hist = emis_dict_CS3_JAX["historical"][0, 111:]
            else:
                hist = emis_dict_tier1_JAX["historical"][0, 111:]
        else:
            hist = None
            if scen == "historical":
                co2_array = co2_array[111:].copy()

        write_emissions_file(co2_array, filepath, hist=hist)
        print(f"Saved {filepath}")


def write_optimized_inputs() -> None:
    """Write the sine-IC inverse-optimized CO2 trajectory to a MESM input file, for the 'optimized' MESM-driving experiment."""
    groups = ["all"]
    for group in groups:
        opt_path = f"checkpoints/co2/inverse_sine_{group}_co2_only_MESM.pkl"
        filepath = OUT_DIR / f"opt_{group}_sine_emis.txt"
        with open(opt_path, "rb") as f:
            res = pickle.load(f)
        co2_array = res["U_traj"][-1]["CO2"]
        write_emissions_file(co2_array, filepath, hist=None)
        print(f"Saved {filepath}")


def main():
    write_scenario_inputs()
    write_optimized_inputs()


if __name__ == "__main__":
    main()
