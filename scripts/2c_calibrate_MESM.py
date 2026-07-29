#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5 and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Companion script for 2c_calibrate_MESM.ipynb: calibrates the JAX SCM against
MESM ensemble output in two *sequential* phases - carbon cycle (emis -> conc,
200 steps) first, then climate sensitivity (conc -> temp, 1000 steps) chained
from the carbon-cycle result - checkpointing theta to
data/JAX_calibration/calib_MESM_*.pkl (via utils_FaIR_JAX.calibrate_
carbon_cycle / calibrate_climate_sensitivity). Runs standalone (no
notebook/kernel needed) so it can be scheduled; the notebook itself only
loads the resulting theta and re-plots.

Sequential, not independent (fixed per user feedback): carbon cycle runs
first from the bare FaIR-derived theta0; climate sensitivity then continues
optimizing *that same theta* (only its own Climate-masked d/q entries move -
the Carbon-masked fields calibrate_carbon_cycle already tuned stay frozen),
so the final theta genuinely has both sub-systems calibrated together rather
than two independently-masked thetas that would need to be manually merged
for anything downstream (e.g. the notebook's "Phase 3: verification" plot)
to reflect the real MESM-calibrated model. base_params is now an explicit,
user-settable argument on both utils_FaIR_JAX functions (default FAIR_PARAMS,
their typical use) - this script passes MESM_PARAMS explicitly since this is
specifically the MESM calibration.

Also fixes a portability bug: the original notebook hardcoded absolute paths
(/Users/chriswomack/Documents/PhD Project 2/data/MESM/...) to the MESM
ensemble text files, which only worked on the original author's machine.
Both this script and the trimmed notebook now use paths.DATA_DIR.

Usage:
    python 2c_calibrate_MESM.py
    python 2c_calibrate_MESM.py --phase carbon    # just the carbon-cycle step
    python 2c_calibrate_MESM.py --phase climate   # just the climate step, chained
                                                   # from the saved carbon-cycle
                                                   # theta if it exists, else from
                                                   # a bare theta0 (with a warning)
"""
import glob
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import jax.numpy as jnp
import numpy as np

import utils_FaIR_JAX
from paths import DATA_DIR
from utils_FaIR_JAX import MESM_PARAMS

CLIMATE_FILEPATH = "data/JAX_calibration/calib_MESM_CO2_only.pkl"
CARBON_FILEPATH = "data/JAX_calibration/calib_MESM_emis_to_conc.pkl"


def _calculate_ensemble_average(file_pattern: str) -> np.ndarray | None:
    """Mean over every MESM ensemble-member file matching `file_pattern` (column 1, whitespace-delimited ascii)."""
    file_list = glob.glob(file_pattern)
    if not file_list:
        print("No files found matching the pattern.")
        return None
    all_data = [np.loadtxt(f, usecols=(1,)) for f in file_list]
    return np.mean(np.stack(all_data), axis=0)


def run_carbon_cycle(theta0: jnp.ndarray) -> jnp.ndarray:
    """Phase 1: calibrate the carbon-cycle theta fields against MESM's 1pctCO2 emissions->concentration response."""
    emis_path = str(DATA_DIR / "MESM" / "emis_driven")
    emis_1pct_path = f"{emis_path}/1PRCO2/carbemiss.txt"

    emis_1pct = np.loadtxt(emis_1pct_path, usecols=(2,), skiprows=2)
    emis_mat_1pct = jnp.zeros((5, len(emis_1pct)))
    emis_mat_1pct = emis_mat_1pct.at[0, :].set(emis_1pct)
    emis_dict = {"1pctCO2": emis_mat_1pct}

    base_co2 = 286.4
    co2_1pct = base_co2 * (1.01 ** np.arange(len(emis_1pct)))
    conc_mat_1pct = jnp.zeros((3, len(co2_1pct)))
    conc_mat_1pct = conc_mat_1pct.at[0, :].set(co2_1pct)
    conc_mat_1pct = conc_mat_1pct.at[1, :].set(720.0)
    conc_mat_1pct = conc_mat_1pct.at[2, :].set(270.0)
    target_conc_dict = {"1pctCO2": conc_mat_1pct}

    theta_opt, _ = utils_FaIR_JAX.calibrate_carbon_cycle(
        CARBON_FILEPATH, emis_dict, target_conc_dict, theta0, dt=0.1, n_steps=200, learning_rate=1e-2,
        mode="FaIR", base_params=MESM_PARAMS,
    )
    print(f"Saved {CARBON_FILEPATH}")
    return theta_opt


def run_climate_sensitivity(theta0: jnp.ndarray) -> jnp.ndarray:
    """Phase 2: calibrate the climate-sensitivity theta fields (d, q), continuing from Phase 1's theta0."""
    conc_path = str(DATA_DIR / "MESM" / "conc_driven")
    conc_1pct_path = f"{conc_path}/1PRCO2/"

    avg_1pct = _calculate_ensemble_average(conc_1pct_path + "dt2m_ascii_*.txt")
    target_dict = {"1pctCO2": avg_1pct}

    n_years_1pct = len(avg_1pct)
    base_co2 = 286.4
    co2_1pct = base_co2 * (1.01 ** np.arange(n_years_1pct))
    conc_mat_1pct = jnp.zeros((3, len(co2_1pct)))
    conc_mat_1pct = conc_mat_1pct.at[0, :].set(co2_1pct)
    conc_mat_1pct = conc_mat_1pct.at[1, :].set(720.0)
    conc_mat_1pct = conc_mat_1pct.at[2, :].set(270.0)

    conc_dict = {"1pctCO2": conc_mat_1pct}
    emis_dict = {"1pctCO2": jnp.zeros((5, len(co2_1pct)))}

    theta_opt, _ = utils_FaIR_JAX.calibrate_climate_sensitivity(
        CLIMATE_FILEPATH, emis_dict, conc_dict, target_dict, theta0, dt=0.1, n_steps=1000, learning_rate=1e-2,
        base_params=MESM_PARAMS,
    )
    print(f"Saved {CLIMATE_FILEPATH}")
    return theta_opt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["climate", "carbon"], default=None,
                         help="Run only this phase (default: run both, sequentially)")
    args = parser.parse_args()

    theta0 = utils_FaIR_JAX.make_theta0(mode="FaIR")

    if args.phase is None:
        theta_carbon = run_carbon_cycle(theta0)
        run_climate_sensitivity(theta_carbon)
    elif args.phase == "carbon":
        run_carbon_cycle(theta0)
    elif args.phase == "climate":
        if os.path.isfile(CARBON_FILEPATH):
            with open(CARBON_FILEPATH, "rb") as f:
                theta0 = pickle.load(f)
        else:
            print(f"No saved carbon-cycle theta at {CARBON_FILEPATH} - "
                  "starting climate calibration from a bare theta0 instead.")
        run_climate_sensitivity(theta0)


if __name__ == "__main__":
    main()
