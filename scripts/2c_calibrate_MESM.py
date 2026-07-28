#!/usr/bin/env python
"""
Companion script for 2c_calibrate_MESM.ipynb: calibrates the JAX SCM against
MESM ensemble output in two phases - climate sensitivity (conc -> temp, 1000
steps) and carbon cycle (emis -> conc, 200 steps) - checkpointing theta to
data/JAX_calibration/calib_MESM_*.pkl (via utils_FaIR_JAX.calibrate_
climate_sensitivity / calibrate_carbon_cycle). Runs standalone (no
notebook/kernel needed) so it can be scheduled; the notebook itself only
loads the resulting theta and re-plots.

Fixes a portability bug along the way: the original notebook hardcoded
absolute paths (/Users/chriswomack/Documents/PhD Project 2/data/MESM/...)
to the MESM ensemble text files, which only worked on the original author's
machine. Both this script and the trimmed notebook now use paths.DATA_DIR.

Usage:
    python 2c_calibrate_MESM.py
    python 2c_calibrate_MESM.py --phase climate    # just Phase 1
    python 2c_calibrate_MESM.py --phase carbon      # just Phase 2
"""
import glob
import os
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

CLIMATE_FILEPATH = "data/JAX_calibration/calib_MESM_CO2_only.pkl"
CARBON_FILEPATH = "data/JAX_calibration/calib_MESM_emis_to_conc.pkl"


def _calculate_ensemble_average(file_pattern: str) -> np.ndarray | None:
    file_list = glob.glob(file_pattern)
    if not file_list:
        print("No files found matching the pattern.")
        return None
    all_data = [np.loadtxt(f, usecols=(1,)) for f in file_list]
    return np.mean(np.stack(all_data), axis=0)


def run_climate_sensitivity():
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

    theta0 = utils_FaIR_JAX.make_theta0(mode="FaIR")
    utils_FaIR_JAX.calibrate_climate_sensitivity(
        CLIMATE_FILEPATH, emis_dict, conc_dict, target_dict, theta0, dt=0.1, n_steps=1000, learning_rate=1e-2
    )
    print(f"Saved {CLIMATE_FILEPATH}")


def run_carbon_cycle():
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

    theta0 = utils_FaIR_JAX.make_theta0(mode="FaIR")
    utils_FaIR_JAX.calibrate_carbon_cycle(
        CARBON_FILEPATH, emis_dict, target_conc_dict, theta0, dt=0.1, n_steps=200, learning_rate=1e-2, mode="FaIR"
    )
    print(f"Saved {CARBON_FILEPATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["climate", "carbon"], default=None,
                         help="Run only this phase (default: run both)")
    args = parser.parse_args()

    if args.phase in (None, "climate"):
        run_climate_sensitivity()
    if args.phase in (None, "carbon"):
        run_carbon_cycle()


if __name__ == "__main__":
    main()
