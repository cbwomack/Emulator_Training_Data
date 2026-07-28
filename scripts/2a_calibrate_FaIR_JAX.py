#!/usr/bin/env python
"""
Companion script for 2a_calibrate_FaIR_JAX.ipynb: calibrates the JAX SCM
against FaIR outputs for each forcing-agent group (CO2, CH4, N2O, Sulfur+BC),
checkpointing theta to data/JAX_calibration/calib_*.pkl every 10 steps (via
utils_FaIR_JAX.calibrate_inverse). Runs standalone (no notebook/kernel
needed) so it can be scheduled; the notebook itself only loads the resulting
theta and re-plots FaIR-vs-JAX.

Usage:
    python 2a_calibrate_FaIR_JAX.py                # run every target below, in order
    python 2a_calibrate_FaIR_JAX.py --target CH4   # run just one
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import utils_FaIR_JAX

# Exact (agents, dt, n_steps, target, learning_rate, filepath) per group,
# transcribed from 2a_calibrate_FaIR_JAX.ipynb. learning_rate=None means the
# notebook didn't override calibrate_inverse's default (1e-2) - CH4 needs a
# smaller rate (see calibrate_inverse's docstring) or it diverges.
TARGETS = {
    "CO2": dict(agents=["CO2"], dt=0.1, n_steps=500, target="CO2", learning_rate=None,
                filepath="data/JAX_calibration/calib_CO2_only_new.pkl"),
    "CH4": dict(agents=["CH4"], dt=0.1, n_steps=150, target="CH4", learning_rate=1e-4,
                filepath="data/JAX_calibration/calib_CH4_only_new.pkl"),
    "N2O": dict(agents=["N2O"], dt=0.1, n_steps=150, target="N2O", learning_rate=None,
                filepath="data/JAX_calibration/calib_N2O_only_new.pkl"),
    "Aer": dict(agents=["Sulfur", "BC"], dt=0.1, n_steps=150, target="Aer", learning_rate=None,
                filepath="data/JAX_calibration/calib_Sulfur_BC.pkl"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=list(TARGETS), default=None,
                         help="Run only this target (default: run all, in order)")
    args = parser.parse_args()

    to_run = [args.target] if args.target else list(TARGETS)
    for name in to_run:
        cfg = TARGETS[name]
        print(f"Calibrating {name!r}...")
        _, emis_dict_calib_JAX, delT_dict_calib_FaIR, _ = utils_FaIR_JAX.generate_calib_data(cfg["agents"], mode="FaIR")
        theta0 = utils_FaIR_JAX.make_theta0(mode="FaIR")
        kwargs = dict(target=cfg["target"])
        if cfg["learning_rate"] is not None:
            kwargs["learning_rate"] = cfg["learning_rate"]
        utils_FaIR_JAX.calibrate_inverse(
            cfg["filepath"], emis_dict_calib_JAX, delT_dict_calib_FaIR, theta0, cfg["dt"], cfg["n_steps"], **kwargs
        )


if __name__ == "__main__":
    main()
