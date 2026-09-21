#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5, Claude Sonnet 5, and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Companion script for 02a_calibrate_FaIR_JAX.ipynb: calibrates the JAX SCM
against FaIR outputs for each forcing-agent group (CO2, CH4, N2O, Sulfur+BC),
saving params to data/JAX_calibration/calib_*.pkl.

Usage:
    python 02a_calibrate_FaIR_JAX.py                # run every target below, in order
    python 02a_calibrate_FaIR_JAX.py --target CH4   # run just one
"""
import os
import sys
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import argparse

import utils_FaIR_JAX

# Calibration hyperparameters (agents, dt, n_steps, target, learning_rate, filepath) per group,
# CH4 needs a smaller learning rate or it diverges.
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
