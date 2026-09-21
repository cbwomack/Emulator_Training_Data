#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5, Claude Sonnet 5, and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Companion script for 02b_optimize_GeoMIP.ipynb: solves for the sulfur
injection profile that cools the H-ext scenario to match M-ext (a G6-sulfur
GeoMIP analogue).

Writes two files:
  - data/GeoMIP/emis_G6sulfur.pickle: raw GeoMIP emissions trajectories.
  - data/GeoMIP/emis_G6sulfur_diagnostics.pickle: plotting diagnoistics
    used only by the companion notebook.

Usage:
    python 02b_optimize_GeoMIP.py
"""
import os
import pickle
import sys
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import jax.numpy as jnp
import numpy as np

import utils_FaIR_JAX

AGENTS = ['CO2', 'CH4', 'N2O', 'Sulfur', 'BC']
OUTPUT_PATH = 'data/GeoMIP/emis_G6sulfur.pickle'
DIAGNOSTICS_PATH = 'data/GeoMIP/emis_G6sulfur_diagnostics.pickle'


def main():
    _, emis_dict_calib_JAX, _, _ = utils_FaIR_JAX.generate_calib_data(AGENTS)
    emis_H = jnp.concatenate([emis_dict_calib_JAX['historical'], emis_dict_calib_JAX['H-ext']], axis=1)
    emis_M = jnp.concatenate([emis_dict_calib_JAX['historical'], emis_dict_calib_JAX['M-ext']], axis=1)

    T = emis_H.shape[1]
    years = jnp.arange(T, dtype=jnp.float32)
    delT_M = utils_FaIR_JAX.simulate_temp(years, emis_M)['GMST']

    opt_sulfur, final_T_pred = utils_FaIR_JAX.solve_sulfur_inverse(
        emis_H, delT_M, years, learning_rate=0.8, n_steps=4000, reg_weight=1e-4
    )

    emis_H_G6 = np.concatenate([emis_dict_calib_JAX['historical'], emis_dict_calib_JAX['H-ext']], axis=1).copy()
    emis_H_G6[3, :] = opt_sulfur.copy()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(emis_H_G6, f)

    with open(DIAGNOSTICS_PATH, "wb") as f:
        pickle.dump({
            "opt_sulfur": opt_sulfur,
            "final_T_pred": final_T_pred,
            "emis_M": emis_M,
            "delT_M": delT_M,
        }, f)

    print(f"Saved {OUTPUT_PATH} and {DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()
