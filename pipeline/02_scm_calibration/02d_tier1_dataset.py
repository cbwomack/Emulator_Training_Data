#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Tier 1 dataset for the adopted three-stage MESM calibration.

02c_calibrate_MESM.ipynb fits the SCM's thermal timescales against real MESM
emissions-driven output.
"""
import importlib.util
import os

from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import jax.numpy as jnp
import numpy as np

import utils_inverse

_spec = importlib.util.spec_from_file_location(
    "scm_mesm_fidelity", PROJECT_ROOT / "pipeline" / "07_results" / "07e_scm_mesm_fidelity.py"
)
fidelity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fidelity)

MESM_ZONAL_DIR = "data/MESM/emis_driven/zonal_data_mean/"
AGENTS = ("CO2",)

TIER1_FUTURE_SCENARIOS = ["H-ext", "M", "ML", "L", "VLHO", "VLLO-ext"]
HISTORICAL_NAME = "historical"
TIER1_SCORED_SCENARIOS = TIER1_FUTURE_SCENARIOS + [HISTORICAL_NAME]

DECK_TRAIN_SCENARIOS = ["1pctCO2", "2xCO2"]
ALL_TRAIN_SCENARIOS = TIER1_SCORED_SCENARIOS + DECK_TRAIN_SCENARIOS

EMIS_OFFSET_HIST = 111


def build_tier1_dataset() -> tuple[dict, dict]:
    """CO2-only emissions and MESM's ensemble-mean global temperature.

    Returns (emissions, target GMST), keyed by scenario, covering the seven
    Tier 1 scenarios and DECK's 1pctCO2/2xCO2 (ALL_TRAIN_SCENARIOS).
    """
    eval_emis_sets, *_ = utils_inverse.generate_eval_data(
        AGENTS, DECK=False, CS3=False, DAMIP=False, GeoMIP=False
    )
    tier1_emis = eval_emis_sets["Tier 1"]  # {historical, H-ext, M, ML, L, VLHO, VLLO-ext}

    eval_targets_sets, *_, lat_coords = utils_inverse.generate_target_data(
        {"Tier 1": TIER1_SCORED_SCENARIOS}, data_dir=MESM_ZONAL_DIR
    )
    tier1_targets_raw = eval_targets_sets["Tier 1"]

    weights = np.cos(np.deg2rad(lat_coords))
    weights = np.maximum(weights, 1e-6)
    weights = weights / weights.sum()

    cropped = utils_inverse.build_dataset_vector_targets(
        tier1_emis, tier1_targets_raw, TIER1_FUTURE_SCENARIOS, agents=AGENTS
    )
    tier1_target_gmst = {
        scen: jnp.asarray(np.sum(np.asarray(y) * weights[None, :], axis=1), dtype=jnp.float32)
        for (_X, y, scen) in cropped
    }

    hist_raw = np.asarray(tier1_targets_raw[HISTORICAL_NAME])
    tier1_target_gmst[HISTORICAL_NAME] = jnp.asarray(
        np.sum(hist_raw * weights[None, :], axis=1), dtype=jnp.float32
    )

    eval_emis_sets_deck, eval_targets_sets_deck, lat_coords_deck = fidelity.build_eval_sets()
    deck_emis = eval_emis_sets_deck["DECK"]  # {"1pctCO2": (1,T), "2xCO2": (1,T)}
    deck_truth = fidelity.mesm_global_truth_by_scenario(
        deck_emis, eval_targets_sets_deck["DECK"], lat_coords_deck
    )
    tier1_emis.update(deck_emis)
    tier1_target_gmst.update({
        scen: jnp.asarray(arr, dtype=jnp.float32) for scen, arr in deck_truth.items()
    })

    return tier1_emis, tier1_target_gmst


