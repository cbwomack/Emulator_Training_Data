#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Companion script for 2d_calibrate_MESM_alt.ipynb: extends
scripts/2c_calibrate_MESM.py's climate-sensitivity stage (Phase 2) from a
single idealized experiment (1pctCO2, concentration-prescribed) to ALL SEVEN
Tier 1 ScenarioMIP experiments (historical, H-ext, M, ML, L, VLHO, VLLO-ext)
PLUS DECK's 1pctCO2 and 2xCO2 (emissions-driven this time, not the
concentration-prescribed pair Phase 1 already uses), run through the FULL
emissions-driven pipeline (emissions -> concentration -> temperature) against
MESM's own real emissions-driven ensemble-mean output.

Why 1pctCO2/2xCO2 were added on this pass (see REVISIONS.md for the fuller
diagnosis): the CO2-only Tier1-only joint fit fit the 6 future Tier 1
scenarios and `historical` well (R^2 >= 0.90 everywhere), but generalized
badly to `2xCO2` (R^2 <~ -15) and, more subtly, decayed too fast right after
each scenario's emissions peak (L/VLHO/VLLO-ext). Root causes, confirmed by
inspecting the SCM's own internal Catm_ppm/RF trajectories (not just GMST):
  - `2xCO2`'s driving file front-loads a huge one-year emissions pulse
    (~2317 GtCO2) to jump concentration to 2x almost instantly. The fitted
    carbon-cycle 4-pool partition (theta's "Carbon" group), calibrated only
    against 1pctCO2's smooth ~1%/yr ramp, had never been tested against a
    pulse this large - the SCM's own Catm_ppm spiked to 566 ppm at year 1
    then kept draining down to 432 ppm by year 239 instead of holding a 2x
    plateau, and GMST inherited that overshoot-then-decay shape faithfully
    (the fitted thermal-box q coefficients are constrained positive, so a
    pure step in forcing cannot itself produce an overshoot - the artifact
    is upstream, in the carbon cycle).
  - The post-peak decay-too-fast pattern traced to a genuine gap in the
    fitted thermal timescale spectrum (d ~= [0.26, 12, 354] years) - nothing
    represents multi-decadal ocean thermal inertia between the ~12-year and
    ~354-year modes, so ~half the total equilibrium warming unwinds almost
    as fast as forcing declines.
Including 1pctCO2 (a slow ramp, reinforcing Phase 1's own signal) and 2xCO2
(a genuine pulse) directly in the joint fit is meant to close exactly this
gap: it is the only way this pipeline's calibration data will ever include a
forcing shape that actually stresses the carbon-cycle pulse response, rather
than leaving it fit only to smooth, monotonic ramps.

Why concentration-prescribed calibration can't simply be reused for Tier 1:
MESM's concentration-driven ensemble (needed for calibrate_climate_sensitivity)
only exists for two idealized experiments - confirmed directly:
data/MESM/conc_driven/ has exactly two subdirectories, 1PRCO2/ and 2CO2/.
There is no concentration-driven Tier 1 run to fit against. What Tier 1 DOES
have is real emissions-driven MESM output (data/MESM/emis_driven/
zonal_data_mean/Tier 1/*.pkl), the same ground truth already used by
scripts/6d_scm_mesm_fidelity.py (Additional Major Point 4) and
6k_scm_mesm_fidelity_plot.ipynb - reused directly here via the same
utils_inverse functions (generate_eval_data, generate_target_data,
build_dataset_vector_targets), not reimplemented a second time.

Two consequences of that data gap, both deliberate:

  - Phase 1 (carbon cycle, emis -> conc) is UNCHANGED, still 1pctCO2-only -
    it is the only experiment with a real concentration target. Reused
    directly from scripts/2c_calibrate_MESM.py via importlib (its filename
    starts with a digit, so it can't be imported with a normal `import`
    statement - same pattern already used by 6k_scm_mesm_fidelity_plot.ipynb
    for 6d_scm_mesm_fidelity.py), not recomputed independently.
  - The new Phase 2 (this script's Tier1-joint stage) REPLACES, rather than
    parallels, the old climate-sensitivity stage: since the forward model now
    runs the full pipeline (utils_inverse.simulate_targets_gmst, not
    simulate_temp_prescribed_conc), errors from both the carbon cycle and the
    thermal response show up mixed together in one number. So this stage
    jointly unmasks the Carbon AND Climate theta groups together (continuing
    from Phase 1's already-carbon-tuned theta, not re-freezing it), rather
    than Climate alone - utils_FaIR_JAX.calibrate_climate_sensitivity's
    target="Climate"-only design assumed a clean prescribed-concentration
    signal that isolates the thermal response; that assumption no longer
    holds once concentration itself is simulated, not given. The aerosol/
    CH4/N2O theta groups (make_theta_mask targets "Aer"/"CH4"/"N2O") stay
    frozen at their theta0 (FaIR-derived, per make_theta0(mode='FaIR') -
    same as 2c's own theta0) values throughout - out of scope for "extend the
    existing calibration to Tier 1"; it would need its own scoped decision.

CO2-only, matching MESM's actual forcing (correction, see REVISIONS.md): MESM's
real Tier 1 runs were themselves only ever forced with CO2 emissions - this is
the same established scope scripts/6d_scm_mesm_fidelity.py and
6k_scm_mesm_fidelity_plot.ipynb already use (agents=['CO2']) for Tier2/DECK/CS3.
An earlier version of this script drove the SCM with real multi-agent
(CO2+CH4+N2O+Sulfur+BC) Tier 1 emissions while scoring against that same
CO2-only-forced MESM ground truth - a genuine input/target mismatch, not a
deliberate joint multi-agent fit. Fixed: build_tier1_dataset and
run_tier1_joint build their emissions dicts with agents=AGENTS=('CO2',), so the
aerosol/CH4/N2O theta groups are not merely left untuned in this stage, they
are never exercised by the forward pass at all (their input emissions are
zero-filled), consistent with MESM's own forcing.

CORRECTION (second bug, found later): that CO2-only scoping was correct for
*building* the emissions dicts, but run_tier1_joint additionally passed
agents=AGENTS through to utils_inverse.simulate_targets_gmst, which is where it
did damage. That builds a 1-ROW (CO2-only) emissions array, and
utils_FaIR_JAX.simulate_temp then indexes rows idx_CH4=1 .. idx_BC=4 - all out
of bounds - which JAX SILENTLY CLAMPS to the last valid row, i.e. row 0. The
CO2 emissions series was therefore ALSO being injected as CH4, N2O, sulfur and
BC emissions at once: the exact opposite of the zero-filling described above,
and large (+0.18 K at year 1, up to 0.65 K on Tier 1's 'M'). simulate_targets_
gmst's own `agents` default is the full 5-tuple and zero-fills whatever is
absent from emis_curr_dict, so the fix is simply not to pass `agents` there
(matching scripts/6d_scm_mesm_fidelity.py, whose scoring this stage is meant to
be comparable with). utils_FaIR_JAX.simulate_temp now also raises on a
short emissions array rather than silently clamping.

NOTE: data/JAX_calibration/calib_MESM_tier1_joint.pkl was produced BEFORE this
fix and is therefore stale - it was fitted against the corrupted forward model.
Re-run this script to regenerate it before using those parameters for anything.
Nothing outside this script currently reads that checkpoint.

Per-scenario error normalization, keeping in line with the convention used
throughout this project's own inner-loop test loss
(utils_inverse.avg_nrmse_over_tests, which every Figure 3/4/5/6/7 result is
built on): each scenario's error is
    calc_nrmse(pred, true) = RMSE(pred, true) / max(|true|)
(utils_FaIR_JAX.calc_nrmse - the same NRMSE utils_inverse._nrmse delegates
to), so no single scenario's absolute temperature scale can dominate the
others purely because it runs warmer (H-ext peaks near 6.5 K; VLHO peaks
near 1.3 K). Scenarios are then combined by a MEAN WEIGHTED BY SCENARIO
LENGTH - not an unweighted mean (this project's other multi-scenario
calibration loss, utils_FaIR_JAX.loss_fn / mean_nrmse_from_outputs, uses an
unweighted mean, but Tier 1's own scenario lengths span 127-477 years, a
>3x range, and avg_nrmse_over_tests - the convention actually used for the
paper's own bilevel-optimization objective - weights by length for exactly
this reason).

A real alignment bug was found and fixed while building this (see
build_tier1_dataset's docstring): 'historical' needed special handling
that 6d_scm_mesm_fidelity.py's SCM-direct comparison does not currently
apply. Flagged, not silently propagated - see REVISIONS.md.

Usage:
    python 2d_calibrate_MESM_alt.py
    python 2d_calibrate_MESM_alt.py --phase carbon   # just Phase 1 (identical to 2c's own)
    python 2d_calibrate_MESM_alt.py --phase tier1    # just the new Tier1-joint phase, chained
                                                       # from the saved carbon-cycle theta if it
                                                       # exists, else from a bare theta0 (warning)
"""
import argparse
import importlib.util
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax

import utils_FaIR_JAX
import utils_inverse
from utils_FaIR_JAX import MESM_PARAMS

# Reuse Phase 1 (carbon cycle) verbatim from 2c_calibrate_MESM.py, and DECK's
# 1pctCO2/2xCO2 emissions-loading from 6d_scm_mesm_fidelity.py (the raw MESM
# driving files, not utils_inverse.generate_eval_data's generic DECK builder -
# 6d's own build_eval_sets loads the exact files MESM itself was forced with,
# see its own comments) - both filenames start with a digit, so neither can
# be `import`ed normally.
_spec = importlib.util.spec_from_file_location(
    "s2c_calibrate_MESM", PROJECT_ROOT / "scripts" / "2c_calibrate_MESM.py"
)
s2c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s2c)

_spec6d = importlib.util.spec_from_file_location(
    "s6d_fidelity", PROJECT_ROOT / "scripts" / "6d_scm_mesm_fidelity.py"
)
s6d = importlib.util.module_from_spec(_spec6d)
_spec6d.loader.exec_module(s6d)

CARBON_FILEPATH = s2c.CARBON_FILEPATH  # data/JAX_calibration/calib_MESM_emis_to_conc.pkl - shared, unchanged
TIER1_FILEPATH = "data/JAX_calibration/calib_MESM_tier1_joint.pkl"  # name predates the DECK addition; scope now Tier1+1pctCO2+2xCO2
MESM_ZONAL_DIR = "data/MESM/emis_driven/zonal_data_mean/"

# MESM's own Tier 1 runs are CO2-only forced - matches
# scripts/6d_scm_mesm_fidelity.py's AGENTS exactly (single source of truth for
# "what did MESM actually run").
AGENTS = ("CO2",)

TIER1_FUTURE_SCENARIOS = ["H-ext", "M", "ML", "L", "VLHO", "VLLO-ext"]
HISTORICAL_NAME = "historical"
TIER1_SCORED_SCENARIOS = TIER1_FUTURE_SCENARIOS + [HISTORICAL_NAME]

# DECK's emissions-driven 1pctCO2/2xCO2 (CO2-only, same MESM driving files
# 6d_scm_mesm_fidelity.py uses), added to directly expose the joint fit to a
# genuine forcing pulse (2xCO2) - see module docstring.
DECK_TRAIN_SCENARIOS = ["1pctCO2", "2xCO2"]
ALL_TRAIN_SCENARIOS = TIER1_SCORED_SCENARIOS + DECK_TRAIN_SCENARIOS

# utils_inverse.build_dataset_vector_targets's own documented convention
# (its inline comment: "Historical Emissions start 1750, Targets start
# 1861."): MESM's real 'historical' emissions-driven run covers calendar
# years 1861 onward, not the SCM's full 1750-2023 historical span. Confirmed
# independently here: the SCM's own historical CO2 emissions at index 111
# (calendar year 1861) match data/MESM/emis_driven/MESM_inputs/
# historical_emis_year.txt's 1861 row exactly (0.347791 GtCO2/yr). That
# function already applies this offset to the MLP-training feature side
# (emis_offset_hist=111) for Figure 7's own MESM evaluation - it is not a
# new convention, just one this script's SCM-direct comparison must apply
# too, since it doesn't go through build_dataset_vector_targets's feature
# path at all.
EMIS_OFFSET_HIST = 111


def build_tier1_dataset() -> tuple[dict, dict]:
    """Real Tier 1 CO2-only emissions (AGENTS) + MESM's own emissions-driven
    ensemble-mean global temperature, one entry per scored scenario, PLUS
    DECK's emissions-driven 1pctCO2/2xCO2 (see ALL_TRAIN_SCENARIOS). Same
    construction as scripts/6d_scm_mesm_fidelity.py's Tier 1 half for the six
    future scenarios (generate_eval_data + generate_target_data +
    build_dataset_vector_targets's existing target_crop_future=162 crop, then
    area-weighted to a global mean by cos(latitude) - not reimplemented a
    second time).

    'historical' is handled separately and NOT via that same crop: unlike the
    other Tier1 scenarios, MESM's own 'historical' emissions-driven run
    starts at calendar year 1861, not 1750 (see EMIS_OFFSET_HIST) - a real,
    previously-undocumented gap in scripts/6d_scm_mesm_fidelity.py, which
    compares the SCM's simulated years 1750-1893 against MESM's real
    1861-2004ish output for this one scenario (confirmed directly: its
    reported 'historical' NRMSE=0.350/R^2=-0.807 predates this fix). Flagged
    in REVISIONS.md rather than silently carried forward here or silently
    patched into 6d's already-published numbers without the user's say-so.

    DECK's 1pctCO2/2xCO2 are pulled from scripts/6d_scm_mesm_fidelity.py's
    own build_eval_sets/mesm_global_truth_by_scenario (the exact raw MESM
    driving files and crop/alignment logic Stage 6d already validated for
    these two scenarios - not reimplemented). Neither needs 'historical'
    prefix context: both are idealized standalone experiments (confirmed:
    neither is in utils_inverse.scens_with_hist), so their SCM run starts
    from year 0 of their own emissions file, matching Stage 6d's own
    treatment exactly.
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

    # DECK's 1pctCO2/2xCO2, reusing Stage 6d's own eval-set + target
    # construction verbatim (see docstring above).
    eval_emis_sets_deck, eval_targets_sets_deck, lat_coords_deck = s6d.build_eval_sets()
    deck_emis = eval_emis_sets_deck["DECK"]  # {"1pctCO2": (1,T), "2xCO2": (1,T)}
    deck_truth = s6d.mesm_global_truth_by_scenario(
        deck_emis, eval_targets_sets_deck["DECK"], lat_coords_deck
    )
    tier1_emis.update(deck_emis)
    tier1_target_gmst.update({
        scen: jnp.asarray(arr, dtype=jnp.float32) for scen, arr in deck_truth.items()
    })

    return tier1_emis, tier1_target_gmst


def run_tier1_joint(theta0: jnp.ndarray, n_steps: int = 1000, learning_rate: float = 1e-2) -> jnp.ndarray:
    """New Phase 2: jointly re-tunes the Carbon + Climate theta groups against
    all seven Tier 1 scenarios PLUS DECK's 1pctCO2/2xCO2 (ALL_TRAIN_SCENARIOS)
    real MESM emissions-driven temperature, through the FULL emissions-driven
    pipeline. See module docstring for why this replaces (rather than
    parallels) calibrate_climate_sensitivity, for the DECK addition's
    motivation, and for the per-scenario NRMSE normalization / length-
    weighting convention.
    """
    tier1_emis, tier1_target_gmst = build_tier1_dataset()

    years_hist_full, emis_hist_dict_full = utils_inverse.extract_years_and_emis(
        tier1_emis[HISTORICAL_NAME], agents=AGENTS
    )

    per_scen = {}
    for scen in TIER1_FUTURE_SCENARIOS:
        yrs_cur, emis_cur_dict = utils_inverse.extract_years_and_emis(
            tier1_emis[scen], agents=AGENTS
        )
        # Every Tier1 future scenario is in utils_inverse.scens_with_hist -
        # confirmed directly, so historical context always applies here.
        per_scen[scen] = dict(
            years_curr=yrs_cur, emis_curr_dict=emis_cur_dict,
            years_hist=years_hist_full, emis_hist_dict=emis_hist_dict_full,
        )
    # 'historical' itself: run stand-alone (no additional prior context),
    # then slice by EMIS_OFFSET_HIST to align with its own real MESM window.
    per_scen[HISTORICAL_NAME] = dict(
        years_curr=years_hist_full, emis_curr_dict=emis_hist_dict_full,
        years_hist=None, emis_hist_dict=None,
    )
    # DECK's 1pctCO2/2xCO2: also stand-alone - neither is in
    # utils_inverse.scens_with_hist (confirmed), matching Stage 6d's own
    # treatment of these two idealized experiments.
    for scen in DECK_TRAIN_SCENARIOS:
        yrs_cur, emis_cur_dict = utils_inverse.extract_years_and_emis(
            tier1_emis[scen], agents=AGENTS
        )
        per_scen[scen] = dict(
            years_curr=yrs_cur, emis_curr_dict=emis_cur_dict,
            years_hist=None, emis_hist_dict=None,
        )
    hist_len = int(tier1_target_gmst[HISTORICAL_NAME].shape[0])

    def scenario_nrmse(theta, scen):
        params = utils_FaIR_JAX.params_from_theta(theta, base_params=MESM_PARAMS)
        # BUGFIX: this used to pass agents=AGENTS (=("CO2",)) here. That built a
        # 1-ROW emissions array, and utils_FaIR_JAX.simulate_temp then indexes
        # rows idx_CH4=1, idx_N2O=2, idx_Sulfur=3, idx_BC=4 - all out of bounds,
        # which JAX SILENTLY CLAMPS to the last valid row (row 0, CO2). The CO2
        # emissions series was therefore also being fed in as CH4, N2O, sulfur
        # and BC emissions simultaneously - the exact opposite of the zero-filled
        # CO2-only forcing this module's docstring describes. Measured effect:
        # +0.18 K at year 1 and up to 0.65 K on Tier 1's 'M'.
        # Leaving `agents` at simulate_targets_gmst's 5-agent default makes it
        # zero-fill the four absent agents from emis_curr_dict (which
        # extract_years_and_emis already built CO2-only), which is what
        # scripts/6d_scm_mesm_fidelity.py does and what makes this loss
        # consistent with 6d's scoring.
        pred = utils_inverse.simulate_targets_gmst(
            mode="MESM", dt=0.1, params=params, use_checkpoint=True, **per_scen[scen]
        )
        if scen == HISTORICAL_NAME:
            pred = pred[EMIS_OFFSET_HIST:EMIS_OFFSET_HIST + hist_len]
        true = tier1_target_gmst[scen]
        n = min(pred.shape[0], true.shape[0])
        return utils_FaIR_JAX.calc_nrmse(pred[:n], true[:n]), n

    # One value_and_grad closure PER SCENARIO, rather than differentiating a
    # single function that calls simulate_targets_gmst (and hence lax.scan)
    # once per scenario. Mathematically identical - the gradient of a
    # length-weighted mean is the length-weighted mean of the per-term
    # gradients, so combining nine separately-computed (loss, grad) pairs
    # below gives exactly the same descent direction a single combined trace
    # would. Forced by a real, deterministic OOM in this container: tracing
    # and differentiating all scenarios' lax.scan calls in ONE jax.grad trace
    # exhausts this container's RAM+swap during XLA's compile step, before
    # any optimization step runs (confirmed deterministically for the
    # Tier1-only fit, see REVISIONS.md). Splitting into independent,
    # much-smaller value_and_grad calls keeps peak compile memory bounded by
    # the single largest scenario instead of their sum.
    per_scenario_value_and_grad = {
        scen: jax.value_and_grad(lambda th, s=scen: scenario_nrmse(th, s)[0])
        for scen in ALL_TRAIN_SCENARIOS
    }
    lengths = jnp.asarray(
        [scenario_nrmse(theta0, scen)[1] for scen in ALL_TRAIN_SCENARIOS], dtype=jnp.float32
    )
    scenario_weights = lengths / jnp.sum(lengths)

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(theta0)
    theta = theta0

    # Unmask Carbon AND Climate together - see module docstring for why this
    # differs from calibrate_climate_sensitivity's Climate-only mask.
    mask = jnp.clip(
        utils_FaIR_JAX.make_theta_mask(theta0, target="Carbon")
        + utils_FaIR_JAX.make_theta_mask(theta0, target="Climate"),
        0.0, 1.0,
    )

    print("Starting Tier1-joint calibration (full emissions -> temperature pipeline, "
          f"{len(ALL_TRAIN_SCENARIOS)} scenarios ({ALL_TRAIN_SCENARIOS}), "
          "length-weighted mean NRMSE, compiled per-scenario)...")
    for step in range(n_steps):
        losses, grads = [], []
        for scen in ALL_TRAIN_SCENARIOS:
            l, g = per_scenario_value_and_grad[scen](theta)
            losses.append(l)
            grads.append(g)
        losses = jnp.stack(losses)
        grads = jnp.stack(grads)  # (n_scenarios, theta_dim)

        loss = jnp.sum(losses * scenario_weights)
        grad = jnp.sum(grads * scenario_weights[:, None], axis=0)

        grad = grad * mask
        updates, opt_state = optimizer.update(grad, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

        if step % 100 == 0 or step == n_steps - 1:
            print(f"Step {step} | Loss (Tier1 length-weighted mean NRMSE): {loss:.4f}")
            # Checkpoint every 100 steps (same cadence as the print above,
            # same pattern as utils_FaIR_JAX.calibrate_inverse's own periodic
            # pickle.dump) - a crash between here and the final save (this
            # run has already been OOM-killed four times in this container,
            # see REVISIONS.md) loses at most 100 steps, not the whole run.
            with open(TIER1_FILEPATH, "wb") as f:
                pickle.dump(theta, f)

    print(f"Saved {TIER1_FILEPATH}")
    return theta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["carbon", "tier1"], default=None,
                         help="Run only this phase (default: run both, sequentially)")
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    args = parser.parse_args()

    # Matches 2c_calibrate_MESM.py's own theta0 exactly (mode='FaIR', not
    # 'MESM' - base_params=MESM_PARAMS only supplies the fields theta has no
    # slot for; this is Phase 1's real starting point, unchanged).
    theta0 = utils_FaIR_JAX.make_theta0(mode="FaIR")

    if args.phase is None:
        theta_carbon = s2c.run_carbon_cycle(theta0)
        run_tier1_joint(theta_carbon, n_steps=args.n_steps, learning_rate=args.learning_rate)
    elif args.phase == "carbon":
        s2c.run_carbon_cycle(theta0)
    elif args.phase == "tier1":
        if os.path.isfile(CARBON_FILEPATH):
            with open(CARBON_FILEPATH, "rb") as f:
                theta0 = pickle.load(f)
        else:
            print(f"No saved carbon-cycle theta at {CARBON_FILEPATH} - "
                  "starting Tier1-joint calibration from a bare theta0 instead.")
        run_tier1_joint(theta0, n_steps=args.n_steps, learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
