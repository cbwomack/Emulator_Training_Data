#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage 6c - Point 3(b) (OOD scenario check): evaluates the existing
multi-agent optimized and baseline emulators (checkpoints/multi/, untouched
by this session's Phase 0 work) on a genuinely new, real, multi-agent
scenario that is NOT ScenarioMIP/DAMIP/GeoMIP/CS3/DECK - `ssp370-lowNTCF`,
a CMIP6 AerChemMIP experiment.

**Scenario and data source (per user direction, cited here and in
REVISIONS.md):**
- Scenario definition: AerChemMIP's SSP3-7.0-lowNTCF experiment - the same
  socioeconomic/CO2/N2O trajectory as SSP3-7.0, but with strong air-pollution
  mitigation driving down CH4, aerosol (SO2/Sulfur, BC), and tropospheric-
  ozone-precursor emissions. Collins, W. J., Lamarque, J.-F., Schulz, M., et
  al.: AerChemMIP: quantifying the effects of chemistry and aerosols in
  CMIP6, Geosci. Model Dev., 10, 585-607,
  https://doi.org/10.5194/gmd-10-585-2017, 2017.
- Emissions data: RCMIP protocol dataset v5.1.0, scenario
  'ssp370-lowNTCF-aerchemmip' (the AerChemMIP-protocol variant, not the
  later 'ssp370-lowNTCF-gidden' re-harmonization - this is the version
  actually used to force CMIP6 ESMs for this experiment), model 'AIM/CGE',
  region 'World'. Nicholls, Z. R. J., Meinshausen, M., Lewis, J., et al.:
  Reduced Complexity Model Intercomparison Project (RCMIP) Phase 1:
  introduction and evaluation of global-mean temperature response,
  Geosci. Model Dev., 13, 5175-5190,
  https://doi.org/10.5194/gmd-13-5175-2020, 2020.
  Downloaded from https://rcmip-protocols-au.s3-ap-southeast-2.amazonaws.com/
  v5.1.0/rcmip-emissions-annual-means-v5-1-0.csv (confirmed reachable and
  matching the archived v5.1.0 release, 2026-08-12) to
  data/RCMIP/rcmip-emissions-annual-means-v5-1-0.csv.

**Why this scenario satisfies the disqualification list:** distinct from
ScenarioMIP (this repo's Tier1/Tier2 come from a separate CMIP7-protocol
file, not RCMIP/SSP), DAMIP (single-forcing GHG/aerosol splits of one
ScenarioMIP scenario), GeoMIP (G6sulfur solar-geoengineering-style
experiment - already in-objective for Fig 6, disqualified regardless), CS3
(a distinct "Global Change Outlook 2025" AA/CT scenario pair), and DECK
(idealized abrupt-4x/1pct single-agent experiments). All 5 agents
(CO2/CH4/N2O/Sulfur/BC) are present and vary in this dataset.

**Units (verified directly from the RCMIP CSV's own 'Unit' column, not
assumed):** CO2 in Mt CO2/yr (-> GtCO2/yr, /1000, matching this repo's
simulate_targets_gmst convention), CH4 in Mt CH4/yr (matches directly, no
conversion), N2O in kt N2O/yr (-> Mt N2O/yr, /1000 - a unit-family
conversion only; NOT a molar-mass N2O->N conversion, since this repo's own
existing ScenarioMIP CSV (data/FaIR/extensions_1750-2500.csv) already stores
and feeds N2O as "Mt N2O/yr" directly with no molar conversion applied
anywhere in run_fair.load_scenarioMIP_CMIP7 - matching that existing,
already-calibrated-against convention exactly, rather than "fixing" a
possible deeper unit-labeling question in utils_FaIR_JAX.py that is outside
this stage's scope), Sulfur in Mt SO2/yr (matches this repo's MtSulfur
convention directly - SO2 mass basis, same as the existing CS3 loader's
'SO2:Tg' with no conversion), BC in Mt BC/yr (matches directly).

**Interpolation:** RCMIP's raw scenario rows report values at irregular
intervals (annual through 2014, then 5- and 10-yearly to 2100, then
10-yearly to 2500) - standard RCMIP protocol convention, expected to be
linearly interpolated by users (Nicholls et al. 2020, Section 2). Linear
interpolation (np.interp) is applied per species to produce a complete
annual series.

**Historical/future split:** for the pre-2024 period, uses this repo's own
existing calibrated historical emissions (data/FaIR's ScenarioMIP CSV, same
'historical' series used to train/evaluate every other scenario in this
pipeline) rather than RCMIP's own historical harmonization, so the new
scenario's features are built with the exact historical context the
optimized/baseline emulators were themselves trained against. From 2024
onward, uses RCMIP's ssp370-lowNTCF-aerchemmip values directly.

**Evaluating the EXISTING (not retrained) optimized emulator:** its trained
MLP params are deterministically reconstructed from checkpoints/multi_
retuned/seed_sweep/inverse_sine_all_all_agents_seed0.pkl's own stored
U_traj[-1] (the final, already-optimized emissions trajectory - no
re-optimization, no new outer-loop compute) replayed through the exact same
one-step inner-loop training (make_inverse_objective_single_train's own
build_train -> split_and_scale -> train_mlp_sgd), using the SAME
hyperparameters (step_size dict, K_inner/lr_inner/wd_inner/batch_size) the
regenerated checkpoint was itself produced with - data/SI_results/hp_retune/
multi/best_config_unified.json (the 2026-08-13 multi-agent Phase 0 retune
winner: K_inner=400, lr_inner=0.0489, wd_inner=0.03, batch_size=16 - see
REVISIONS.md) - used to produce that checkpoint's own paramsK_k in the first
place - deterministic and bit-exact, not a fresh fit. init_cond='sine' (not
'constant') for group 'all' matches 3b_inverse_all_agents.py's own
EXPERIMENTS['all']['init_cond'] exactly (the pre-Phase-0 checkpoints/multi/
directory used inconsistent per-group update counts and this file's own
former OPT_CKPT_PATH mismatched IC label; both are now moot since that
checkpoint dir is no longer used here - see REVISIONS.md Session Log on
checkpoint staleness). Reconstruction is self-validated by re-evaluating the
result against the checkpoint's own eval_sets and comparing to its stored
errors[-1] before trusting it on the new scenario (see main()).

**Evaluating the baseline emulator:** genuinely retrained (seed=0), using
data/SI_results/baseline_hp/k400_search_multi/best_baseline_config_K400.json
(the multi-agent K=400 dense search winner: K=400, lr=0.0271, weight_decay=
0.003 - confirmed NOT to transfer from CO2's single-forcing tuned config,
see REVISIONS.md), matching this repo's existing convention that baseline
scores are always freshly (but deterministically) retrained per evaluation
call, never loaded from a saved param file.

Usage:
    python 6c_ood_scenario_ssp370_lowntcf.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import json
import pickle
import numpy as np
import jax
import jax.numpy as jnp

import utils_inverse

RCMIP_CSV = "data/RCMIP/rcmip-emissions-annual-means-v5-1-0.csv"
RCMIP_SCENARIO = "ssp370-lowNTCF-aerchemmip"
RCMIP_MODEL = "AIM/CGE"
RCMIP_REGION = "World"
RCMIP_VARIABLES = {
    "CO2": "Emissions|CO2",       # Mt CO2/yr -> GtCO2/yr (/1000)
    "CH4": "Emissions|CH4",       # Mt CH4/yr, no conversion
    "N2O": "Emissions|N2O",       # kt N2O/yr -> Mt N2O/yr (/1000, unit-family only)
    "Sulfur": "Emissions|Sulfur", # Mt SO2/yr, no conversion
    "BC": "Emissions|BC",         # Mt BC/yr, no conversion
}
RCMIP_UNIT_DIVISOR = {"CO2": 1000.0, "CH4": 1.0, "N2O": 1000.0, "Sulfur": 1.0, "BC": 1.0}

AGENTS = ["CO2", "CH4", "N2O", "Sulfur", "BC"]
ACTIVE_AGENTS = ("CO2", "CH4", "N2O", "Sulfur", "BC")
MODE = "FaIR"
SEED = 0
MULTI_CKPT_DIR = "checkpoints/multi_retuned/seed_sweep"
OPT_CKPT_PATH = f"{MULTI_CKPT_DIR}/inverse_sine_all_all_agents_seed{SEED}.pkl"
# group 'all' uses init_cond='sine' per 3b_inverse_all_agents.py's own EXPERIMENTS['all']
UNIFIED_CONFIG_PATH = "data/SI_results/hp_retune/multi/best_config_unified.json"
BASELINE_CONFIG_PATH = "data/SI_results/baseline_hp/k400_search_multi/best_baseline_config_K400.json"
UNIFIED_CFG = json.load(open(UNIFIED_CONFIG_PATH))["config"]
BASELINE_CFG = json.load(open(BASELINE_CONFIG_PATH))["config"]
# The exact hyperparameters checkpoints/multi_retuned/ was regenerated with
# (0c_regenerate_checkpoints_multi.py) - must match bit-exactly for the
# reconstruction replay below to be deterministic against the checkpoint's
# own stored paramsK_k.
OPT_HP = {"K_inner": UNIFIED_CFG["K_inner"], "lr_inner": UNIFIED_CFG["lr_inner"],
          "wd_inner": UNIFIED_CFG["wd_inner"], "batch_size": UNIFIED_CFG["batch_size"]}
BASELINE_HP = {"K": BASELINE_CFG["K"], "lr": BASELINE_CFG["lr"], "weight_decay": BASELINE_CFG["weight_decay"]}
FUTURE_START_YEAR = 2024
FUTURE_END_YEAR = 2100
OUT_DIR = Path("data/SI_results/ood_scenario")


def load_rcmip_species(agent):
    """One species' {year: value} from the RCMIP CSV, native units."""
    variable = RCMIP_VARIABLES[agent]
    with open(RCMIP_CSV, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        year_cols = [(i, int(h)) for i, h in enumerate(header) if h.isdigit()]
        for row in r:
            if row[0] == RCMIP_MODEL and row[1] == RCMIP_SCENARIO and row[2] == RCMIP_REGION and row[3] == variable:
                return {yr: float(row[i]) for (i, yr) in year_cols if row[i].strip() != ""}
    raise ValueError(f"No RCMIP row found for {RCMIP_MODEL}/{RCMIP_SCENARIO}/{RCMIP_REGION}/{variable}")


def build_future_emissions():
    """{agent: (T,) array}, T = FUTURE_END_YEAR - FUTURE_START_YEAR + 1, linearly
    interpolated from RCMIP's sparse reported years, native-to-repo units.
    Returned years are REAL calendar years (needed to interpolate from
    RCMIP's own year-indexed columns, and useful for plotting) - NOT the
    0-based relative index simulate_targets_gmst expects; see
    build_new_scenario_features for why that distinction matters."""
    years_curr = np.arange(FUTURE_START_YEAR, FUTURE_END_YEAR + 1)
    out = {}
    for agent in AGENTS:
        raw = load_rcmip_species(agent)
        yrs = np.array(sorted(raw.keys()))
        vals = np.array([raw[y] for y in yrs]) / RCMIP_UNIT_DIVISOR[agent]
        out[agent] = np.interp(years_curr, yrs, vals).astype(np.float32)
    return years_curr, out


def build_new_scenario_features(setup):
    """Explicit, correct (historical-prepended, relative-years) construction
    of the new scenario's (X, y) pair. This bypasses two failure modes a
    more convenient path would hit silently:

    1. extract_years_and_emis (used everywhere else in this codebase) always
       returns a 0-based relative index (np.arange(T)), NOT real calendar
       years - every existing scenario's "years" are relative, and
       simulate_targets_gmst's _make_contiguous_years relies on that to
       rebase the future segment to start right after historical ends. This
       scenario's own years_curr from build_future_emissions are REAL
       calendar years (2024, 2025, ...) so that they can be used to
       interpolate RCMIP's year-indexed data - but fed directly into
       simulate_targets_gmst, _make_contiguous_years sees 2024 > 273
       (historical's own 0-based length) and wrongly concludes the future
       segment is "already contiguous," concatenating [0..273, 2024..2100]
       with a huge fake gap instead of rebasing to [0..273, 274..350].
       Fixed by passing a fresh 0-based relative array (matching every other
       scenario's own convention) into simulate_targets_gmst/
       make_features_emissions_generic instead of the real calendar years.
    2. build_dataset_from_runfair_dict (used by evaluate_baseline_over_
       multiple_tests, the "convenient" path for the baseline emulator)
       decides whether to prepend historical context via a hardcoded
       scenario-name allowlist (scens_with_hist) that has no entry for
       'ssp370-lowNTCF' - routing through it would silently build features
       WITHOUT historical context at all. This function always prepends
       historical explicitly, for both the optimized and baseline paths.
    """
    years_hist, emis_hist_dict = utils_inverse.extract_years_and_emis(
        setup["emis_dict_train_JAX"]["historical"], agents=AGENTS
    )
    years_curr_calendar, emis_curr_dict = build_future_emissions()
    years_curr_relative = jnp.arange(len(years_curr_calendar), dtype=jnp.float32)

    X = utils_inverse.make_features_emissions_generic(
        emis_curr_dict=emis_curr_dict, emis_hist_dict=emis_hist_dict, agents=AGENTS,
        ema_windows_years=(5.0, 30.0, 100.0), dt_years=1.0, zero_fill_missing=True,
    )
    y = utils_inverse.simulate_targets_gmst(
        years_curr=years_curr_relative, emis_curr_dict=emis_curr_dict,
        years_hist=years_hist, emis_hist_dict=emis_hist_dict, agents=AGENTS, mode=MODE,
    )
    n = min(X.shape[0], y.shape[0])
    return X[:n], y[:n], years_curr_calendar[:n]


def reconstruct_optimized_paramsK(setup, ckpt, u_index=-1):
    """Deterministically replay the inner-loop training step that produces a
    paramsK from a given U_traj[u_index] - see module docstring.

    NOTE on indexing: optimize_emissions_inverse's outer loop computes
    loss_k/aux_pure (hence paramsK_k) from update_step(U, ...) at the INPUT
    U, but update_step returns new_U (the POST-update state); both get
    appended at the same list index. So errors[i]/the checkpoint's own
    stored paramsK_k actually correspond to U_traj[i-1], not U_traj[i] - a
    pre-existing off-by-one in that function, not introduced here. main()
    validates this by reconstructing from U_traj[-2] and checking it matches
    errors[-1], before using U_traj[-1] (the true final, most-optimized
    state) for the actual new-scenario evaluation."""
    U_final = ckpt["U_traj"][u_index]
    U_eff = utils_inverse._apply_active_mask_to_emis(U_final, ACTIVE_AGENTS, inactive_mode="zeros")

    train_updated = utils_inverse.build_train(
        U_eff, agents=AGENTS, ema_windows_years=(5.0, 30.0, 100.0),
        years_hist=None, emis_hist_dict=None, mode=MODE,
    )
    train_s, _test_s, stats = utils_inverse.split_and_scale(train_updated, train_updated)

    Xtr = jnp.concatenate([X for (X, _, _) in train_s], axis=0).astype(jnp.float32)
    ytr = jnp.concatenate([y for (_, y, _) in train_s], axis=0).astype(jnp.float32)

    paramsK, _losses = utils_inverse.train_mlp_sgd(
        setup["params0"], Xtr, ytr,
        K=OPT_HP["K_inner"], lr=OPT_HP["lr_inner"], weight_decay=OPT_HP["wd_inner"],
        batch_size=OPT_HP["batch_size"], key=jax.random.PRNGKey(SEED),
    )
    return paramsK, stats


def validate_reconstruction(setup, paramsK, stats, ckpt):
    """Self-check: re-evaluate the reconstructed paramsK (from U_traj[-2],
    see reconstruct_optimized_paramsK's docstring) on the checkpoint's own
    eval_sets and compare to its stored errors[-1] before trusting the
    reconstruction method at all."""
    group_emis_dicts = utils_inverse.build_group_emis_dicts(setup["emis_dict_train_JAX"], setup["eval_sets"])
    test_dataset_all = utils_inverse.build_valid(
        group_emis_dicts["all"], historical_name="historical", agents=AGENTS, mode=MODE,
    )
    test_scaled = [(utils_inverse.apply_scaler(X, stats), y, scen) for (X, y, scen) in test_dataset_all]
    reconstructed_nrmse = float(utils_inverse.avg_nrmse_over_tests(paramsK, test_scaled))
    stored_nrmse = float(ckpt["errors"][-1])
    print(f"[validate] reconstructed NRMSE={reconstructed_nrmse:.6f} vs. checkpoint's stored "
          f"errors[-1]={stored_nrmse:.6f} (diff={abs(reconstructed_nrmse - stored_nrmse):.6f})")
    return reconstructed_nrmse, stored_nrmse


def evaluate_on_new_scenario(setup, paramsK, stats):
    X, y, _years = build_new_scenario_features(setup)
    Xs = utils_inverse.apply_scaler(X, stats)
    yhat = utils_inverse.mlp_forward(paramsK, Xs)
    nrmse = float(utils_inverse._nrmse(jnp.asarray(yhat), jnp.asarray(y)))
    return nrmse, len(y)


def evaluate_baseline_on_new_scenario(setup):
    # Train the baseline fresh on Tier 1 (matches evaluate_baseline_over_
    # multiple_tests's own logic exactly - baseline scores are always
    # freshly, deterministically retrained per evaluation call in this
    # repo, never loaded from a saved param file), then apply it to the
    # new scenario's explicitly-built (historical-prepended) features
    # rather than routing through build_dataset_from_runfair_dict's
    # scenario-name allowlist (see build_new_scenario_features's docstring).
    train_s, _test_s, stats = utils_inverse.prepare_baseline_data(
        emis_dict_train=setup["eval_sets"]["Tier 1"],
        emis_dict_test=setup["eval_sets"]["Tier 1"],
        historical_name="historical", mode=MODE,
    )
    paramsK_base, _losses, _meta = utils_inverse.train_baseline_emulator(
        train_scaled=train_s, key=jax.random.PRNGKey(0),
        K=BASELINE_HP["K"], lr=BASELINE_HP["lr"], weight_decay=BASELINE_HP["weight_decay"],
    )
    X, y, _years = build_new_scenario_features(setup)
    Xs = utils_inverse.apply_scaler(X, stats)
    yhat = utils_inverse.mlp_forward(paramsK_base, Xs)
    return float(utils_inverse._nrmse(jnp.asarray(yhat), jnp.asarray(y)))


def main():
    setup = utils_inverse.run_inverse_experiment_setup(
        AGENTS, ACTIVE_AGENTS, mode=MODE, CS3=True, DAMIP=True, GeoMIP=True,
        idx_demo=None,  # skip create_baseline's demo plot (needs latex, unavailable on compute nodes)
        seed=SEED,
    )

    ckpt = utils_inverse.load_inverse_ckpt(OPT_CKPT_PATH)

    # Validate the reconstruction METHOD using U_traj[-2], which is what
    # errors[-1]/the checkpoint's own paramsK_k actually correspond to (see
    # reconstruct_optimized_paramsK's docstring on the off-by-one).
    paramsK_check, stats_check = reconstruct_optimized_paramsK(setup, ckpt, u_index=-2)
    recon_nrmse, stored_nrmse = validate_reconstruction(setup, paramsK_check, stats_check, ckpt)
    if abs(recon_nrmse - stored_nrmse) > 0.005:
        raise RuntimeError(
            f"Reconstructed optimized-emulator NRMSE from U_traj[-2] ({recon_nrmse:.4f}) doesn't "
            f"match {OPT_CKPT_PATH}'s own stored errors[-1] ({stored_nrmse:.4f}) closely enough to "
            f"trust the reconstruction method - stopping rather than reporting a number built on a "
            f"broken replay."
        )
    print(f"[validate] reconstruction method confirmed (U_traj[-2] matches errors[-1] to "
          f"{abs(recon_nrmse - stored_nrmse):.6f}) - proceeding with U_traj[-1], the true final state")

    # Now apply the (just-validated) method to U_traj[-1] - the actual,
    # fully-optimized final state - for the real new-scenario evaluation.
    paramsK_opt, stats = reconstruct_optimized_paramsK(setup, ckpt, u_index=-1)

    opt_nrmse, n_years = evaluate_on_new_scenario(setup, paramsK_opt, stats)
    print(f"[ssp370-lowNTCF] optimized emulator NRMSE={opt_nrmse:.4f} (n={n_years} years)")

    baseline_nrmse = evaluate_baseline_on_new_scenario(setup)
    print(f"[ssp370-lowNTCF] baseline emulator NRMSE={baseline_nrmse:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ssp370_lowntcf_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "scenario": RCMIP_SCENARIO,
            "optimized_nrmse": opt_nrmse,
            "baseline_nrmse": baseline_nrmse,
            "n_years": n_years,
            "reconstruction_validation": {"reconstructed_nrmse": recon_nrmse, "stored_nrmse": stored_nrmse},
        }, f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
