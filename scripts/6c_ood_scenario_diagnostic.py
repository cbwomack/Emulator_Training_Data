#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage 6c(b) diagnostic cache: builds everything needed to visually verify the
ssp370-lowNTCF ingestion (scripts/6c_ood_scenario_ssp370_lowntcf.py) before
trusting its NRMSE numbers.

This diagnostic exists because a real bug was found while building it (per
user direction to "confirm the historical period is prepended correctly,
otherwise the emulation will fail" - that concern was well-founded):
extract_years_and_emis (used throughout this codebase) always returns a
0-based relative index, not real calendar years, and simulate_targets_gmst's
_make_contiguous_years relies on that convention to rebase a future
segment's years onto the end of historical. Feeding it REAL calendar years
(2024, 2025, ...) instead - needed to interpolate RCMIP's own year-indexed
data - defeated that rebase check (2024 > historical's own 0-based length of
273) and silently produced a huge fake gap in the internal "years" axis used
for GMST integration. Fixed in build_new_scenario_features (see that
function's docstring); this script's own historical/future concatenation
below uses REAL calendar years throughout (1750-2023 historical, 2024-2100
future - a directly human-checkable, contiguous span) specifically so the
boundary can be visually/numerically confirmed with no relative-vs-absolute
ambiguity, independent of - and as a cross-check on - the fix in the
evaluation script.

Per-agent emissions (historical, future, and the concatenated full series)
and one continuous SCM temperature simulation spanning both (mode='FaIR',
run directly via utils_FaIR_JAX.simulate_temp on the real-calendar-year
concatenated series) are saved, plus the baseline emulator's actual
predictions vs. simulated truth on the new scenario (using the same
build_new_scenario_features helper the evaluation script uses, so this
diagnostic reflects exactly what that script computes).

Usage:
    python 6c_ood_scenario_diagnostic.py
"""
import os
import sys
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import numpy as np
import jax

import utils_inverse
import utils_FaIR_JAX

OUT_DIR = Path("data/SI_results/ood_scenario")
HISTORICAL_START_YEAR = 1750  # matches run_fair.load_scenarioMIP_CMIP7's n_years_hist=274 convention


def load_ssp370_module():
    spec = importlib.util.spec_from_file_location(
        "_stage6c_ssp370", PROJECT_ROOT / "scripts" / "6c_ood_scenario_ssp370_lowntcf.py")
    ssp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ssp)
    return ssp


def main():
    ssp = load_ssp370_module()

    setup = utils_inverse.run_inverse_experiment_setup(
        ssp.AGENTS, ssp.ACTIVE_AGENTS, mode=ssp.MODE, CS3=True, DAMIP=True, GeoMIP=True,
        idx_demo=None,
    )

    _years_hist_relative, emis_hist_dict = utils_inverse.extract_years_and_emis(
        setup["emis_dict_train_JAX"]["historical"], agents=ssp.AGENTS
    )
    n_hist = len(np.asarray(emis_hist_dict[ssp.AGENTS[0]]))
    years_hist_calendar = np.arange(HISTORICAL_START_YEAR, HISTORICAL_START_YEAR + n_hist)

    years_curr_calendar, emis_curr_dict = ssp.build_future_emissions()

    print(f"historical years: {years_hist_calendar[0]:.0f}-{years_hist_calendar[-1]:.0f} (n={n_hist})")
    print(f"future years: {years_curr_calendar[0]:.0f}-{years_curr_calendar[-1]:.0f} (n={len(years_curr_calendar)})")
    gap = years_curr_calendar[0] - years_hist_calendar[-1]
    print(f"boundary gap: {gap:.0f} year(s) between last historical year and first future year "
          f"(should be exactly 1)")
    if gap != 1:
        print("WARNING: historical/future years are not contiguous - see module docstring")

    for a in ssp.AGENTS:
        print(f"[{a}] last historical value: {np.asarray(emis_hist_dict[a])[-1]:.4f}, "
              f"first future value: {np.asarray(emis_curr_dict[a])[0]:.4f}")

    years_full = np.concatenate([years_hist_calendar, years_curr_calendar])
    emis_full = np.stack(
        [np.concatenate([np.asarray(emis_hist_dict[a]), np.asarray(emis_curr_dict[a])]) for a in ssp.AGENTS],
        axis=0,
    )
    sim_full = utils_FaIR_JAX.simulate_temp(years=years_full, emissions_by_agent=emis_full, mode=ssp.MODE)
    gmst_full = np.asarray(sim_full["GMST"])
    print(f"simulated GMST: {gmst_full.shape[0]} points, "
          f"range [{gmst_full.min():.3f}, {gmst_full.max():.3f}] K")

    # Baseline emulator's actual predictions vs. simulated truth on the new
    # scenario, using the exact same (now-fixed) feature construction the
    # evaluation script uses.
    X, y_truth, years_feat = ssp.build_new_scenario_features(setup)
    train_s, _test_s, stats = utils_inverse.prepare_baseline_data(
        emis_dict_train=setup["eval_sets"]["Tier 1"],
        emis_dict_test=setup["eval_sets"]["Tier 1"],
        historical_name="historical", mode=ssp.MODE,
    )
    paramsK_base, _losses, _meta = utils_inverse.train_baseline_emulator(
        train_scaled=train_s, key=jax.random.PRNGKey(0),
        K=ssp.BASELINE_HP["K"], lr=ssp.BASELINE_HP["lr"], weight_decay=ssp.BASELINE_HP["weight_decay"],
    )
    Xs = utils_inverse.apply_scaler(X, stats)
    baseline_pred = np.asarray(utils_inverse.mlp_forward(paramsK_base, Xs))
    baseline_truth_arr = np.asarray(y_truth)
    baseline_nrmse = float(utils_inverse._nrmse(jax.numpy.asarray(baseline_pred), jax.numpy.asarray(baseline_truth_arr)))
    print(f"[baseline] ssp370-lowNTCF NRMSE={baseline_nrmse:.4f} (n={len(baseline_pred)})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ssp370_lowntcf_diagnostic.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "agents": ssp.AGENTS,
            "years_hist": years_hist_calendar, "emis_hist_dict": {a: np.asarray(emis_hist_dict[a]) for a in ssp.AGENTS},
            "years_curr": years_curr_calendar, "emis_curr_dict": {a: np.asarray(emis_curr_dict[a]) for a in ssp.AGENTS},
            "years_full": years_full, "gmst_full": gmst_full,
            "years_feat": np.asarray(years_feat),
            "baseline_pred": baseline_pred, "baseline_truth": baseline_truth_arr,
            "baseline_nrmse": baseline_nrmse,
        }, f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
