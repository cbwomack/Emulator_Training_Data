#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
SCM-vs-MESM fidelity: runs the MESM-calibrated SCM forward on the CO2-only
Tier1/Tier2/DECK/CS3 emissions and compares its global mean surface temperature
against MESM's own zonal output, area-weighted to a global mean. Reports NRMSE
and R^2 per scenario, per scenario set, and overall, and caches the panels for
the supplement figure.

Usage:
    python 07e_scm_mesm_fidelity.py            # metrics table
    python 07e_scm_mesm_fidelity.py --panels   # panel cache for SI_SCM_MESM
"""
import os
import sys
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import argparse
import pickle

import numpy as np
import jax.numpy as jnp

import utils_FaIR_JAX
import utils_inverse
from paths import DATA_DIR

EVAL_DIR = "data/MESM/emis_driven/zonal_data_mean/"
AGENTS = ["CO2"]
MODE = "MESM_tier1"
OUT_DIR = Path("data/SI_results/scm_mesm_fidelity")

SCENARIOS_EVAL = {
    "Tier 1": ["historical", "H-ext", "L", "M", "ML", "VLHO", "VLLO-ext"],
    "Tier 2": ["H-ext-OS", "M-ext", "ML-ext", "L-ext", "VLHO-ext"],
    "DECK": ["1pctCO2", "2xCO2"],
    "CS3": ["AA", "CT", "historical"],
}


def build_eval_sets():
    """Same construction as the MESM evaluation path's build_eval_sets
    (CO2-only emissions + MESM zonal targets for Tier1/Tier2/DECK/CS3),
    duplicated rather than imported since that module's main() has side
    effects (trains an MLP) this script doesn't want."""
    eval_emis_sets, emis_dict_tier1_JAX, emis_dict_tier2_JAX, emis_dict_CS3_JAX, emis_dict_all_JAX = (
        utils_inverse.generate_eval_data(AGENTS, DECK=False, CS3=True, DAMIP=False, GeoMIP=False)
    )
    eval_targets_sets, *_rest, lat_coords = utils_inverse.generate_target_data(SCENARIOS_EVAL, data_dir=EVAL_DIR)

    emis_path = str(DATA_DIR / "MESM" / "emis_driven")
    emis_1pct = np.loadtxt(f"{emis_path}/1PRCO2/carbemiss.txt", usecols=(2,), skiprows=2)
    emis_mat_1pct = np.zeros((5, len(emis_1pct)))
    emis_mat_1pct[0, :] = emis_1pct

    emis_2xCO2 = np.loadtxt(f"{emis_path}/2xCO2/implco2emiss.3100.25.txt", usecols=(2,))
    emis_mat_2xCO2 = np.zeros((5, len(emis_2xCO2)))
    emis_mat_2xCO2[0, :] = emis_2xCO2

    eval_emis_sets["DECK"] = {"1pctCO2": emis_mat_1pct, "2xCO2": emis_mat_2xCO2}

    return eval_emis_sets, eval_targets_sets, lat_coords


def scm_gmst_by_scenario(emis_dict, mode: str = MODE):
    """{scenario: GMST array}, SCM-simulated, historical-context handling
    identical to the existing baseline-training path
    (utils_inverse.build_dataset_from_runfair_dict -> simulate_targets_gmst)."""
    raw = utils_inverse.build_dataset_from_runfair_dict(emis_dict, agents=AGENTS, mode=mode)
    return {scen: np.asarray(y) for (_X, y, scen) in raw}


def mesm_global_truth_by_scenario(emis_dict, targets_dict, lat_coords):
    """{scenario: area-weighted global-mean MESM truth array}, using the
    existing, already-in-production crop/align logic
    (utils_inverse.build_dataset_vector_targets)."""
    weights = np.cos(np.deg2rad(lat_coords))
    weights = np.maximum(weights, 1e-6)
    weights = weights / np.sum(weights)

    scens = list(emis_dict.keys())
    raw = utils_inverse.build_dataset_vector_targets(emis_dict, targets_dict, scens, agents=AGENTS)
    out = {}
    for (_X, y_target, scen) in raw:
        y_target = np.asarray(y_target)
        out[scen] = np.sum(y_target * weights[None, :], axis=1)
    return out


def nrmse_r2(pred, truth):
    n = min(len(pred), len(truth))
    pred, truth = pred[:n], truth[:n]
    rmse = np.sqrt(np.mean((pred - truth) ** 2))
    nrmse = rmse / (np.max(np.abs(truth)) + 1e-8)
    ss_res = np.sum((truth - pred) ** 2)
    ss_tot = np.sum((truth - np.mean(truth)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return float(nrmse), float(r2), n


def main():
    params = utils_FaIR_JAX.PARAMS_BY_MODE[MODE]
    eval_emis_sets, eval_targets_sets, lat_coords = build_eval_sets()

    results = {}
    for set_name, emis_d in eval_emis_sets.items():
        targets_d = eval_targets_sets.get(set_name)
        if targets_d is None:
            print(f"[{set_name}] no MESM targets found, skipping")
            continue

        scm_gmst = scm_gmst_by_scenario(emis_d)
        mesm_truth = mesm_global_truth_by_scenario(emis_d, targets_d, lat_coords)

        set_results = {}
        for scen in emis_d:
            if set_name not in ("Tier 1", "All") and scen == "historical":
                continue 
            if scen not in scm_gmst or scen not in mesm_truth:
                print(f"[{set_name}/{scen}] missing SCM or MESM data, skipping")
                continue
            nrmse, r2, n = nrmse_r2(scm_gmst[scen], mesm_truth[scen])
            if set_name == "DECK" and scen == "2xCO2":
                scm_diag = _scm_2xco2_conc_driven(params, n)[:n]
                nrmse, r2, n = nrmse_r2(scm_diag, np.asarray(mesm_truth[scen])[:n])
            set_results[scen] = {"nrmse": nrmse, "r2": r2, "n_years": n}
            print(f"[{set_name}/{scen}] NRMSE={nrmse:.4f} R2={r2:.4f} (n={n} years)")

        if set_results:
            set_results["mean"] = {
                "nrmse": float(np.mean([v["nrmse"] for v in set_results.values()])),
                "r2": float(np.mean([v["r2"] for v in set_results.values()])),
            }
            print(f"[{set_name}] mean NRMSE={set_results['mean']['nrmse']:.4f} "
                  f"mean R2={set_results['mean']['r2']:.4f}")
        results[set_name] = set_results

    all_nrmse = [v["nrmse"] for s in results.values() for k, v in s.items() if k != "mean"]
    all_r2 = [v["r2"] for s in results.values() for k, v in s.items() if k != "mean"]
    overall = {"nrmse": float(np.mean(all_nrmse)), "r2": float(np.mean(all_r2))} if all_nrmse else None
    if overall:
        print(f"[overall] mean NRMSE={overall['nrmse']:.4f} mean R2={overall['r2']:.4f}")

    opt_metrics = optimized_scm_vs_mesm(lat_coords)
    opt_results = {}
    for ic, m in opt_metrics.items():
        opt_results[ic] = {"nrmse": m["nrmse"], "r2": m["r2"], "n_years": m["n_years"]}
        print(f"[Optimized/{ic}] NRMSE={m['nrmse']:.4f} R2={m['r2']:.4f} "
              f"(n={m['n_years']} years)")
    if opt_results:
        opt_results["mean"] = {
            "nrmse": float(np.mean([v["nrmse"] for v in opt_results.values()])),
            "r2": float(np.mean([v["r2"] for v in opt_results.values()])),
        }
        print(f"[Optimized] mean NRMSE={opt_results['mean']['nrmse']:.4f} "
              f"mean R2={opt_results['mean']['r2']:.4f}")
    results["Optimized"] = opt_results

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "scm_mesm_fidelity_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"per_scenario_set": results, "overall": overall}, f)
    print(f"wrote {out_path}")


# --- Panel cache for the supplement figure --------------------------------

PANEL_CACHE = OUT_DIR / "si_scm_mesm_panels.pkl"
CALENDAR_SETS = {"Tier 1", "Tier 2", "CS3"}
OPT_ICS = ["constant", "sine"]
BASE_CO2 = 286.4  # MESM preindustrial baseline, utils_FaIR_JAX.MESM_PARAMS['C0_PI']


def _scm_2xco2_conc_driven(params, n_years: int) -> np.ndarray:
    """2xCO2 GMST under prescribed concentration, bypassing the carbon cycle.
    """
    conc = jnp.zeros((3, n_years)).at[0, :].set(2 * BASE_CO2).at[1, :].set(720.0).at[2, :].set(270.0)
    emis0 = jnp.zeros((5, n_years))
    years = jnp.arange(n_years, dtype=jnp.float32)
    return np.asarray(utils_FaIR_JAX.simulate_temp_prescribed_conc(years, conc, emis0, params, dt=0.1))


def optimized_scm_vs_mesm(lat_coords) -> dict:
    """SCM-vs-MESM fidelity on the optimized trajectories MESM was driven with.
    """
    emis = {}
    for ic in OPT_ICS:
        e = np.loadtxt(f"{DATA_DIR}/MESM/emis_driven/MESM_inputs/opt_all_{ic}.txt",
                       usecols=(2,), skiprows=2)
        mat = np.zeros((5, len(e)))
        mat[0, :] = e
        emis[f"opt_{ic}"] = mat

    _, targets, _, _ = utils_inverse.generate_target_data(
        {"optimized": [f"all_{ic}" for ic in OPT_ICS]}, data_dir=EVAL_DIR, opt=True)

    weights = np.cos(np.deg2rad(lat_coords))
    weights = np.maximum(weights, 1e-6)
    weights = weights / np.sum(weights)

    scm = scm_gmst_by_scenario(emis)
    out = {}
    for ic in OPT_ICS:
        truth = np.sum(targets[f"all_{ic}"] * weights[None, :], axis=1)
        nrmse, r2, n = nrmse_r2(np.asarray(scm[f"opt_{ic}"]), np.asarray(truth))
        out[ic] = dict(nrmse=nrmse, r2=r2, n_years=n,
                       scm=np.asarray(scm[f"opt_{ic}"])[:n],
                       mesm=np.asarray(truth)[:n])
    return out


def _optimized_panels(lat_coords) -> list[dict]:
    """One panel per optimized initial guess, the figure's (q) and (r)."""
    display = {"constant": r"Constant\ initial\ guess", "sine": r"Sine\ initial\ guess"}
    metrics = optimized_scm_vs_mesm(lat_coords)
    return [dict(set_name="Optimized", scenario=display[ic],
                 x=np.arange(m["n_years"]), xlabel="Simulation year",
                 scm=m["scm"], mesm=m["mesm"], nrmse=m["nrmse"], r2=m["r2"])
            for ic, m in metrics.items()]


def build_panels():
    """Panels for SI_SCM_MESM.pdf: the 16 evaluation scenarios followed by the
    two optimized initial guesses. Written to PANEL_CACHE."""
    params = utils_FaIR_JAX.PARAMS_BY_MODE[MODE]
    eval_emis_sets, eval_targets_sets, lat_coords = build_eval_sets()

    panels = []
    for set_name in ["Tier 1", "Tier 2", "DECK", "CS3"]:
        targets_d = eval_targets_sets.get(set_name)
        if targets_d is None:
            continue
        scm = scm_gmst_by_scenario(eval_emis_sets[set_name])
        truth = mesm_global_truth_by_scenario(eval_emis_sets[set_name], targets_d, lat_coords)
        for scen, scm_arr in scm.items():
            if scen == "historical" and set_name != "Tier 1":
                continue
            truth_arr = truth.get(scen)
            if truth_arr is None:
                continue
            nrmse, r2, n = nrmse_r2(np.asarray(scm_arr), np.asarray(truth_arr))
            if set_name in CALENDAR_SETS:
                start = 1750 if scen == "historical" else 2024
                x, xlabel = np.arange(start, start + n), "Year"
            else:
                x, xlabel = np.arange(n), "Simulation year"
            panel = dict(set_name=set_name, scenario=scen, x=x, xlabel=xlabel,
                         scm=np.asarray(scm_arr)[:n], mesm=np.asarray(truth_arr)[:n],
                         nrmse=nrmse, r2=r2)
            if set_name == "DECK" and scen == "2xCO2":
                scm_diag = _scm_2xco2_conc_driven(params, n)[:n]
                panel["nrmse"], panel["r2"], _ = nrmse_r2(scm_diag, panel["mesm"])
                panel["scm"] = scm_diag
            panels.append(panel)

    panels += _optimized_panels(lat_coords)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PANEL_CACHE, "wb") as f:
        pickle.dump(panels, f)
    print(f"wrote {PANEL_CACHE} ({len(panels)} panels)")
    return panels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", action="store_true",
                        help="Build the SI_SCM_MESM panel cache instead of the metrics table.")
    if parser.parse_args().panels:
        build_panels()
    else:
        main()
