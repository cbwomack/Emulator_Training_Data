#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5, Claude Sonnet 5, and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Companion script for 05e_process_MESM_data.ipynb: for every (experiment, scenario)
pair, takes the ensemble mean over the 30-member initial-condition ensemble of
MESM zonal-mean temperature output (NetCDF) and writes it out as a plain numpy
array pickle. 

Usage:
    python 05e_process_MESM_data.py
"""
import glob
import os
import pickle
import sys
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import xarray as xr

EXPERIMENT_LABELS = ["CS3", "optimized", "Tier 1", "Tier 2", "DECK"]
SCENARIOS = {
    "CS3": ["AA", "CT"],
    "optimized": ["opt_all", "opt_all_sine", "opt_CS3", "opt_DECK", "opt_tier1", "opt_tier2"],
    "Tier 1": ["H-ext", "historical", "L", "M", "ML", "VLHO", "VLLO-ext"],
    "Tier 2": ["H-ext-OS", "M-ext", "ML-ext", "L-ext", "VLHO-ext"],
    "DECK": ["1%-CO2"],
}


def process_scenario(exp_label: str, scen: str) -> None:
    """Ensemble-mean one scenario's zonal-temperature NetCDF files and pickle the result."""
    path = f"data/MESM/emis_driven/zonal_data/{exp_label}/ZONALANN.{scen}*.nc"
    files = sorted(glob.glob(path))
    if not files:
        print(f"No files found for {exp_label}/{scen}, skipping.")
        return

    ds = xr.open_mfdataset(path, combine="nested", concat_dim="member", parallel=False, coords="minimal")
    ensemble_mean = ds["DT2M"].mean(dim="member")
    ensemble_mean_np = ensemble_mean.compute().values

    out_dir = Path(f"data/MESM/emis_driven/zonal_data_mean/{exp_label}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path_np = out_dir / f"{scen}_mean.pkl"
    with open(path_np, "wb") as f:
        pickle.dump(ensemble_mean_np, f)
    print(f"Saved {path_np}")


def main():
    for exp_label in EXPERIMENT_LABELS:
        for scen in SCENARIOS[exp_label]:
            process_scenario(exp_label, scen)


if __name__ == "__main__":
    main()
