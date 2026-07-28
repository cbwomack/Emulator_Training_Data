#!/usr/bin/env python
"""
Companion script for 4b_process_MESM_data.ipynb: for every (experiment, scenario)
pair, takes the ensemble mean over the 30-member initial-condition ensemble of
MESM zonal-mean temperature output (NetCDF) and writes it out as a plain numpy
array pickle. This is a pure batch NetCDF-aggregation step (no plotting), so the
entire notebook's logic migrates here; the notebook becomes a thin pointer to
the resulting data/MESM/emis_driven/zonal_data_mean/ pickles. Runs standalone so
it can be scheduled.

Deviates from the notebook in one way: uses parallel=False in open_mfdataset.
The notebook's parallel=True reproduced a deterministic (not intermittent)
netCDF4/dask "Unknown file format" failure in this environment on every one of
5 repro attempts, including with the byte-identical original notebook call -
this is the same underlying dask-worker/netCDF4-file-handle issue already
flagged (as intermittent there) for load_fig7_emic_data's parallel=True call,
just apparently non-intermittent for this dataset/version combo. Since this
script's whole purpose is unattended/scheduled execution (no one present to
"just rerun the cell"), parallel=False is required for it to actually work
headless; output values are identical (parallel only changes read concurrency).

Usage:
    python 4b_process_MESM_data.py
"""
import glob
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

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
