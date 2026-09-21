#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Ensemble-mean the optimized MESM runs .

Usage:
    python 05f_process_MESM_optimized_data.py                 # check, then write
    python 05f_process_MESM_optimized_data.py --check-only    # check, write nothing
    python 05f_process_MESM_optimized_data.py --scenario sine

"""
import argparse
import glob
import os
import pickle
import sys
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import numpy as np
import xarray as xr

RAW_DIR = "data/MESM/emis_driven/zonal_data/optimized"
OUT_DIR = Path("data/MESM/emis_driven/zonal_data_mean/optimized")
SCENARIOS = ["constant", "sine", "gaussian"]
N_MEMBERS = 30


def _zonal_files(ic: str) -> list[str]:
    return sorted(glob.glob(f"{RAW_DIR}/ZONALANN.opt_all_revision_{ic}.*.nc"))


def _globalmean_files(ic: str) -> list[str]:
    return sorted(glob.glob(f"{RAW_DIR}/GLOBALMEAN.opt_all_revision_{ic}.*"))


def ensemble_mean(ic: str) -> tuple[np.ndarray, np.ndarray]:
    """(ensemble-mean DT2M (T, 46), Gaussian area weights (46,)) for one scenario."""
    files = _zonal_files(ic)
    if len(files) != N_MEMBERS:
        raise SystemExit(
            f"{ic}: expected {N_MEMBERS} ZONALANN members, found {len(files)} - "
            f"refusing to average an incomplete or over-matched ensemble."
        )

    ds = xr.open_mfdataset(files, combine="nested", concat_dim="member",
                           parallel=False, coords="minimal")
    try:
        mean_np = ds["DT2M"].mean(dim="member").compute().values
        # gw is identical across members (fixed model grid); take member 0's.
        gw = ds["gw"].isel(member=0).compute().values
    finally:
        ds.close()
    return mean_np, gw


def globalmean_truth(ic: str) -> np.ndarray:
    """Ensemble mean of MESM's own ascii global-mean DT2M series, (T,)."""
    files = _globalmean_files(ic)
    if len(files) != N_MEMBERS:
        raise SystemExit(
            f"{ic}: expected {N_MEMBERS} GLOBALMEAN members, found {len(files)}."
        )
    # Column 2 is DT2M (column 0 = model year, column 1 = absolute T2M).
    return np.stack([np.loadtxt(f, usecols=(2,)) for f in files]).mean(axis=0)


def check_scenario(ic: str, mean_np: np.ndarray, gw: np.ndarray, max_error: float) -> float:
    """Contract the zonal ensemble mean with `gw` and compare to MESM's own
    global-mean output. Returns the max abs difference; raises above max_error."""
    truth = globalmean_truth(ic)
    n = min(mean_np.shape[0], truth.shape[0])
    err = float(np.abs(mean_np[:n] @ gw - truth[:n]).max())
    if err > max_error:
        raise SystemExit(
            f"{ic}: gw-weighted zonal mean disagrees with MESM's GLOBALMEAN output by "
            f"{err:.6f} K (limit {max_error}). Likely causes: the ZONALANN and "
            f"GLOBALMEAN globs picked up different runs, or the wrong diagnostic column."
        )
    return err


def cos_weight_offset(mean_np: np.ndarray, gw: np.ndarray) -> float:
    """Max difference between the gw-weighted global mean and the cos(lat) weighting
    the emulator path uses"""
    w = np.cos(np.deg2rad(np.linspace(-88, 88, mean_np.shape[1])))
    w = np.maximum(w, 1e-6)
    w = w / np.sum(w)
    return float(np.abs(mean_np @ w - mean_np @ gw).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=SCENARIOS, default=None,
                        help="process one scenario; default is all three")
    parser.add_argument("--check-only", action="store_true",
                        help="run the GLOBALMEAN consistency check and report, write nothing")
    parser.add_argument("--max-weighting-error", type=float, default=1e-3,
                        help="abort if the gw-weighted zonal mean and MESM's own "
                             "global mean differ by more than this many K (default 1e-3)")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else SCENARIOS

    for ic in scenarios:
        mean_np, gw = ensemble_mean(ic)
        err = check_scenario(ic, mean_np, gw, args.max_weighting_error)
        cos_off = cos_weight_offset(mean_np, gw)

        print(f"[{ic}] {N_MEMBERS} members, shape {mean_np.shape} {mean_np.dtype}")
        print(f"    gw-weighted vs MESM GLOBALMEAN : max |diff| = {err:.6f} K  (limit {args.max_weighting_error})")
        print(f"    cos(lat) vs gw weighting       : max |diff| = {cos_off:.6f} K  (informational)")
        print(f"    final-year global mean         : gw {mean_np[-1] @ gw:.4f} K")

        if args.check_only:
            continue

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"opt_all_{ic}_mean.pkl"
        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(mean_np, f)
        os.replace(tmp, out_path)
        print(f"    saved {out_path}")


if __name__ == "__main__":
    main()
