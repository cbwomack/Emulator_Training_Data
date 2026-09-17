#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5 and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Convert one MESM run's zonal-temperature ascii output into a ZONALANN NetCDF
file matching the archived ensemble in data/MESM/emis_driven/zonal_data/.

The IGSM source has no zonal NetCDF writer - `grep -i zonalann` over the whole
tree returns nothing, and the only analysis tool, util/analysis/antsrf.f90,
writes a global-mean ascii series. The archived ZONALANN files were produced by
a tool this project does not have. This script is the second half of the
replacement: `antzon` (util/analysis/antzon.f90, antsrf widened from the global
column to all 46 latitude bands) writes the ascii, and this turns it into
NetCDF with the archive's exact structure.

The ascii has one line per year:

    <year> <global mean> <band 1> ... <band 46>

`lat` and `gw` are taken from the archived files rather than recomputed, so the
geometry matches exactly.

The DT2M baseline needs more care. By default it too comes from the archive,
recovered as the mean of (T2M - DT2M) over the reference ensemble's members and
years: that difference is a fixed preindustrial climatology per latitude,
carrying only float32-level scatter (standard deviation 0.001-0.004 K per band,
against a signal of ~0.5 K).

But our runs start from our own preindustrial spin-up, not the one the archive
used, so our absolute preindustrial climate is close to but not the same as
theirs. Subtracting *their* climatology would fold that difference into DT2M as
a uniform offset, and the comparison would read it as a configuration error when
it is only a different starting state. Pass `--baseline-ascii` pointing at the
spin-up's own antzon output to define DT2M against our own climatology instead;
the offset between the two is printed either way.

Usage:
    python 4d_mesm_zonal_to_netcdf.py \
        --ascii /orcd/pool/005/cwomack/MESM/runs/2151.90/zonal.2151.90.txt \
        --scenario historical --run 2151 --yy 90 \
        --out-dir data/MESM/rerun/zonal_data/Tier1
"""
import argparse
import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import xarray as xr

N_LAT = 46

# Any archived scenario works as the geometry/baseline reference: the
# preindustrial climatology agrees between scenarios to <0.1 K per band.
DEFAULT_REFERENCE = "data/MESM/emis_driven/zonal_data/Tier 1/ZONALANN.historical_emis.*.nc"


def load_reference(pattern: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lat, gw, baseline) from the archived ensemble matching `pattern`."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No reference files match {pattern}")

    with xr.open_dataset(files[0]) as ds:
        lat = ds["lat"].values.astype("float32")
        gw = ds["gw"].values.astype("float32")

    baselines = []
    for path in files:
        with xr.open_dataset(path) as ds:
            baselines.append((ds["T2M"] - ds["DT2M"]).values)
    baseline = np.stack(baselines).mean(axis=(0, 1)).astype("float32")

    return lat, gw, baseline


def read_antzon(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (years, global_mean, bands) from an antzon ascii file."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw[None, :]
    expected = 2 + N_LAT
    if raw.shape[1] != expected:
        raise ValueError(
            f"{path}: expected {expected} columns (year, global, {N_LAT} bands), "
            f"got {raw.shape[1]}"
        )
    return raw[:, 0].astype(int), raw[:, 1], raw[:, 2:]


def check_weighting(global_col: np.ndarray, bands: np.ndarray, gw: np.ndarray) -> float:
    """Return the largest disagreement between the area-weighted bands and the model's own global column."""
    return float(np.abs(bands @ gw - global_col).max())


def own_baseline(pattern: str, n_years: int, window: str) -> np.ndarray:
    """Return the per-latitude preindustrial climatology from one or more control runs.

    `pattern` may name a single file or glob several members of a control
    ensemble, which are averaged together. `window` selects which years to
    average: "all" for the whole run, "first"/"last" for the first or last
    `n_years`.

    Which to use depends on what the reference run is. A contemporaneous
    fixed-CO2 control shares the forced run's starting state and drift, so its
    whole length is the right average ("all"). A spin-up is different: ours is
    still cooling ~0.05 K/century at year 500, so a long window is centred
    further back in time and reads warm - take only its tail.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No baseline files match {pattern}")

    per_member = []
    for path in files:
        _, _, bands = read_antzon(path)
        if window == "all":
            sel = bands
        else:
            if bands.shape[0] < n_years:
                raise ValueError(
                    f"{path}: only {bands.shape[0]} years, need {n_years} for the baseline"
                )
            sel = bands[:n_years] if window == "first" else bands[-n_years:]
        per_member.append(sel.mean(axis=0))

    n_yr = "all" if window == "all" else f"{window} {n_years}"
    print(f"Baseline: {len(files)} run(s) matching {pattern}, averaging {n_yr} years")
    return np.stack(per_member).mean(axis=0).astype("float32")


def build_dataset(bands, lat, gw, baseline, scenario) -> xr.Dataset:
    """Assemble the ZONALANN dataset in the archive's exact structure."""
    t2m = bands.astype("float32")
    dt2m = (t2m - baseline[None, :]).astype("float32")
    time = np.arange(1, t2m.shape[0] + 1, dtype="float32")

    return xr.Dataset(
        data_vars={
            "gw": ("lat", gw),
            "T2M": (("time", "lat"), t2m),
            "DT2M": (("time", "lat"), dt2m),
        },
        coords={"time": time, "lat": lat},
        attrs={"case": f"{scenario}_emis"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ascii", required=True, help="antzon output file")
    parser.add_argument("--scenario", required=True, help="e.g. historical, H-ext")
    parser.add_argument("--run", required=True, help="run number, e.g. 2151")
    parser.add_argument("--yy", required=True, help="run-family tag, e.g. 90")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--baseline-ascii",
        default=None,
        help=(
            "antzon output from this configuration's own preindustrial spin-up. "
            "If given, DT2M is defined against its last --baseline-years mean "
            "instead of the archive's climatology. Preferred: our spin-up is not "
            "the one the archive used, so an absolute climate offset between the "
            "two would otherwise show up in DT2M as a uniform bias and be "
            "misread as a configuration error."
        ),
    )
    parser.add_argument("--baseline-years", type=int, default=100)
    parser.add_argument(
        "--baseline-window",
        choices=("all", "first", "last"),
        default="last",
        help=(
            "Which years of the baseline run to average. 'all' for a "
            "contemporaneous fixed-CO2 control; 'last' for a spin-up tail "
            "(default, since a spin-up that is still drifting reads warm over a "
            "long window)."
        ),
    )
    parser.add_argument(
        "--max-weighting-error",
        type=float,
        default=1e-3,
        help="Fail if the area-weighted bands disagree with the model's global column by more than this (K).",
    )
    args = parser.parse_args()

    lat, gw, arch_baseline = load_reference(args.reference)
    _, global_col, bands = read_antzon(args.ascii)

    baseline = arch_baseline
    if args.baseline_ascii:
        baseline = own_baseline(
            args.baseline_ascii, args.baseline_years, args.baseline_window
        )
        offset = baseline - arch_baseline
        # Report it rather than hide it: a large offset means our preindustrial
        # climate genuinely differs from the archive's, which is worth knowing
        # even though it is removed from DT2M.
        print(
            f"Offset of that baseline from the archive's climatology: "
            f"global {offset @ gw:+.4f} K, max |per band| {np.abs(offset).max():.4f} K"
        )

    err = check_weighting(global_col, bands, gw)
    if err > args.max_weighting_error:
        raise SystemExit(
            f"Area-weighted bands disagree with the model's own global column by "
            f"{err:.4g} K (limit {args.max_weighting_error:.4g}). The wrong "
            f"diagnostic index (ivr) or the wrong scaling is the usual cause."
        )
    print(f"Area-weighting check: max |weighted bands - global column| = {err:.3e} K")

    ds = build_dataset(bands, lat, gw, baseline, args.scenario)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"ZONALANN.{args.scenario}_emis.{args.run}.{args.yy}."
        f"{1:04d}-{ds.sizes['time']:04d}.nc"
    )
    out_path = out_dir / name
    ds.to_netcdf(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
