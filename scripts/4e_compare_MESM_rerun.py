#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5 and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Compare newly run MESM zonal temperature output against the archived ensemble
in data/MESM/emis_driven/zonal_data/.

data/MESM/ was produced elsewhere and only cached in this repo
(MISSING_DATA.md:87). These runs reproduce it from the IGSM source. MESM has
internal variability, so a rerun cannot match the archive member for member.
The question is whether the new ensemble mean sits inside the spread the
archived ensemble already has.

The test, fixed before looking at any result: at each (year, latitude), the new
ensemble mean must lie within +/- 2 archived member standard deviations of the
archived ensemble mean, across a large majority of points, with no systematic
drift in the time-averaged zonal difference profile. A uniform offset or a
growing trend is a configuration mismatch, not internal variability.

Note that the archived member standard deviation is the spread of single
members. The new ensemble mean is an average over its own members, so it is a
tighter quantity than any single member. The band is therefore a generous test
of "consistent with internal variability", not a tight one, which is the
intent: it is meant to catch configuration errors, not to certify agreement to
the last decimal.

Usage:
    python 4e_compare_MESM_rerun.py \
        --new "data/MESM/rerun/zonal_data/Tier1/ZONALANN.historical_emis.*.nc" \
        --archived "data/MESM/emis_driven/zonal_data/Tier 1/ZONALANN.historical_emis.*.nc" \
        --scenario historical
"""
import argparse
import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from paths import FIGURES_DIR


def load_ensemble(pattern: str) -> xr.Dataset:
    """Open one ensemble as a member-stacked dataset."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern}")
    # parallel=False deliberately: scripts/4b_process_MESM_data.py documents a
    # deterministic netCDF4/dask "Unknown file format" failure with parallel=True
    # in this environment.
    ds = xr.open_mfdataset(
        files, combine="nested", concat_dim="member", parallel=False, coords="minimal"
    )
    print(f"  {len(files)} members from {pattern}")
    return ds


def compare(new: xr.Dataset, archived: xr.Dataset, n_sigma: float, control: xr.Dataset | None = None):
    """Return the comparison arrays, truncated to the years both ensembles cover."""
    n_time = min(new.sizes["time"], archived.sizes["time"])
    if new.sizes["time"] != archived.sizes["time"]:
        print(
            f"  Ensembles differ in length ({new.sizes['time']} vs "
            f"{archived.sizes['time']} years); comparing the first {n_time}."
        )

    new_dt = new["DT2M"].isel(time=slice(0, n_time)).compute()
    arch_dt = archived["DT2M"].isel(time=slice(0, n_time)).compute()

    new_mean = new_dt.mean(dim="member").values

    # Drift correction. Our preindustrial state is not in perfect carbon
    # balance: the ocean carbon initial state replicates surface DIC at every
    # depth, so the deep ocean is under-filled and keeps absorbing. A parallel
    # zero-emissions control carries the same spurious sink and no forcing, so
    # subtracting it isolates the forced response. Standard drift correction;
    # exact to first order, since sink strength depends weakly on the CO2 level
    # the two runs reach.
    if control is not None:
        n_time_c = min(n_time, control.sizes["time"])
        if n_time_c < n_time:
            raise ValueError(
                f"Control is {control.sizes['time']} years, shorter than the "
                f"{n_time} being compared; it cannot correct the whole run."
            )
        ctrl_mean = control["DT2M"].isel(time=slice(0, n_time)).compute().mean(dim="member").values
        new_mean = new_mean - ctrl_mean
    arch_mean = arch_dt.mean(dim="member").values
    # ddof=1: these are samples of a finite ensemble, not a population.
    arch_std = arch_dt.std(dim="member", ddof=1).values

    diff = new_mean - arch_mean
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(arch_std > 0, diff / arch_std, np.nan)

    inside = np.abs(z) <= n_sigma
    valid = np.isfinite(z)

    # The z above is the pre-registered test: the difference of means against
    # the spread of *single* archived members. It is deliberately generous -
    # it is meant to catch configuration errors, not to certify agreement.
    #
    # This second statistic is the strict one: the difference of means against
    # its own standard error, which accounts for both ensembles being finite.
    # If the two ensembles really differ only by internal variability, 95.4% of
    # points should fall within 2 SE. Substantially fewer means a systematic
    # difference that ensemble size alone does not explain.
    se = np.sqrt(
        new_dt.var(dim="member", ddof=1).values / new.sizes["member"]
        + arch_dt.var(dim="member", ddof=1).values / archived.sizes["member"]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z_se = np.where(se > 0, diff / se, np.nan)
    valid_se = np.isfinite(z_se)

    gw = archived["gw"].isel(member=0).values if "member" in archived["gw"].dims else archived["gw"].values
    lat = archived["lat"].values

    return {
        "n_time": n_time,
        "lat": lat,
        "gw": gw,
        "new_mean": new_mean,
        "arch_mean": arch_mean,
        "arch_std": arch_std,
        "diff": diff,
        "z": z,
        "frac_inside": float(inside[valid].mean()),
        "z_se": z_se,
        "frac_inside_se": float((np.abs(z_se) <= 2)[valid_se].mean()),
        "mean_z_se": float(np.nanmean(z_se)),
        "n_new": new.sizes["member"],
        "n_arch": archived.sizes["member"],
    }


def report(r: dict, n_sigma: float) -> None:
    """Print the headline numbers."""
    gw = r["gw"]
    new_gm = r["new_mean"] @ gw
    arch_gm = r["arch_mean"] @ gw
    resid = new_gm - arch_gm

    print()
    print(f"Years compared            : {r['n_time']}")
    print(f"Members (new / archived)  : {r['n_new']} / {r['n_arch']}")
    print(f"Points within {n_sigma:g} sigma    : {100 * r['frac_inside']:.1f}%  "
          f"(pre-registered test, vs single-member spread)")
    print(f"Points within 2 SE        : {100 * r['frac_inside_se']:.1f}%  "
          f"(strict test, vs the standard error of the difference; "
          f"expect 95.4%, mean z {r['mean_z_se']:+.2f})")

    # Trends are independent of the DT2M baseline, so they compare the forced
    # response itself rather than the two runs' preindustrial states.
    lat = r["lat"]
    yrs = np.arange(r["n_time"])
    tr_new = np.polyfit(yrs, r["new_mean"], 1)[0] * 100
    tr_arch = np.polyfit(yrs, r["arch_mean"], 1)[0] * 100
    def hemi(t, south):
        m = (lat < 0) if south else (lat > 0)
        return t[m] @ gw[m] / gw[m].sum()
    print(f"Warming trend (K/century) : new {tr_new @ gw:+.4f}, archived {tr_arch @ gw:+.4f}")
    print(f"  SH  new {hemi(tr_new, True):+.4f}, archived {hemi(tr_arch, True):+.4f}")
    print(f"  NH  new {hemi(tr_new, False):+.4f}, archived {hemi(tr_arch, False):+.4f}")
    print(f"Zonal difference          : mean {r['diff'].mean():+.4f} K, "
          f"max |.| {np.abs(r['diff']).max():.4f} K")
    print(f"Global-mean DT2M residual : mean {resid.mean():+.4f} K, "
          f"RMSE {np.sqrt((resid ** 2).mean()):.4f} K")
    print(f"Global-mean DT2M, final year: new {new_gm[-1]:+.4f} K, "
          f"archived {arch_gm[-1]:+.4f} K")

    # A drift check: the slope of the global-mean residual over the run.
    years = np.arange(r["n_time"])
    slope = np.polyfit(years, resid, 1)[0]
    print(f"Residual drift            : {slope * 100:+.4f} K per century of run")


def plot(r: dict, scenario: str, n_sigma: float, out_path: Path) -> None:
    """Draw the difference field, the sigma-normalised field, the global means and the zonal profile."""
    gw, lat = r["gw"], r["lat"]
    years = np.arange(1, r["n_time"] + 1)
    new_gm = r["new_mean"] @ gw
    arch_gm = r["arch_mean"] @ gw

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    lim = float(np.abs(r["diff"]).max())
    m = axes[0, 0].pcolormesh(years, lat, r["diff"].T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    axes[0, 0].set(title="New minus archived ensemble mean", xlabel="model year", ylabel="latitude")
    fig.colorbar(m, ax=axes[0, 0], label="$\\Delta$T2M difference (K)")

    m = axes[0, 1].pcolormesh(years, lat, r["z"].T, cmap="RdBu_r", vmin=-3, vmax=3, shading="auto")
    axes[0, 1].set(title=f"Difference in archived member sigma ({100 * r['frac_inside']:.1f}% within {n_sigma:g})",
                   xlabel="model year", ylabel="latitude")
    fig.colorbar(m, ax=axes[0, 1], label="difference / archived $\\sigma$")

    arch_gm_sigma = (r["arch_std"] @ gw)
    axes[1, 0].fill_between(years, arch_gm - n_sigma * arch_gm_sigma, arch_gm + n_sigma * arch_gm_sigma,
                            alpha=0.25, label=f"archived $\\pm{n_sigma:g}\\sigma$")
    axes[1, 0].plot(years, arch_gm, label="archived mean")
    axes[1, 0].plot(years, new_gm, "--", label="new mean")
    axes[1, 0].set(title="Global-mean $\\Delta$T2M", xlabel="model year", ylabel="K")
    axes[1, 0].legend()

    prof = r["diff"].mean(axis=0)
    prof_sigma = r["arch_std"].mean(axis=0)
    axes[1, 1].fill_between(lat, -n_sigma * prof_sigma, n_sigma * prof_sigma, alpha=0.25,
                            label=f"archived $\\pm{n_sigma:g}\\sigma$")
    axes[1, 1].plot(lat, prof, color="k", label="time-mean difference")
    axes[1, 1].axhline(0, lw=0.8, color="0.5")
    axes[1, 1].set(title="Time-mean zonal difference", xlabel="latitude", ylabel="K")
    axes[1, 1].legend()

    fig.suptitle(f"MESM rerun vs archive: {scenario}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", required=True, help="glob for the new ZONALANN files")
    parser.add_argument("--archived", required=True, help="glob for the archived ZONALANN files")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--n-sigma", type=float, default=2.0)
    parser.add_argument(
        "--control",
        default=None,
        help=(
            "glob for a parallel zero-emissions control ensemble, same length and "
            "same initial state. If given, its ensemble-mean DT2M is subtracted "
            "from the new ensemble mean to remove the drift of our preindustrial "
            "state before comparing."
        ),
    )
    parser.add_argument("--out", default=None, help="figure path (default: Figures/mesm_rerun_<scenario>.png)")
    args = parser.parse_args()

    print("Loading ensembles:")
    new = load_ensemble(args.new)
    archived = load_ensemble(args.archived)
    control = load_ensemble(args.control) if args.control else None
    if control is not None:
        print("  drift-correcting the new ensemble against the control")

    r = compare(new, archived, args.n_sigma, control)
    report(r, args.n_sigma)

    out = Path(args.out) if args.out else FIGURES_DIR / f"mesm_rerun_{args.scenario}.png"
    plot(r, args.scenario, args.n_sigma, out)


if __name__ == "__main__":
    main()
