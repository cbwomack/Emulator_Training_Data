#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage C comparison: what did extending 1000 -> 2000 outer iterations buy?

Because the extension RESUMED in place, every 2000-iteration checkpoint still
contains its own 1000-iteration history as a prefix. So arm 1 (1000 steps) and
arm 2 (2000 steps) are read from the same file and are paired by construction -
same seed, same init, same trajectory up to k=1000. That makes the per-seed
paired difference the right statistic, and it is what this reports.

Reports TRUE NRMSE, not the recorded objective: `errors` stores
NRMSE + smoothness_weight * penalty, which for CO2 (w=1e-6) is a ~1.4% overstatement
and for N2O/BC far larger. recover_nrmse_trajectory subtracts it exactly.

This is the IN-SAMPLE training curve - it falls by construction and is not
evidence that the emulator generalizes better. 6h_convergence_preflight.py is
the out-of-sample measure. Both are reported in the manuscript for a reason;
do not quote this one alone.

Usage:
    python scripts/6h_extension_comparison.py --family co2
    python scripts/6h_extension_comparison.py --family all-available
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import utils_inverse

FAMILIES = {
    "co2":        ("checkpoints/co2_retuned/seed_sweep", "co2_only"),
    "CH4":        ("checkpoints/CH4_retuned/seed_sweep", "CH4_only"),
    "N2O":        ("checkpoints/N2O_retuned/seed_sweep", "N2O_only"),
    "Sulfur":     ("checkpoints/Sulfur_retuned/seed_sweep", "Sulfur_only"),
    "BC":         ("checkpoints/BC_retuned/seed_sweep", "BC_only"),
    "multi":      ("checkpoints/multi_retuned/seed_sweep", "all_agents"),
    "multi_fig4": ("checkpoints/multi_fig4/seed_sweep", "multi_fig4"),
}


def collect(cdir: str, tag: str, k_old: int, k_new: int):
    """{group: {seed: (nrmse_at_k_old, nrmse_at_k_new)}} from resumed checkpoints."""
    out, skipped = {}, []
    for p in sorted(Path(cdir).glob("*.pkl")):
        name = p.name
        if name.startswith("baseline"):
            continue  # baseline emulator, no U trajectory
        # inverse_{init}_{group}_{tag}_seed{N}.pkl
        stem = name[len("inverse_"):-len(".pkl")]
        if f"_{tag}_seed" not in stem:
            continue
        head, seed_s = stem.split(f"_{tag}_seed")
        init, group = head.split("_", 1)
        try:
            with open(p, "rb") as f:
                raw = pickle.load(f)
            nrmse = utils_inverse.recover_nrmse_trajectory(raw)
        except Exception as e:                      # noqa: BLE001
            skipped.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if len(nrmse) <= k_new:
            skipped.append(f"{name}: only {len(nrmse)} entries (not yet extended)")
            continue
        out.setdefault(f"{init}/{group}", {})[int(seed_s)] = (
            float(nrmse[k_old]), float(nrmse[k_new]))
    return out, skipped


def report(family: str, data: dict, k_old: int, k_new: int):
    print(f"\n=== {family}: in-sample NRMSE, iteration {k_old} -> {k_new} ===")
    print(f"{'group':18s} {'n':>3s} {'med@'+str(k_old):>10s} {'med@'+str(k_new):>10s} "
          f"{'med Δ%':>8s} {'IQR Δ%':>16s} {'improved':>9s}")
    for group in sorted(data):
        seeds = sorted(data[group])
        old = np.array([data[group][s][0] for s in seeds])
        new = np.array([data[group][s][1] for s in seeds])
        rel = (new - old) / old * 100          # paired, per seed
        n_imp = int((rel < 0).sum())
        print(f"{group:18s} {len(seeds):3d} {np.median(old):10.5f} {np.median(new):10.5f} "
              f"{np.median(rel):+8.2f} "
              f"[{np.percentile(rel,25):+6.2f},{np.percentile(rel,75):+6.2f}] "
              f"{n_imp:4d}/{len(seeds):<4d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="co2",
                    choices=list(FAMILIES) + ["all-available"])
    ap.add_argument("--k-old", type=int, default=1000)
    ap.add_argument("--k-new", type=int, default=2000)
    args = ap.parse_args()

    fams = list(FAMILIES) if args.family == "all-available" else [args.family]
    any_data = False
    for fam in fams:
        cdir, tag = FAMILIES[fam]
        if not Path(cdir).is_dir():
            print(f"[{fam}] {cdir} missing - skipping")
            continue
        data, skipped = collect(cdir, tag, args.k_old, args.k_new)
        if not data:
            print(f"[{fam}] no extended checkpoints yet "
                  f"({len(skipped)} files skipped)")
            continue
        any_data = True
        report(fam, data, args.k_old, args.k_new)
        if skipped:
            print(f"  ({len(skipped)} skipped; first: {skipped[0]})")
    if not any_data:
        sys.exit("no extended checkpoints found for any requested family")


if __name__ == "__main__":
    main()
