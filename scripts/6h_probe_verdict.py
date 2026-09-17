#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Does the Figure 3 "now beats baseline" claim survive OUT OF SAMPLE?

Figure 3 plots the bilevel objective in-sample on the group being optimized, so
it falls by construction; extending 1000 -> 2000 improves it whether or not the
emulator generalizes better. This reads the sharded out-of-sample probe and
answers the two questions that actually matter:

  Q1  Did out-of-sample error improve from iteration 1000 to 2000?
  Q2  At iteration 2000, does the optimized emulator beat its baseline?

Both are answered with PAIRED per-seed statistics. Each seed carries its own
init and its own baseline, so pairing removes seed-level level differences,
which here are much larger than the within-seed effect being measured.
Differencing medians of unpaired curves instead inverted two of ten verdicts on
an earlier 5-seed run - see REVISIONS.md 2026-08-25.

Reads the shard pickles directly rather than preflight_results.pkl so it does
not depend on the collect step having been run.

Usage:
    python scripts/6h_probe_verdict.py
    python scripts/6h_probe_verdict.py --shard-dir <dir> --k-old 1000 --k-new 2000
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

TARGET_SET = "Tier 1"


def load_shards(shard_dir: Path):
    pooled = {}
    for f in sorted(shard_dir.glob("*.pkl")):
        with open(f, "rb") as fh:
            sh = pickle.load(fh)
        pooled.setdefault(sh["family"], {}).update(sh["raw"])
    if not pooled:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    return pooled


def _sign_note(n_better, n):
    """Flag results a 10-seed sample cannot separate from a coin flip."""
    return "" if (n_better == n or n_better == 0) else (
        "  (split)" if abs(n_better - n / 2) <= 1 else "")


def report(family, raw, k_old, k_new):
    seeds = sorted(raw)
    names = sorted(k for k in raw[seeds[0]][k_new])
    print(f"\n=== {family}: OUT-OF-SAMPLE NRMSE, n={len(seeds)} seeds ===")
    print(f"{'eval set':10s} {'@'+str(k_old):>9s} {'@'+str(k_new):>9s} {'Δ%':>8s} "
          f"{'better':>7s} | {'baseline':>9s} {'vs base%':>9s} {'beats':>7s}")
    verdict = {}
    for es in names:
        old = np.array([raw[s][k_old][es] for s in seeds])
        new = np.array([raw[s][k_new][es] for s in seeds])
        base = np.array([raw[s]["_baseline"][es] for s in seeds if es in raw[s]["_baseline"]])

        d_rel = np.median((new - old) / old * 100)          # Q1, paired
        n_better = int((new < old).sum())

        if len(base) == len(new):
            b_rel = np.median((new - base) / base * 100)     # Q2, paired
            n_beat = int((new < base).sum())
        else:                                                # unpaired fallback
            b_rel, n_beat = np.median(new) / np.median(base) * 100 - 100, -1

        verdict[es] = {"d_rel": float(d_rel), "n_better": n_better,
                       "b_rel": float(b_rel), "n_beat": n_beat,
                       "final": float(np.median(new)), "base": float(np.median(base))}
        print(f"{es:10s} {np.median(old):9.5f} {np.median(new):9.5f} {d_rel:+8.2f} "
              f"{n_better:3d}/{len(seeds):<3d} | {np.median(base):9.5f} {b_rel:+9.2f} "
              f"{n_beat:3d}/{len(seeds):<3d}{_sign_note(n_beat, len(seeds))}")
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir", default="data/SI_results/convergence_preflight/shards")
    ap.add_argument("--k-old", type=int, default=1000)
    ap.add_argument("--k-new", type=int, default=2000)
    args = ap.parse_args()

    pooled = load_shards(Path(args.shard_dir))
    all_v = {}
    for fam in sorted(pooled):
        all_v[fam] = report(fam, pooled[fam], args.k_old, args.k_new)

    print("\n" + "=" * 78)
    print(f"SUMMARY - does the extension to {args.k_new} hold up out of sample?")
    print("=" * 78)
    print(f"{'family':10s} {'held-out sets improved':>24s} {'held-out sets beating base':>28s}")
    for fam, v in all_v.items():
        held = [es for es in v if es != TARGET_SET]
        imp = sum(v[es]["d_rel"] < 0 for es in held)
        beat = sum(v[es]["b_rel"] < 0 for es in held)
        print(f"{fam:10s} {imp:>10d}/{len(held):<13d} {beat:>14d}/{len(held):<13d}")
    print(f"\n('{TARGET_SET}' is the optimization target, excluded from the held-out counts;")
    print(" it is shown per-family above for reference.)")


if __name__ == "__main__":
    main()
