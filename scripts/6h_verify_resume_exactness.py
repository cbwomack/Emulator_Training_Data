#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage C verification: prove that resuming 1000 -> 2000 did not perturb history.

`resume_if_exists=True` reopens a finished checkpoint and keeps optimizing from
its stored optimizer state, writing back to the SAME path. That preserves ~29 GB
of compute instead of rerunning from scratch, but it also means a subtly wrong
resume would silently rewrite the first half of every trajectory rather than
failing. tests/test_pipeline_functions.py proves bit-exactness on a synthetic
fixture; this proves it on the real artifacts.

Compares a resumed checkpoint against a pre-resume copy of itself:
  - errors[:1001]  must be bit-identical (not allclose)
  - U_traj[k]      must be bit-identical for every k <= 1000
  - the resumed file must actually be longer (2001 entries)

Deliberately uses array_equal rather than allclose: float32 SGD is
deterministic here, so "close" would hide exactly the drift we are testing for.

Usage:
    python scripts/6h_verify_resume_exactness.py                 # all pre-images
    python scripts/6h_verify_resume_exactness.py --preimage-dir X --expect 2001
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

# Where each pre-image came from, so the resumed counterpart can be located by
# basename alone (the pre-image dir is flat).
SEARCH_DIRS = [
    "checkpoints/co2_retuned/seed_sweep",
    "checkpoints/multi_retuned/seed_sweep",
    "checkpoints/multi_fig4/seed_sweep",
    "checkpoints/BC_retuned/seed_sweep",
    "checkpoints/CH4_retuned/seed_sweep",
    "checkpoints/N2O_retuned/seed_sweep",
    "checkpoints/Sulfur_retuned/seed_sweep",
]


def _locate(basename: str) -> Path | None:
    for d in SEARCH_DIRS:
        p = Path(d) / basename
        if p.exists():
            return p
    return None


def _tree_equal(a, b) -> bool:
    """Bit-exact comparison of two U_traj entries ({agent: array} dicts)."""
    if set(a) != set(b):
        return False
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)


def compare(pre_path: Path, new_path: Path, expect: int) -> dict:
    with open(pre_path, "rb") as f:
        old = pickle.load(f)
    with open(new_path, "rb") as f:
        new = pickle.load(f)

    n_old = len(old["errors"])
    res = {"file": new_path.name, "n_old": n_old, "n_new": len(new["errors"]),
           "problems": []}

    if len(new["errors"]) != expect:
        res["problems"].append(
            f"expected {expect} errors after resume, found {len(new['errors'])}"
            + ("  (did the job actually run?)" if len(new["errors"]) == n_old else ""))

    e_old = np.asarray(old["errors"])
    e_new = np.asarray(new["errors"])[:n_old]
    if not np.array_equal(e_old, e_new):
        n_diff = int((e_old != e_new).sum())
        first = int(np.argmax(e_old != e_new))
        res["problems"].append(
            f"errors[:{n_old}] differ in {n_diff} places, first at index {first} "
            f"({e_old[first]!r} -> {e_new[first]!r})")

    n_u = min(len(old["U_traj"]), len(new["U_traj"]))
    bad = [k for k in range(n_u) if not _tree_equal(old["U_traj"][k], new["U_traj"][k])]
    if bad:
        res["problems"].append(
            f"U_traj differs at {len(bad)} of the first {n_u} iterates "
            f"(first: {bad[0]})")

    res["ok"] = not res["problems"]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preimage-dir", default="_stageC_preimage")
    ap.add_argument("--expect", type=int, default=2001,
                    help="expected len(errors) after the resume (num_updates+1)")
    args = ap.parse_args()

    pre_dir = Path(args.preimage_dir)
    pres = sorted(pre_dir.glob("*.pkl"))
    if not pres:
        raise FileNotFoundError(f"no pre-image checkpoints in {pre_dir}")

    results, missing = [], []
    for pre in pres:
        new = _locate(pre.name)
        if new is None:
            missing.append(pre.name)
            continue
        results.append(compare(pre, new, args.expect))

    print(f"{'checkpoint':52s} {'old':>6s} {'new':>6s}  verdict")
    for r in results:
        print(f"{r['file']:52s} {r['n_old']:6d} {r['n_new']:6d}  "
              + ("OK - history bit-identical" if r["ok"] else "MISMATCH"))
        for p in r["problems"]:
            print(f"    !! {p}")
    for m in missing:
        print(f"{m:52s} {'?':>6s} {'?':>6s}  NOT FOUND in any search dir")

    n_ok = sum(r["ok"] for r in results)
    total = len(results) + len(missing)
    print(f"\n{n_ok}/{total} verified bit-identical")
    if n_ok != total:
        print("RESUME IS NOT SAFE - do not release the full arrays.")
        sys.exit(1)
    print("Resume preserved all pre-resume history exactly.")


if __name__ == "__main__":
    main()
