#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage D assessment: how rough are the optimized emissions, and what would a
normalized smoothness penalty have to weigh to matter?

Two diagnostics, both dimensionless so agents whose native units span 100x can
be compared at all:

  R = RMS(dU) / SD(U)      roughness. R->0 smooth; R=sqrt(2)~1.414 is white
                           noise (successive years independent). Real
                           ScenarioMIP profiles sit far below 0.1.

  P = mean_t[(dU)^2] / sigma_ref^2     the proposed penalty, per agent.
                           sigma_ref is the SD of that agent pooled over the
                           real ScenarioMIP Tier-1 ensemble - a fixed constant,
                           NOT a function of U, so the optimizer cannot inflate
                           the denominator instead of smoothing the numerator.

The existing penalty is an unnormalized sum in native units. That is why the
tuned weight is 0 for the multi-agent case: sum_t (dU)^2 differs by ~5 orders of
magnitude between regimes, so one grid could not serve both and 0 was the only
survivable value. This quantifies the gap the normalized form has to close.

Reports, per family, the weight w that would make w*P equal to a target
fraction of the current NRMSE. That is the anchor for the tuning sweep - it
replaces guessing a grid.

Usage:
    python scripts/6h_penalty_assessment.py --seeds 5
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

# SD of each agent pooled over the real ScenarioMIP Tier-1 ensemble.
# Reproducible from data/FaIR_IO/emissions/ScenarioMIP_tier1_CO2_CH4_N2O_Sulfur_BC.pkl;
# verified to 4 s.f. against the values recorded in the plan.
SIGMA_REF = {"CO2": 25.0477, "CH4": 185.0055, "N2O": 5.2361,
             "Sulfur": 29.9742, "BC": 1.8038}

REAL_EMIS = "data/FaIR_IO/emissions/ScenarioMIP_tier1_CO2_CH4_N2O_Sulfur_BC.pkl"

FAMILIES = {
    "co2":        ("checkpoints/co2_retuned/seed_sweep",    "co2_only",    ("CO2",)),
    "CH4":        ("checkpoints/ch4_retuned/seed_sweep",    "ch4_only",    ("CH4",)),
    "N2O":        ("checkpoints/n2o_retuned/seed_sweep",    "n2o_only",    ("N2O",)),
    "Sulfur":     ("checkpoints/Sulfur_retuned/seed_sweep", "Sulfur_only", ("Sulfur",)),
    "BC":         ("checkpoints/BC_retuned/seed_sweep",     "BC_only",     ("BC",)),
    "multi_fig4": ("checkpoints/multi_fig4/seed_sweep",     "multi_fig4",
                   ("CO2", "CH4", "N2O", "Sulfur", "BC")),
}


def roughness(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return float(np.sqrt(np.mean(np.diff(x) ** 2)) / s) if s > 0 else 0.0


def penalty_P(x, agent) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(np.diff(x) ** 2) / SIGMA_REF[agent] ** 2)


def real_scenario_band():
    with open(REAL_EMIS, "rb") as f:
        d = pickle.load(f)
    out = {}
    for a in SIGMA_REF:
        Rs = [roughness(d[s][a]) for s in d]
        Ps = [penalty_P(d[s][a], a) for s in d]
        out[a] = {"R": (min(Rs), max(Rs)), "P": (min(Ps), max(Ps))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--group", default="tier1")
    ap.add_argument("--target-frac", type=float, default=0.10,
                    help="w is reported such that w*P equals this fraction of NRMSE")
    args = ap.parse_args()

    band = real_scenario_band()
    print("=== real ScenarioMIP Tier-1 profiles (the target band) ===")
    print(f"{'agent':8s} {'R min':>8s} {'R max':>8s} {'P min':>11s} {'P max':>11s}")
    for a, v in band.items():
        print(f"{a:8s} {v['R'][0]:8.4f} {v['R'][1]:8.4f} {v['P'][0]:11.3e} {v['P'][1]:11.3e}")

    print(f"\n=== optimized profiles at iteration 2000, group={args.group}, "
          f"n={args.seeds} seeds (median) ===")
    print(f"{'family':11s} {'agent':8s} {'R':>8s} {'R/real_max':>11s} "
          f"{'P':>11s} {'P/real_max':>11s} {'NRMSE':>9s} {'w for '+str(int(args.target_frac*100))+'%':>12s}")
    for fam, (cdir, tag, agents) in FAMILIES.items():
        Rs = {a: [] for a in agents}
        Ps = {a: [] for a in agents}
        Pfam, nrmses = [], []
        for seed in range(args.seeds):
            p = Path(cdir) / f"inverse_constant_{args.group}_{tag}_seed{seed}.pkl"
            if not p.exists():
                continue
            with open(p, "rb") as f:
                raw = pickle.load(f)
            U = raw["U_traj"][-1]
            nrmses.append(float(utils_inverse.recover_nrmse_trajectory(raw)[-1]))
            per_agent = []
            for a in agents:
                Rs[a].append(roughness(U[a]))
                pa = penalty_P(U[a], a)
                Ps[a].append(pa)
                per_agent.append(pa)
            Pfam.append(np.mean(per_agent))     # P is the mean over active agents
        if not nrmses:
            print(f"{fam:11s} (no checkpoints)")
            continue
        med_nrmse = float(np.median(nrmses))
        med_Pfam = float(np.median(Pfam))
        w = args.target_frac * med_nrmse / med_Pfam if med_Pfam > 0 else float("nan")
        for j, a in enumerate(agents):
            mR, mP = np.median(Rs[a]), np.median(Ps[a])
            head = fam if j == 0 else ""
            tail = (f"{med_nrmse:9.5f} {w:12.3e}" if j == 0
                    else f"{'':9s} {'':12s}")
            print(f"{head:11s} {a:8s} {mR:8.4f} {mR/band[a]['R'][1]:11.1f} "
                  f"{mP:11.3e} {mP/band[a]['P'][1]:11.1f} {tail}")
    print("\nR/real_max and P/real_max are multiples of the roughest REAL scenario.")
    print("white noise is R = sqrt(2) ~ 1.414.")


if __name__ == "__main__":
    main()
