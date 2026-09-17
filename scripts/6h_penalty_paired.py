"""Paired per-seed comparison of each nonzero w against the w=0 control.

Differencing medians of separate curves is invalid when each seed carries its
own init and its own control, so every statistic here is a within-seed
difference first, summarized second.
"""
import json, glob, re
from collections import defaultdict
import numpy as np

recs = defaultdict(dict)  # (family, w) -> seed -> record
for p in glob.glob('data/SI_results/penalty_sweep/*.json'):
    d = json.load(open(p))
    recs[(d['family'], float(d['w']))][int(d['seed'])] = d

fams = sorted({f for f, _ in recs})
SETS = ['Tier 1', 'Tier 2', 'DECK', 'CS3', 'All']

for fam in fams:
    ws = sorted({w for f, w in recs if f == fam})
    ctrl = recs[(fam, 0.0)]
    seeds = sorted(ctrl)
    print(f"\n=== {fam}: paired change vs w=0 control (n={len(seeds)} seeds) ===")
    print(f"{'w':>9s} {'dR%':>8s} {'dP%':>8s} " +
          " ".join(f"{s:>16s}" for s in SETS))
    print(f"{'':9s} {'':8s} {'':8s} " +
          " ".join(f"{'med% (n worse)':>16s}" for s in SETS))
    for w in ws:
        if w == 0.0:
            continue
        cur = recs[(fam, w)]
        common = [s for s in seeds if s in cur]
        def rel(get):
            a = np.array([get(cur[s]) for s in common])
            b = np.array([get(ctrl[s]) for s in common])
            return (a - b) / b * 100.0
        Rk = list(ctrl[common[0]]['R'])
        dR = rel(lambda d: float(np.mean([d['R'][k] for k in Rk])))
        dP = rel(lambda d: d['P'])
        cells = []
        for s in SETS:
            r = rel(lambda d, s=s: d['oos'][s])
            cells.append(f"{np.median(r):+7.2f} ({int((r > 0).sum()):2d}/{len(r)})")
        print(f"{w:9.2e} {np.median(dR):+7.2f} {np.median(dP):+7.2f} " +
              " ".join(f"{c:>16s}" for c in cells))

    # Does the control beat baseline, and does the best-smoothing arm still?
    print(f"  -- beats baseline (per-seed, n worse than baseline) --")
    for w in [0.0, ws[-1]]:
        cur = recs[(fam, w)]
        common = [s for s in seeds if s in cur]
        out = []
        for s in SETS:
            a = np.array([cur[k]['oos'][s] for k in common])
            b = np.array([cur[k]['baseline'][s] for k in common])
            r = (a - b) / b * 100.0
            out.append(f"{s}={np.median(r):+6.1f}% ({int((r < 0).sum())}/{len(r)} win)")
        print(f"    w={w:.0e}: " + "  ".join(out))
