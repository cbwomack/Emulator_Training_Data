"""Paired per-seed reading of the extended Sulfur sweep and the multi 2x2.

Medians of separate curves mislead when each seed carries its own init and its
own baseline, so every number here is a within-seed difference first.
"""
import json, glob
from collections import defaultdict
import numpy as np

recs = defaultdict(dict)
for p in glob.glob('data/SI_results/penalty_sweep/*.json'):
    d = json.load(open(p)); recs[(d['family'], float(d['w']))][int(d['seed'])] = d

SETS = ['Tier 1', 'Tier 2', 'DECK', 'CS3', 'All']
REAL_BAND = 0.0959


def vs_baseline(cur, seeds):
    out = []
    for s in SETS:
        a = np.array([cur[k]['oos'][s] for k in seeds])
        b = np.array([cur[k]['baseline'][s] for k in seeds])
        r = (a - b) / b * 100
        out.append(f"{np.median(r):+6.1f}%({int((r < 0).sum())}/{len(r)})")
    return out


def table(fam, ctrl_key, title):
    ws = sorted({w for f, w in recs if f == fam})
    ctrl = recs[ctrl_key]
    cseeds = sorted(ctrl)
    print(f"\n{'='*118}\n{title}\n{'='*118}")
    print(f"{'w':>8s} {'R_med':>7s} {'in band':>8s} | {'paired vs control: median % (n worse/10)':^52s} "
          f"| {'vs baseline: median % (n win/10)':^46s}")
    print(f"{'':8s} {'':7s} {'':8s} | " + " ".join(f"{s:>10s}" for s in SETS)
          + " | " + " ".join(f"{s:>9s}" for s in SETS))
    for w in ws:
        cur = recs[(fam, w)]
        seeds = sorted(set(cur) & set(cseeds))
        Rm = np.median([np.median(list(cur[s]['R'].values())) for s in sorted(cur)])
        cells = []
        for s in SETS:
            a = np.array([cur[k]['oos'][s] for k in seeds])
            b = np.array([ctrl[k]['oos'][s] for k in seeds])
            r = (a - b) / b * 100
            cells.append(f"{np.median(r):+6.2f}({int((r > 0).sum())})")
        band = "YES" if Rm <= REAL_BAND else ""
        print(f"{w:8.3g} {Rm:7.4f} {band:>8s} | " + " ".join(f"{c:>10s}" for c in cells)
              + " | " + " ".join(f"{c:>9s}" for c in vs_baseline(cur, sorted(cur))))


table('Sulfur', ('Sulfur', 0.0),
      "SULFUR: full sweep, paired against its own w=0 control")

table('multi_retunedsteps', ('multi', 0.0),
      "MULTI arm A: retuned step sizes + penalty ladder, paired against arm D (retuned, w=0)")

table('multi_origsteps', ('multi', 0.0),
      "MULTI arms B/C: ORIGINAL step sizes, paired against arm D (retuned, w=0)")

print(f"\n(real ScenarioMIP band: R <= {REAL_BAND}; white noise R = 1.414)")
