#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage C audit: re-rank the Stage 0b hyperparameter candidates out of sample.

The problem being audited
-------------------------
0b_hyperparameter_retune_{multi,agent}.py scored each candidate with

    score = mean(errors[-20:]),   errors[k] = NRMSE[k] + w * penalty(U[k-1])

Two faults follow from that one line:

  1. The penalty sits INSIDE the score, so every nonzero smoothness weight adds
     to the number being minimized and w=0 wins mechanically. We have already
     worked around this by choosing w by hand, out of sample (Stage D).
  2. The score is computed IN SAMPLE, on the same group being optimized. It
     never touches a held-out scenario. This fault was never addressed, and it
     applies to every OTHER hyperparameter the search chose - step_size,
     momentum, nesterov, lr_inner, wd_inner, batch_size.

A rule that scores training error rewards configurations that descend fastest on
the training group. Those are not necessarily the configurations that generalize.
The very large selected step sizes are what such a rule would be expected to pick.

What this script does NOT re-litigate
-------------------------------------
The smoothness weight. Every candidate here is run at the weight the author
chose in Stage D (Sulfur 0.02, multi 0.1, normalized form), so the only thing
varying between candidates is the rest of the configuration. Running each
candidate at its own sampled w would confound the two questions.

The question this answers
-------------------------
Under a corrected rule - true NRMSE, on held-out scenarios, at the production
2000 updates - is the deployed configuration clearly the wrong choice? A null
result (deployed winner within the seed noise of the best) is a real and
reportable outcome; the aim is not to find a new optimum.

Usage:
    python scripts/6h_hp_audit.py --mode run-one --family multi --cand 34 --seed 0
    python scripts/6h_hp_audit.py --mode candidates --family multi
    python scripts/6h_hp_audit.py --mode collect
"""
import argparse
import glob
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_sweep", str(PROJECT_ROOT / "scripts" / "6h_penalty_sweep.py"))
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

OUT_DIR = Path("data/SI_results/hp_audit")
CKPT_ROOT = Path("checkpoints/hp_audit")
N_CANDIDATES = 16          # top stable candidates by the existing (flawed) score
NUM_UPDATES = 2000
GROUP = "tier1"

# The weight the author selected in Stage D, held fixed across every candidate.
CHOSEN_W = {"Sulfur": 0.02, "multi": 0.1}
# Which config_idx is currently deployed, for the report. multi's
# best_config_unified.json records cand_idx=2, which indexes the TOP-5 list, not
# the original config_idx - the third entry there is config 34. Sulfur's
# cand_idx=0 is the first entry, config 53.
DEPLOYED = {"Sulfur": 53, "multi": 34}
HP_DIR = {"Sulfur": "data/SI_results/hp_retune/Sulfur",
          "multi": "data/SI_results/hp_retune/multi"}


def _norm_batch(b):
    """Decode the sweep's batch_size label.

    0b_hyperparameter_retune_multi.py writes it through _bs_label, which turns
    None into the string "full"; its own reader inverts that as
    `None if row["batch_size"] == "full" else int(...)` (line ~314). Mirrored
    here exactly rather than reinvented, so a "full" candidate is run
    full-batch and not silently coerced.
    """
    if b in (None, "full", "None", ""):
        return None
    return int(b)


def load_candidates(family: str) -> dict:
    """{config_idx: {config, mean_score, stable, any_nonzero_w}} for complete runs."""
    rows = defaultdict(dict)
    for p in glob.glob(f"{HP_DIR[family]}/cheap_unified_cfg*_result.json"):
        d = json.load(open(p))
        rows[int(d["config_idx"])][d["group"]] = d
    groups = sorted({g for v in rows.values() for g in v})
    out = {}
    for idx, byg in rows.items():
        if set(byg) != set(groups) or not all(v.get("stable") for v in byg.values()):
            continue
        any_g = next(iter(byg.values()))
        out[idx] = {
            "config": {
                "step_size": any_g["step_size"], "momentum": any_g["momentum"],
                "nesterov": any_g["nesterov"], "K_inner": any_g["K_inner"],
                "lr_inner": any_g["lr_inner"], "wd_inner": any_g["wd_inner"],
                "batch_size": _norm_batch(any_g.get("batch_size")),
            },
            "mean_score": float(np.mean([byg[g]["score"] for g in groups])),
            "sampled_w": float(any_g.get("smoothness_weight", 0.0)),
        }
    return out


def shortlist(family: str, n: int = N_CANDIDATES) -> list:
    """Top n by the existing score, with the deployed winner always included."""
    cands = load_candidates(family)
    order = sorted(cands, key=lambda i: cands[i]["mean_score"])[:n]
    if DEPLOYED[family] not in order:
        order.append(DEPLOYED[family])
    return order


def run_one(family: str, cand: int, seed: int):
    import jax
    import utils_inverse

    cands = load_candidates(family)
    if cand not in cands:
        raise SystemExit(f"candidate {cand} is not a complete+stable {family} candidate")
    hp = cands[cand]["config"]
    cfg = sweep.FAMILIES[family]
    baseline = json.load(open(cfg["baseline_cfg"]))["config"]

    setup = utils_inverse.run_inverse_experiment_setup(
        cfg["agents"], cfg["active"], mode="FaIR",
        CS3=True, DAMIP=False, GeoMIP=False, idx_demo=None, seed=seed,
        baseline_K=baseline["K"], baseline_lr=baseline["lr"],
        baseline_weight_decay=baseline["weight_decay"],
    )
    groups = utils_inverse.build_group_emis_dicts(setup["emis_dict_train_JAX"],
                                                  setup["eval_sets"])
    ck_dir = CKPT_ROOT / family / f"cand{cand}"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ck_dir / f"inverse_constant_{GROUP}_{family}_cand{cand}_seed{seed}.pkl"

    utils_inverse.optimize_emissions_inverse(
        groups[GROUP], setup["params0"], num_updates=NUM_UPDATES,
        step_size=hp["step_size"], momentum=hp["momentum"], nesterov=hp["nesterov"],
        K_inner=hp["K_inner"], lr_inner=hp["lr_inner"], wd_inner=hp["wd_inner"],
        agents=tuple(utils_inverse.AGENTS_DEFAULT), active_agents=cfg["active"],
        init_cond=cfg["init_cond"], T=cfg["T"], filter_hist=cfg["filter_hist"],
        mode="FaIR", checkpoint_path=str(ckpt), checkpoint_every=200,
        resume_if_exists=False, preds_every=200, batch_size=hp["batch_size"],
        key=jax.random.PRNGKey(seed),
        smoothness_weight=CHOSEN_W[family], penalty_form="normalized",
    )

    with open(ckpt, "rb") as f:
        raw = pickle.load(f)
    U_final = raw["U_traj"][-1]
    extra = {} if cfg["eval_agents"] is None else {"agents": cfg["eval_agents"]}
    oos = utils_inverse.evaluate_optimal_emulator(
        training_paths=[str(ckpt)], train_scenarios=["final"],
        eval_sets=setup["eval_sets"], params0=setup["params0"],
        active_agents=cfg["active"], inactive_mode="zeros",
        historical_name="historical", key=jax.random.PRNGKey(seed),
        K=hp["K_inner"], lr=hp["lr_inner"], weight_decay=hp["wd_inner"],
        mode="FaIR", batch_size=hp["batch_size"], **extra,
    )
    rec = {
        "family": family, "cand": cand, "seed": seed,
        "deployed": cand == DEPLOYED[family],
        "old_score": cands[cand]["mean_score"], "sampled_w": cands[cand]["sampled_w"],
        "nrmse_in_sample": float(utils_inverse.recover_nrmse_trajectory(raw)[-1]),
        "oos": {es: float(oos["final"][es]["mean"]) for es in oos["final"]},
        "baseline": {es: float(setup["baseline_results"][es]["mean"])
                     for es in setup["baseline_results"]},
        "R": {a: sweep.roughness(U_final[a]) for a in cfg["active"]},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{family}_cand{cand}_seed{seed}.json"
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"wrote {out}")


def collect():
    recs = [json.load(open(p)) for p in sorted(OUT_DIR.glob("*_cand*_seed*.json"))]
    if not recs:
        raise FileNotFoundError(f"no audit results in {OUT_DIR}")
    rng = np.random.default_rng(0)
    for family in sorted({r["family"] for r in recs}):
        fam = [r for r in recs if r["family"] == family]
        eval_names = sorted(fam[0]["oos"])
        by = defaultdict(list)
        for r in fam:
            by[r["cand"]].append(r)
        print(f"\n=== {family}: candidates re-ranked OUT OF SAMPLE "
              f"(group={GROUP}, {NUM_UPDATES} iters, w={CHOSEN_W[family]}) ===")
        print(f"{'cand':>6s} {'old rank':>9s} {'old score':>10s} {'n':>3s} {'R':>7s} "
              + " ".join(f"{e:>9s}" for e in eval_names) + "   note")
        old_order = sorted(by, key=lambda c: by[c][0]["old_score"])
        rank = {c: i + 1 for i, c in enumerate(old_order)}
        # rank candidates by held-out mean over all eval sets
        def key(c):
            return np.median([np.mean(list(r["oos"].values())) for r in by[c]])
        for c in sorted(by, key=key):
            sub = by[c]
            note = "<= DEPLOYED" if sub[0]["deployed"] else ""
            if sub[0]["sampled_w"]:
                note += "  (search sampled w>0)"
            print(f"{c:6d} {rank[c]:9d} {sub[0]['old_score']:10.4f} {len(sub):3d} "
                  f"{np.median([np.median(list(r['R'].values())) for r in sub]):7.4f} "
                  + " ".join(f"{np.median([r['oos'][e] for r in sub]):9.5f}"
                             for e in eval_names) + f"   {note}")
        base = {e: np.median([r["baseline"][e] for r in fam]) for e in eval_names}
        print(f"{'basel.':>6s} {'':9s} {'':10s} {'':3s} {'':7s} "
              + " ".join(f"{base[e]:9.5f}" for e in eval_names))

        # Is the deployed candidate distinguishable from the best?
        dep = DEPLOYED[family]
        best = min(by, key=key)
        if dep in by and best != dep:
            seeds = sorted(set(r["seed"] for r in by[dep]) & set(r["seed"] for r in by[best]))
            d = np.array([np.mean(list(next(r for r in by[dep] if r["seed"] == s)["oos"].values()))
                          - np.mean(list(next(r for r in by[best] if r["seed"] == s)["oos"].values()))
                          for s in seeds])
            bs = np.percentile(np.median(d[rng.integers(0, len(d), (20000, len(d)))], axis=1),
                               [2.5, 97.5])
            print(f"\n  deployed (cand {dep}) minus best (cand {best}), paired over "
                  f"{len(seeds)} seeds: median {np.median(d):+.5f}, 95% CI [{bs[0]:+.5f}, {bs[1]:+.5f}]")
            print("  " + ("DEPLOYED IS NOT DISTINGUISHABLE FROM THE BEST - audit passes."
                          if bs[0] <= 0 <= bs[1]
                          else "DEPLOYED IS SIGNIFICANTLY WORSE - review the choice."))
        elif best == dep:
            print(f"\n  deployed (cand {dep}) also ranks best out of sample - audit passes.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["run-one", "candidates", "collect"], default="collect")
    ap.add_argument("--family", choices=list(CHOSEN_W))
    ap.add_argument("--cand", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    if args.mode == "candidates":
        for c in shortlist(args.family):
            print(c)
    elif args.mode == "run-one":
        if args.family is None or args.cand is None or args.seed is None:
            ap.error("--mode run-one needs --family, --cand and --seed")
        run_one(args.family, args.cand, args.seed)
    else:
        collect()


if __name__ == "__main__":
    main()
