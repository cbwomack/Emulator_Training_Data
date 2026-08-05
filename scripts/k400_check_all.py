#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
One-off diagnostic (not part of the reusable 0b pipeline): the manuscript's
methodology fixes K_inner=400 for every experiment, but Stage 0b's random
search for the CO2 'all' group happened to never sample K_inner=400 among
its 25 configs, so its winner (K_inner=600, full-batch) isn't directly
comparable to what the paper's fixed K_inner=400 would give. This checks
K_inner=400 x batch_size in {full,16,32,64}, holding every other
hyperparameter at the 'all' group's actual winning values, 3 seeds each,
full num_updates=1000 - same rigor as Stage 0b's validate phase.

Usage:
    python k400_check_all.py --batch-size full --seed 0
    python k400_check_all.py --report   # after all combos are done
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import jax

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import utils_inverse

OUT_DIR = Path("data/SI_results/hp_retune")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGENTS = ["CO2"]
ACTIVE_AGENTS = ("CO2",)
MODE = "FaIR"
GROUP = "all"

# 'all' group's actual Stage 0b winner, minus K_inner/batch_size (the two knobs under test here)
BASE_CFG = {
    "step_size": 848.0664863850201,
    "momentum": 0.85,
    "nesterov": True,
    "lr_inner": 0.020803099890523524,
    "wd_inner": 0.0,
    "smoothness_weight": 0.0,
}
K_INNER = 400
BATCH_SIZES = [None, 16, 32, 64]
SEEDS = [0, 1, 2]


def _bs_label(bs):
    return "full" if bs is None else str(bs)


def result_path(bs, seed):
    return OUT_DIR / f"k400_all_bs{_bs_label(bs)}_seed{seed}.json"


def run_one(bs, seed):
    rp = result_path(bs, seed)
    if rp.is_file():
        print(f"[bs={_bs_label(bs)} seed={seed}] already done, skipping")
        return

    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("_exp_mod_3a", PROJECT_ROOT / "scripts" / "3a_inverse_CO2_only.py")
    exp_mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(exp_mod)
    base_exp = exp_mod.EXPERIMENTS[GROUP]

    setup = utils_inverse.run_inverse_experiment_setup(
        AGENTS, ACTIVE_AGENTS, mode=MODE, CS3=True, DAMIP=False, GeoMIP=False,
        seed=seed, idx_demo=None,
    )
    group_emis_dicts = utils_inverse.build_group_emis_dicts(
        setup["emis_dict_train_JAX"], setup["eval_sets"],
    )

    t0 = time.perf_counter()
    result = utils_inverse.optimize_emissions_inverse(
        emis_dict=group_emis_dicts[GROUP],
        params0=setup["params0"],
        num_updates=1000,
        K_inner=K_INNER,
        active_agents=ACTIVE_AGENTS,
        init_cond=base_exp["init_cond"], T=base_exp["T"], filter_hist=base_exp["filter_hist"],
        mode=MODE,
        checkpoint_path=None, checkpoint_every=1000, resume_if_exists=False, preds_every=1000,
        batch_size=bs, key=jax.random.PRNGKey(1000 + seed),
        **BASE_CFG,
    )
    dt = time.perf_counter() - t0
    errors = np.asarray(result["errors"])
    stable = bool(np.isfinite(errors).all())
    tail = errors[-min(20, len(errors)):]
    score = float(np.mean(tail)) if stable else float("inf")
    row = {"batch_size": _bs_label(bs), "seed": seed, "K_inner": K_INNER,
           "score": score, "lossF": float(errors[-1]), "stable": stable, "seconds": round(dt, 2)}
    print(f"[bs={_bs_label(bs)} seed={seed}] score={score:.4g} stable={stable} ({dt:.1f}s)", flush=True)

    tmp = rp.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, rp)


def report():
    rows = []
    for bs in BATCH_SIZES:
        for seed in SEEDS:
            rp = result_path(bs, seed)
            if rp.is_file():
                rows.append(json.load(open(rp)))
    if not rows:
        print("no results yet")
        return
    by_bs = {}
    for r in rows:
        by_bs.setdefault(r["batch_size"], []).append(r)
    print(f"{'batch_size':>10} {'n':>3} {'mean_score':>12} {'std':>10} {'all_stable':>10}")
    for bs_label in ["full", "16", "32", "64"]:
        rs = by_bs.get(bs_label, [])
        if not rs:
            continue
        scores = [r["score"] for r in rs]
        stable = all(r["stable"] for r in rs)
        print(f"{bs_label:>10} {len(rs):>3} {np.mean(scores):>12.4g} {np.std(scores):>10.4g} {str(stable):>10}")
    print("\nFor reference, Stage 0b's actual 'all' winner: K_inner=600, batch_size=full, mean_score=0.06594")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=str, default=None, help="'full', '16', '32', or '64'")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if args.report:
        report()
    else:
        bs = None if args.batch_size in (None, "full") else int(args.batch_size)
        run_one(bs, args.seed)
