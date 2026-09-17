#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Why were the ORIGINAL multi-agent profiles smooth, and can we get that back?

Background
----------
The Jan-2026 'first stab' runs in checkpoints/multi/ have R = 0.035-0.16, i.e.
at or below the roughest real ScenarioMIP profile (0.0959). The current retuned
seed sweeps have R ~ 1.20, essentially white noise (sqrt(2) = 1.414). Two things
changed at once between them, so they are confounded:

  1. step_size. Original multi tier1 used
       {CO2: 10, CH4: 500, N2O: 10, Sulfur: 10, BC: 5}
     The Stage 0b retune selected
       {CO2: 397, CH4: 13870, N2O: 22, Sulfur: 4490, BC: 91}
     - up to 449x larger (Sulfur).

  2. smoothness_weight. Original multi tier1 used w_legacy = 1e-5; the retune
     set it to 0. That was not a modelling result: the retune scored candidates
     on mean(errors[-20:]) where errors[k] = NRMSE[k] + w*penalty, so the
     penalty sat inside the objective being minimized and w=0 won mechanically.

Measured on the original profiles, w_legacy = 1e-5 corresponds to
w_normalized = 3.46 (and the single-agent Sulfur original, w_legacy = 5e-6,
to w_normalized = 3.37 analytically). The Stage D sweep covered [1e-5, 1e-3]
- roughly 3,400x below the historically working setting. Its "the penalty does
nothing for multi" result therefore says nothing about the region that matters.

Design
------
A 2x2 in {step_size} x {smoothness weight}, all at 2000 iterations so the
iteration count is held fixed and cannot confound:

    arm                     step_size      w_norm
    D  (already on disk)    retuned        0        <- Stage C control
    A                       retuned        ladder   <- does w alone fix it?
    B                       original       0        <- does step_size alone?
    C                       original       3.46     <- reproduce the original

Everything is scored OUT OF SAMPLE through the same evaluate_optimal_emulator
path as the Stage D sweep, so the numbers are directly comparable to it.

Usage:
    python scripts/6h_origin_config_probe.py --mode run-one --family multi_origsteps --w 0 --seed 0
    python scripts/6h_origin_config_probe.py --mode collect
"""
import argparse
import copy
import json
import os
import sys
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

# The original multi-agent tier1 step sizes, transcribed from the config cell of
# 3b_inverse_all_agents.ipynb (the cell the notebook's own prose points at when
# it says results "are sensitive to the step size taken for each forcing agent,
# along with the weight of the smoothness penalty").
ORIGINAL_STEP_SIZE = {"CO2": 10.0, "CH4": 500.0, "N2O": 10.0,
                      "Sulfur": 10.0, "BC": 5.0}
# ...and the inner-loop settings that went with them. The retune moved these too
# (momentum 0.9 -> 0.95, wd_inner 0.01 -> 0.03), so reproducing the original
# arm means reproducing all of it, not just step_size.
ORIGINAL_INNER = {"momentum": 0.9, "nesterov": True, "K_inner": 400,
                  "lr_inner": 0.05, "wd_inner": 0.01}

# w_norm equivalent of the original w_legacy = 1e-5, measured on the original
# profiles themselves (_conv.py): legacy/normalized = 3.4625e5.
W_ORIGINAL_EQUIV = 3.46

# Ladder for arm A: does the penalty alone rescue the retuned configuration?
# Spans the original-equivalent value and two decades below it, because Stage D
# already showed [1e-5, 1e-3] is inert.
W_LADDER = [1e-2, 2e-2, 5e-2, 1e-1, 1e0, 3.46, 1e1]
# 2e-2 and 5e-2 were added 2026-08-26 after the first pass: paired Tier-1
# confidence intervals at n=10 are 31-74 points wide and every one straddles
# zero, so the ladder cannot separate w=0.01 from w=0.1 on skill. It CAN resolve
# roughness (R has little seed spread), and these two points sit where a single
# shared w across Sulfur and multi would land - Sulfur enters the real
# ScenarioMIP band at w=0.05. Filling them keeps a uniform-w choice available
# without committing to multi's noisy skill optimum.

VARIANTS = {
    # arm B and C: original step sizes and inner-loop settings
    "multi_origsteps": {"step_size": ORIGINAL_STEP_SIZE, "inner": ORIGINAL_INNER},
    # arm A: retuned everything, only w varies (step_size=None keeps the tuned dict)
    "multi_retunedsteps": {"step_size": None, "inner": None},
}


def _install(variant: str):
    """Register a variant as a pseudo-family so the sweep's own run/eval path
    can be reused verbatim - same setup, same evaluation, same file layout."""
    cfg = copy.deepcopy(sweep.FAMILIES["multi"])
    cfg["tag"] = variant
    sweep.FAMILIES[variant] = cfg
    return cfg


def run_one(variant: str, w: float, seed: int):
    import jax
    import utils_inverse

    cfg = _install(variant)
    over = VARIANTS[variant]
    unified = json.load(open(cfg["unified_cfg"]))["config"]
    if over["inner"]:
        unified = {**unified, **over["inner"]}
    step_size = over["step_size"] or unified["step_size"]

    baseline = json.load(open(cfg["baseline_cfg"]))["config"]
    setup = utils_inverse.run_inverse_experiment_setup(
        cfg["agents"], cfg["active"], mode="FaIR",
        CS3=True, DAMIP=False, GeoMIP=False, idx_demo=None, seed=seed,
        baseline_K=baseline["K"], baseline_lr=baseline["lr"],
        baseline_weight_decay=baseline["weight_decay"],
    )
    groups = utils_inverse.build_group_emis_dicts(setup["emis_dict_train_JAX"],
                                                  setup["eval_sets"])

    ck_dir = sweep.CKPT_ROOT / variant / f"w{sweep._wtag(w)}"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ck_dir / f"inverse_constant_{sweep.GROUP}_{variant}_seed{seed}.pkl"

    utils_inverse.optimize_emissions_inverse(
        groups[sweep.GROUP], setup["params0"],
        num_updates=sweep.NUM_UPDATES,
        step_size=step_size, momentum=unified["momentum"],
        nesterov=unified["nesterov"], K_inner=unified["K_inner"],
        lr_inner=unified["lr_inner"], wd_inner=unified["wd_inner"],
        agents=tuple(utils_inverse.AGENTS_DEFAULT), active_agents=cfg["active"],
        init_cond=cfg["init_cond"], T=cfg["T"], filter_hist=cfg["filter_hist"],
        mode="FaIR", checkpoint_path=str(ckpt), checkpoint_every=100,
        resume_if_exists=False, preds_every=100, batch_size=unified["batch_size"],
        key=jax.random.PRNGKey(seed),
        smoothness_weight=w, penalty_form="normalized",
    )
    sweep._evaluate_and_write(variant, w, seed, cfg, setup, unified, ckpt)


def collect():
    recs = []
    for f in sorted(sweep.OUT_DIR.glob("multi_*steps_w*_seed*.json")):
        with open(f) as fh:
            recs.append(json.load(fh))
    # Stage C control (arm D) for reference, scored through the identical path.
    ctrl = [json.load(open(f)) for f in sorted(sweep.OUT_DIR.glob("multi_w0_seed*.json"))]
    if not recs:
        raise FileNotFoundError(f"no probe results in {sweep.OUT_DIR}")

    eval_names = sorted(recs[0]["oos"])
    print(f"\n=== multi tier1, 2000 iters: step_size x smoothness (out of sample) ===")
    print(f"{'arm':22s} {'w_norm':>8s} {'n':>3s} {'R_med':>7s} {'P_med':>10s} "
          f"{'in-samp':>9s} " + " ".join(f"{e:>9s}" for e in eval_names))

    def row(label, sub):
        if not sub:
            return
        Rm = np.median([np.median(list(r["R"].values())) for r in sub])
        print(f"{label:22s} {sub[0]['w']:8.3g} {len(sub):3d} {Rm:7.4f} "
              f"{np.median([r['P'] for r in sub]):10.3e} "
              f"{np.median([r['nrmse_in_sample'] for r in sub]):9.5f} "
              + " ".join(f"{np.median([r['oos'][e] for r in sub]):9.5f}"
                         for e in eval_names))

    row("D retuned/w=0", ctrl)
    for variant in ["multi_retunedsteps", "multi_origsteps"]:
        sub_all = [r for r in recs if r["family"] == variant]
        for w in sorted({r["w"] for r in sub_all}):
            if variant == "multi_retunedsteps":
                tag = "A retuned steps"
            else:
                tag = "B original steps" if w == 0 else "C original (paper)"
            row(tag, [r for r in sub_all if r["w"] == w])
    if ctrl:
        base = {e: np.median([r["baseline"][e] for r in ctrl]) for e in eval_names}
        print(f"{'baseline':22s} {'':8s} {'':3s} {'':7s} {'':10s} {'':9s} "
              + " ".join(f"{base[e]:9.5f}" for e in eval_names))
    print(f"  (roughest real ScenarioMIP profile: R = 0.0959;  white noise R = 1.414)")
    print(f"  (original paper multi tier1: R = 0.1124, w_legacy=1e-5 == w_norm={W_ORIGINAL_EQUIV})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["run-one", "collect"], default="collect")
    ap.add_argument("--family", choices=list(VARIANTS))
    ap.add_argument("--w", type=float)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    if args.mode == "run-one":
        if args.family is None or args.w is None or args.seed is None:
            ap.error("--mode run-one needs --family, --w and --seed")
        run_one(args.family, args.w, args.seed)
    else:
        collect()


if __name__ == "__main__":
    main()
