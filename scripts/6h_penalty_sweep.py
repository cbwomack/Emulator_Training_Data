#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage D: sweep the normalized smoothness weight w, scored OUT OF SAMPLE.

Why this does not reuse scripts/0b_hyperparameter_retune_*.py
-------------------------------------------------------------
Those scripts rank candidates on `mean(errors[-20:])`, where

    errors[k] = NRMSE[k] + w * penalty(U[k-1])

The penalty is INSIDE the score. Every w > 0 therefore adds a strictly positive
quantity to the very number being minimized, so w = 0 wins almost mechanically.
That, not the grid spacing, is the deeper reason the historical search returned
smoothness_weight = 0 everywhere: the grid did contain {0, 1e-6, 3e-6, 1e-5,
3e-5, 1e-4}, but the criterion was biased against all five nonzero entries
regardless of their scale. Normalizing the penalty does not fix that bias.
They also never call evaluate_optimal_emulator - there is no held-out set
anywhere in the sweep; each group is scored in-sample on itself.

This scores differently, in two ways:
  1. TRUE NRMSE, via recover_nrmse_trajectory, so the quantity compared across
     w is the error term alone and not the objective that grows with w.
  2. OUT OF SAMPLE, via evaluate_optimal_emulator on the held-out eval sets.
     Smoothing is almost always an in-sample COST - a rougher U fits its own
     training group better. Any benefit is in generalization, so that is where
     it has to be measured.

The output is a trade-off curve, not a single winner: for each w, the
out-of-sample NRMSE and the resulting roughness R. Choosing a point on that
curve is a modelling judgement and is left to the author.

Scope: Sulfur and multi only. The other four single-forcing agents already sit
at or below the roughest real ScenarioMIP profile (BC 0.2x, co2 1.0x, CH4 1.3x,
N2O 1.2x of real max) and see <0.03% penalty pressure at w=1e-4, so
regenerating them would cost compute to reproduce what they already are.

Usage:
    python scripts/6h_penalty_sweep.py --mode run-one --family Sulfur --w 1e-4 --seed 0
    python scripts/6h_penalty_sweep.py --mode collect
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import numpy as np

import utils_inverse

OUT_DIR = Path("data/SI_results/penalty_sweep")
CKPT_ROOT = Path("checkpoints/penalty_sweep")

# Bracketing the anchor derived in 6h_penalty_assessment.py: at w=1e-4 the
# penalty is ~8.5% of the multi-agent NRMSE and ~2% of Sulfur's, while staying
# under 0.03% for the four agents that are already smooth. Above 1e-3 the
# multi-agent penalty exceeds 85% of NRMSE and would dominate the objective.
# 12 log-spaced points across those two decades (ratio ~1.52 between
# neighbours) - dense enough to locate a knee in the trade-off curve rather
# than only bracket it.
W_GRID = [float(f"{w:.4g}") for w in np.logspace(-5, -3, 12)]

# --- 2026-08-26 extension, Sulfur only -------------------------------------
# The first grid topped out before the penalty bit. At w=1e-3 Sulfur's R was
# still 0.561, i.e. 5.8x the roughest real ScenarioMIP profile (0.0959), while
# the paired out-of-sample cost was indistinguishable from zero on Tier 1 /
# Tier 2 / All. Fitting R ~ w^-0.20 to the 1e-4 -> 1e-3 segment puts the real
# band near w ~ 1, so [1e-3, 1e-1] would have stopped short a second time.
# A coarse 1-2-5 ladder over three decades instead: wide enough to bracket
# saturation (where the NRMSE term stops mattering and U goes flat), coarse
# enough to stay cheap. multi is deliberately excluded - its R is flat and
# non-monotone across the first three decades, so more weight buys amplitude
# shrinkage, not smoothing.
W_GRID_EXT = [2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0]

# w=0 is the control and is NOT re-optimized: Stage C already produced exactly
# it. penalty_form only ever multiplies the penalty by smoothness_weight, so a
# legacy run at w=0 and a normalized run at w=0 are the same computation - the
# penalty term is identically zero either way. Re-running would spend 2 x
# n_seeds full 2000-iteration optimizations to reproduce bit-identical output.
# The control's out-of-sample metrics are still computed here, through the same
# evaluation path as every other w, so the comparison stays like-for-like.
W_CONTROL = 0.0
STAGE_C_CKPT = {
    "Sulfur": "checkpoints/Sulfur_retuned/seed_sweep/inverse_constant_tier1_Sulfur_only_seed{seed}.pkl",
    "multi": "checkpoints/multi_fig4/seed_sweep/inverse_constant_tier1_multi_fig4_seed{seed}.pkl",
}

NUM_UPDATES = 2000          # matches Stage C, so w is the only thing that varies
GROUP = "tier1"

FAMILIES = {
    "Sulfur": {
        "agents": ["Sulfur"], "active": ("Sulfur",), "tag": "Sulfur_only",
        "unified_cfg": "data/SI_results/hp_retune/Sulfur/best_config_unified.json",
        "baseline_cfg": "data/SI_results/baseline_hp/k400_search_Sulfur/best_baseline_config_K400.json",
        "T": 751, "init_cond": "constant", "filter_hist": False,
        "eval_agents": None,
    },
    "multi": {
        "agents": ["CO2", "CH4", "N2O", "Sulfur", "BC"],
        "active": ("CO2", "CH4", "N2O", "Sulfur", "BC"), "tag": "multi_fig4",
        "unified_cfg": "data/SI_results/hp_retune/multi/best_config_unified.json",
        "baseline_cfg": "data/SI_results/baseline_hp/k400_search_multi/best_baseline_config_K400.json",
        "T": 751, "init_cond": "constant", "filter_hist": False,
        "eval_agents": ["CO2", "CH4", "N2O", "Sulfur", "BC"],
    },
}


def roughness(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return float(np.sqrt(np.mean(np.diff(x) ** 2)) / s) if s > 0 else 0.0


def _wtag(w: float) -> str:
    return "0" if w == 0 else f"{w:g}".replace("-", "m").replace(".", "p")


def run_one(family: str, w: float, seed: int):
    cfg = FAMILIES[family]
    unified = json.load(open(cfg["unified_cfg"]))["config"]
    baseline = json.load(open(cfg["baseline_cfg"]))["config"]

    setup = utils_inverse.run_inverse_experiment_setup(
        cfg["agents"], cfg["active"], mode="FaIR",
        CS3=True, DAMIP=False, GeoMIP=False, idx_demo=None, seed=seed,
        baseline_K=baseline["K"], baseline_lr=baseline["lr"],
        baseline_weight_decay=baseline["weight_decay"],
    )
    groups = utils_inverse.build_group_emis_dicts(setup["emis_dict_train_JAX"],
                                                  setup["eval_sets"])

    if w == W_CONTROL:
        # Reuse Stage C's arm rather than recomputing an identical trajectory.
        ckpt = Path(STAGE_C_CKPT[family].format(seed=seed))
        if not ckpt.exists():
            raise FileNotFoundError(
                f"{ckpt} missing - the w=0 control reuses Stage C's 2000-iteration "
                f"checkpoints; run Stage C for {family} first")
        _evaluate_and_write(family, w, seed, cfg, setup, unified, ckpt)
        return

    ck_dir = CKPT_ROOT / family / f"w{_wtag(w)}"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ck_dir / f"inverse_constant_{GROUP}_{cfg['tag']}_seed{seed}.pkl"

    utils_inverse.optimize_emissions_inverse(
        groups[GROUP], setup["params0"],
        num_updates=NUM_UPDATES,
        step_size=unified["step_size"], momentum=unified["momentum"],
        nesterov=unified["nesterov"], K_inner=unified["K_inner"],
        lr_inner=unified["lr_inner"], wd_inner=unified["wd_inner"],
        agents=tuple(utils_inverse.AGENTS_DEFAULT), active_agents=cfg["active"],
        init_cond=cfg["init_cond"], T=cfg["T"], filter_hist=cfg["filter_hist"],
        mode="FaIR", checkpoint_path=str(ckpt), checkpoint_every=100,
        # A fresh objective: never resume onto a checkpoint written under a
        # different w, which would silently mix two objectives in one history.
        resume_if_exists=False,
        preds_every=100, batch_size=unified["batch_size"],
        key=jax.random.PRNGKey(seed),
        smoothness_weight=w, penalty_form="normalized",
    )

    _evaluate_and_write(family, w, seed, cfg, setup, unified, ckpt)


def _evaluate_and_write(family, w, seed, cfg, setup, unified, ckpt):
    """Out-of-sample evaluation + roughness, identical for every w including 0."""
    with open(ckpt, "rb") as f:
        raw = pickle.load(f)
    U_final = raw["U_traj"][-1]

    extra = {} if cfg["eval_agents"] is None else {"agents": cfg["eval_agents"]}
    oos = utils_inverse.evaluate_optimal_emulator(
        training_paths=[str(ckpt)], train_scenarios=["final"],
        eval_sets=setup["eval_sets"], params0=setup["params0"],
        active_agents=cfg["active"], inactive_mode="zeros",
        historical_name="historical", key=jax.random.PRNGKey(seed),
        K=unified["K_inner"], lr=unified["lr_inner"],
        weight_decay=unified["wd_inner"], mode="FaIR",
        batch_size=unified["batch_size"], **extra,
    )

    rec = {
        "family": family, "w": w, "seed": seed,
        # in-sample TRUE NRMSE (penalty removed), for the in/out comparison
        "nrmse_in_sample": float(utils_inverse.recover_nrmse_trajectory(raw)[-1]),
        "oos": {es: float(oos["final"][es]["mean"]) for es in oos["final"]},
        "baseline": {es: float(setup["baseline_results"][es]["mean"])
                     for es in setup["baseline_results"]},
        "R": {a: roughness(U_final[a]) for a in cfg["active"]},
        "P": float(utils_inverse.smoothness_penalty(U_final, "normalized", cfg["active"])),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{family}_w{_wtag(w)}_seed{seed}.json"
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"wrote {out}")


def collect():
    # Restrict to this sweep's own families. 6h_origin_config_probe.py writes
    # its arms into the same directory under pseudo-family names
    # (multi_origsteps / multi_retunedsteps); globbing everything would add
    # sections for them whose w values are absent from W_GRID, rendering rows
    # that silently go missing rather than failing.
    recs = []
    for f in sorted(OUT_DIR.glob("*_seed*.json")):
        with open(f) as fh:
            r = json.load(fh)
        if r.get("family") in FAMILIES:
            recs.append(r)
    if not recs:
        raise FileNotFoundError(f"no sweep results in {OUT_DIR}")

    real_band = {"Sulfur": 0.0959, "multi": 0.0959}   # roughest real profile
    for family in sorted({r["family"] for r in recs}):
        fam = [r for r in recs if r["family"] == family]
        eval_names = sorted(fam[0]["oos"])
        print(f"\n=== {family}: trade-off vs normalized weight w "
              f"(group={GROUP}, {NUM_UPDATES} iters) ===")
        print(f"{'w':>8s} {'n':>3s} {'R_med':>7s} {'P_med':>10s} {'in-samp':>9s} "
              + " ".join(f"{es:>9s}" for es in eval_names))
        base = {es: np.median([r["baseline"][es] for r in fam]) for es in eval_names}
        for w in [W_CONTROL] + W_GRID + W_GRID_EXT:
            sub = [r for r in fam if r["w"] == w]
            if not sub:
                continue
            Rmed = np.median([np.median(list(r["R"].values())) for r in sub])
            print(f"{w:8.0e} {len(sub):3d} {Rmed:7.4f} "
                  f"{np.median([r['P'] for r in sub]):10.3e} "
                  f"{np.median([r['nrmse_in_sample'] for r in sub]):9.5f} "
                  + " ".join(f"{np.median([r['oos'][es] for r in sub]):9.5f}"
                             for es in eval_names))
        print(f"{'base':>8s} {'':3s} {'':7s} {'':10s} {'':9s} "
              + " ".join(f"{base[es]:9.5f}" for es in eval_names))
        print(f"  (roughest real ScenarioMIP profile: R = {real_band[family]:.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["run-one", "collect"], default="collect")
    ap.add_argument("--family", choices=list(FAMILIES))
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
