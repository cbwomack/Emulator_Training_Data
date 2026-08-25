#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage A - pre-flight gate for the 1000 -> 2000 iteration extension.

Figures 3 and 5 plot the bilevel objective evaluated IN-SAMPLE on the very
scenario group being optimized, so it falls by construction and running longer
improves it whether or not the emulator is actually getting better. Figure 4's
fresh-retrain path is the out-of-sample measure, and it could move either way:
more outer iterations means the emissions profile is fitted harder to the
target group, which is precisely how a bilevel optimizer overfits.

This evaluates out-of-sample NRMSE at a series of outer iterates U_traj[k] for
the same checkpoints, so the decision to extend rests on generalization rather
than on the training curve.

**Pre-registered decision rule** (fixed before the runs, per the plan):
  - out-of-sample NRMSE still FALLING at k=1000  -> extending to 2000 is
    justified; proceed to Stage C.
  - already TURNED (rising, or flat within seed noise) -> extending makes
    Figure 4 worse, not better. Stop and reconsider.

The same rule and the same run length apply to every agent. This matters: at
1000 iterations N2O and multi-agent fail to beat their baselines while a tail
extrapolation to 2000 says all six would, so choosing the stopping point after
seeing which value makes the result positive would be exactly the kind of
post-hoc selection a reviewer should object to.

Usage:
    python scripts/6h_convergence_preflight.py                    # co2 + multi, 5 seeds
    python scripts/6h_convergence_preflight.py --seeds 3 --quick  # smaller/faster

    # sharded (one SLURM array task per family/seed), then aggregate:
    python scripts/6h_convergence_preflight.py --family co2 --seed-list 0 --mode shard
    python scripts/6h_convergence_preflight.py --mode collect

Memory note: run_inverse_experiment_setup peaks at ~2.8 GB before any iterate is
evaluated, which is why this does not fit in the dev container (~2 GB free) and
is sharded onto the cluster at --mem=8G instead. The checkpoints themselves are
only 15 MB and are not the constraint.
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import json

import jax
import numpy as np

import utils_inverse

OUT_DIR = Path("data/SI_results/convergence_preflight")

# Which outer iterates to probe. 0 is the initial condition (the Stage 6b
# ablation's comparand); the rest span the run so a turning point is visible
# rather than inferred from the endpoints alone.
DEFAULT_ITERATES = [0, 100, 200, 400, 600, 800, 1000]

# Setup/eval arguments are copied from the corresponding production cache
# builders so this probe is measuring the same quantity Figure 4 reports, just
# at a different iterate. Note the asymmetry, which is load-bearing and matches
# utils_inverse: CO2-only passes ['CO2'] to setup but does NOT pass `agents` to
# evaluate_optimal_emulator (it keeps the full 5-agent feature space with the
# other four zeroed via active_agents), whereas the multi-agent path passes all
# five to both. Narrowing `agents` for CO2 produces a 5-column feature matrix
# against a 25-column params0 and fails outright.
FAMILIES = {
    "co2": {
        # mirrors utils_inverse.regenerate_fig4_co2_only_cache_seed_sweep
        "checkpoint_dir": "checkpoints/co2_retuned/seed_sweep",
        "tag": "co2_only",
        "setup_agents": ["CO2"],
        "eval_agents": None,          # -> AGENTS_DEFAULT, all five
        "active": ("CO2",),
        "unified_cfg": "data/SI_results/hp_retune/best_config_unified.json",
        "baseline_cfg": "data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json",
    },
    "multi": {
        # mirrors utils_inverse.regenerate_fig4_all_agents_cache_seed_sweep
        "checkpoint_dir": "checkpoints/multi_fig4/seed_sweep",
        "tag": "multi_fig4",
        "setup_agents": ["CO2", "CH4", "N2O", "Sulfur", "BC"],
        "eval_agents": ["CO2", "CH4", "N2O", "Sulfur", "BC"],
        "active": ("CO2", "CH4", "N2O", "Sulfur", "BC"),
        "unified_cfg": "data/SI_results/hp_retune/multi/best_config_unified.json",
        "baseline_cfg": "data/SI_results/baseline_hp/k400_search_multi/best_baseline_config_K400.json",
    },
}

# The out-of-sample eval sets to report. 'Tier 1' is the optimization target
# here (so it is the in-objective one, kept for reference); the others are the
# genuinely held-out ones that decide the gate.
TARGET_SET = "Tier 1"


def evaluate_iterates(family, seeds, iterates, group="tier1"):
    cfg = FAMILIES[family]
    unified = json.load(open(cfg["unified_cfg"]))["config"]
    baseline = json.load(open(cfg["baseline_cfg"]))["config"]

    out = {}
    for seed in seeds:
        # Per-seed setup, matching the production builders: the baseline shares
        # this seed's init, so comparisons stay within-seed.
        setup = utils_inverse.run_inverse_experiment_setup(
            cfg["setup_agents"], cfg["active"], mode="FaIR",
            CS3=True, DAMIP=False, GeoMIP=False,
            idx_demo=None, seed=seed,
            baseline_K=baseline["K"], baseline_lr=baseline["lr"],
            baseline_weight_decay=baseline["weight_decay"],
        )
        ckpt = f"{cfg['checkpoint_dir']}/inverse_constant_{group}_{cfg['tag']}_seed{seed}.pkl"
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"{ckpt} missing")

        extra = {} if cfg["eval_agents"] is None else {"agents": cfg["eval_agents"]}
        out[seed] = {"_baseline": {es: float(setup["baseline_results"][es]["mean"])
                                   for es in setup["baseline_results"]}}
        for k in iterates:
            t0 = time.time()
            res = utils_inverse.evaluate_optimal_emulator(
                training_paths=[ckpt],
                train_scenarios=[f"iter{k}"],
                eval_sets=setup["eval_sets"],
                params0=setup["params0"],
                active_agents=cfg["active"],
                inactive_mode="zeros",
                historical_name="historical",
                key=jax.random.PRNGKey(seed),
                K=unified["K_inner"], lr=unified["lr_inner"], weight_decay=unified["wd_inner"],
                mode="FaIR",
                batch_size=unified["batch_size"],
                u_index=k,
                **extra,
            )
            out[seed][k] = {es: float(res[f"iter{k}"][es]["mean"]) for es in res[f"iter{k}"]}
            print(f"  [{family}] seed={seed} iter={k:5d} "
                  + "  ".join(f"{es}={out[seed][k][es]:.5f}" for es in sorted(out[seed][k]))
                  + f"   ({time.time()-t0:.1f}s)", flush=True)
    return out


def summarize(family, out, iterates):
    """Median across seeds per iterate, and the verdict on the pre-registered rule."""
    eval_names = sorted(out[next(iter(out))][iterates[0]].keys())
    print(f"\n=== {family}: out-of-sample NRMSE, median across {len(out)} seeds ===")
    header = f"{'iter':>6s} " + " ".join(f"{es:>11s}" for es in eval_names)
    print(header)
    med = {}
    for k in iterates:
        med[k] = {es: float(np.median([out[s][k][es] for s in out])) for es in eval_names}
        print(f"{k:6d} " + " ".join(f"{med[k][es]:11.5f}" for es in eval_names))
    base = {es: float(np.median([out[s]["_baseline"][es] for s in out if es in out[s]["_baseline"]]))
            for es in eval_names}
    print(f"{'base':>6s} " + " ".join(f"{base.get(es, float('nan')):11.5f}" for es in eval_names)
          + "   <- baseline emulator (median across seeds)")

    print(f"\n--- verdict ({family}) ---")
    verdicts = {}
    for es in eval_names:
        curve = np.array([med[k][es] for k in iterates])
        # Compare the last probed interval: still falling means the run had not
        # finished extracting generalization by iteration 1000.
        last_delta = curve[-1] - curve[-2]
        rel = last_delta / curve[-2] * 100
        best_k = iterates[int(np.argmin(curve))]
        still_falling = last_delta < 0
        verdicts[es] = {"still_falling": bool(still_falling), "last_delta_pct": float(rel),
                        "argmin_iterate": int(best_k), "final": float(curve[-1]),
                        "min": float(curve.min())}
        flag = "FALLING" if still_falling else "TURNED"
        print(f"  {es:11s} {flag:8s} last step {rel:+7.2f}%   best at iter {best_k:5d}"
              + ("   <-- minimum is NOT at the end" if best_k != iterates[-1] else ""))
    return med, verdicts


def _report(all_out):
    """Print the per-family tables and the pre-registered gate summary."""
    print("\n" + "=" * 62)
    print("GATE (pre-registered): extend to 2000 only if out-of-sample NRMSE")
    print("is still falling at iteration 1000 on the held-out eval sets.")
    print("=" * 62)
    for fam, d in all_out.items():
        held_out = [es for es in d["verdicts"] if es != TARGET_SET]
        n_falling = sum(d["verdicts"][es]["still_falling"] for es in held_out)
        print(f"  {fam:6s}: {n_falling}/{len(held_out)} held-out eval sets still falling "
              f"(in-objective '{TARGET_SET}': "
              f"{'falling' if d['verdicts'][TARGET_SET]['still_falling'] else 'turned'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["run", "shard", "collect"], default="run",
                    help="run: everything in one process. shard: one family/seed, "
                         "writes a per-shard pickle for a SLURM array. collect: "
                         "aggregate the shards and apply the gate.")
    ap.add_argument("--family", choices=list(FAMILIES) + ["both"], default="both")
    ap.add_argument("--seeds", type=int, default=5, help="use seeds 0..N-1")
    ap.add_argument("--seed-list", type=str, default=None,
                    help="explicit comma-separated seeds; overrides --seeds")
    ap.add_argument("--quick", action="store_true", help="probe only 0/500/1000")
    args = ap.parse_args()

    iterates = [0, 500, 1000] if args.quick else DEFAULT_ITERATES
    seeds = ([int(s) for s in args.seed_list.split(",")] if args.seed_list
             else list(range(args.seeds)))
    families = list(FAMILIES) if args.family == "both" else [args.family]
    shard_dir = OUT_DIR / "shards"

    if args.mode == "collect":
        # Re-derive the medians/verdicts from the union of the shards, so the
        # gate is applied to the pooled seeds rather than per-shard.
        pooled = {}
        for f in sorted(shard_dir.glob("*.pkl")):
            with open(f, "rb") as fh:
                sh = pickle.load(fh)
            pooled.setdefault(sh["family"], {}).update(sh["raw"])
        if not pooled:
            raise FileNotFoundError(f"no shards in {shard_dir}")
        all_out = {}
        for fam, raw in pooled.items():
            its = sorted(k for k in next(iter(raw.values())) if isinstance(k, int))
            med, verdicts = summarize(fam, raw, its)
            all_out[fam] = {"raw": raw, "median": med, "verdicts": verdicts,
                            "iterates": its, "seeds": sorted(raw)}
        with open(OUT_DIR / "preflight_results.pkl", "wb") as f:
            pickle.dump(all_out, f)
        print(f"\nwrote {OUT_DIR / 'preflight_results.pkl'}")
        _report(all_out)
        return

    if args.mode == "shard":
        if args.family == "both" or len(seeds) != 1:
            ap.error("--mode shard needs exactly one --family and one --seed-list value")
        shard_dir.mkdir(parents=True, exist_ok=True)
        fam = families[0]
        raw = evaluate_iterates(fam, seeds, iterates)
        out = shard_dir / f"{fam}_seed{seeds[0]}.pkl"
        with open(out, "wb") as f:
            pickle.dump({"family": fam, "raw": raw, "iterates": iterates}, f)
        print(f"wrote {out}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_out = {}
    for fam in families:
        print(f"\n########## {fam} ##########", flush=True)
        raw = evaluate_iterates(fam, seeds, iterates)
        med, verdicts = summarize(fam, raw, iterates)
        all_out[fam] = {"raw": raw, "median": med, "verdicts": verdicts,
                        "iterates": iterates, "seeds": seeds}

    with open(OUT_DIR / "preflight_results.pkl", "wb") as f:
        pickle.dump(all_out, f)
    print(f"\nwrote {OUT_DIR / 'preflight_results.pkl'}")
    _report(all_out)


if __name__ == "__main__":
    main()
