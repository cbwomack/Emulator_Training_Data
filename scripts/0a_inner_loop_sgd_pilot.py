#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Phase 0, Stage 0a: controlled full-batch vs. minibatch (true SGD) comparison
for the inner-loop emulator training (train_mlp_sgd). Companion for
0a_inner_loop_sgd_pilot.ipynb.

Background: the manuscript's Methods (main text S1.1/1.2, SI Algorithm 1)
explicitly describe "Stochastic Gradient Descent" for the inner emulator
training, but the code has always been full-batch gradient descent with
momentum - optax's "sgd" optimizer is just the update-rule name, not a claim
about batch sampling, and neither loop ever shuffled/subsampled data. This
script tests whether adding genuine minibatch sampling (now available via
train_mlp_sgd's new `batch_size` parameter) helps, hurts, or is a wash,
before deciding whether it feeds into Stage 0b's hyperparameter retuning.

Two comparisons, in increasing order of scope:
  1. Standalone inner-loop fit: train on real CO2-only Tier-1 baseline
     training data (the same data create_baseline uses), full-batch vs.
     several batch_size choices, across several keys. This isolates the
     inner loop's own behavior without the confound of the outer bilevel
     optimizer.
  2. End-to-end sanity check: one short full outer-loop reoptimization run
     (reduced num_updates) with minibatching enabled inside the inner loop,
     to confirm the full pipeline doesn't crash/NaN when wired together.
     Caveat (documented, not fixed here): the outer loop's JIT'd carry does
     not thread a per-outer-step key, so within one run every outer step's
     inner-loop minibatch draws are seeded identically - this pilot tests
     "does a fixed-but-random minibatch subset change bilevel behavior,"
     not full outer-loop-varying stochasticity. That would require touching
     optimize_emissions_inverse's carry, out of scope for this first pilot.

Usage:
    python 0a_inner_loop_sgd_pilot.py --mode standalone
    python 0a_inner_loop_sgd_pilot.py --mode bilevel
"""
import os
import sys
import csv
import time
import pickle
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless/unattended run - never let a plt.show() block

import numpy as np
import jax
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import utils_inverse
import utils_FaIR_JAX

OUT_DIR = Path("data/SI_results/sgd_pilot")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGENTS = ["CO2"]
ACTIVE_AGENTS = ("CO2",)
MODE = "FaIR"
HIDDEN_SIZES = [16]
K = 400  # matches create_baseline's default training length


def build_real_training_data():
    """
    Reconstruct the exact CO2-only Tier-1 baseline training tensors
    (Xtr, ytr) that create_baseline uses internally, so this pilot tests
    real production-scale data rather than synthetic placeholders.
    """
    (emis_dict_train_FaIR, emis_dict_test_FaIR, emis_dict_train_JAX, emis_dict_test_JAX,
     delT_dict_train_FaIR, delT_dict_test_FaIR, delT_dict_train_JAX, delT_dict_test_JAX
     ) = utils_FaIR_JAX.generate_train_test(AGENTS, mode=MODE)

    train_data = utils_inverse.build_dataset_from_runfair_dict(emis_dict_train_JAX, mode=MODE)
    test_data = utils_inverse.build_dataset_from_runfair_dict(emis_dict_test_JAX, mode=MODE)
    train_s, _, _ = utils_inverse.split_and_scale(train_data, test_data)

    Xtr = jnp.concatenate([X for (X, _, _) in train_s], axis=0).astype(jnp.float32)
    ytr = jnp.concatenate([y for (_, y, _) in train_s], axis=0).astype(jnp.float32)
    return Xtr, ytr


def run_standalone_comparison(seeds, batch_sizes):
    Xtr, ytr = build_real_training_data()
    print(f"Real training data: Xtr {Xtr.shape}, ytr {ytr.shape}", flush=True)
    input_dim = Xtr.shape[1]

    log_rows = []
    for seed in seeds:
        params0 = utils_inverse.init_mlp_params(
            jax.random.PRNGKey(seed), input_dim=input_dim, hidden_sizes=HIDDEN_SIZES,
        )
        for bs in batch_sizes:
            label = "full-batch" if bs is None else f"batch_size={bs}"
            t0 = time.perf_counter()
            _, losses = utils_inverse.train_mlp_sgd(
                params0, Xtr, ytr, K=K, batch_size=bs, key=jax.random.PRNGKey(1000 + seed),
            )
            dt = time.perf_counter() - t0
            losses_np = np.asarray(losses)
            has_nan = bool(np.isnan(losses_np).any() or np.isinf(losses_np).any())
            row = {
                "seed": seed, "batch_size": "full" if bs is None else bs,
                "loss0": float(losses_np[0]), "lossF": float(losses_np[-1]),
                "stable": not has_nan, "seconds": round(dt, 2),
            }
            log_rows.append(row)
            print(f"[seed={seed} {label}] loss0={row['loss0']:.4g} lossF={row['lossF']:.4g} "
                  f"stable={row['stable']} ({dt:.1f}s)", flush=True)

    log_path = OUT_DIR / "standalone_comparison.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n=== Standalone comparison done: {len(log_rows)} runs. Log: {log_path} ===")
    return log_rows


def _bs_label(bs):
    return "full" if bs is None else str(bs)


def run_bilevel_scaled(seeds, batch_sizes, num_updates):
    """
    Scaled-up end-to-end bilevel comparison: for every (batch_size, seed)
    combination, run the real H-ext CO2-only outer-loop optimization for the
    full `num_updates` steps, saving the complete per-outer-step error
    trajectory (not just loss0/lossF) so the caller can plot performance vs.
    training step with spread across seeds, one plot per batch size, against
    the full-batch (batch_size=None) run as the same-protocol baseline.

    Idempotent/resumable at two levels:
      1. Per-(batch_size, seed) run: uses optimize_emissions_inverse's own
         checkpoint_path/resume_if_exists=True, so an interrupted individual
         run picks back up rather than restarting from step 0.
      2. Across the whole sweep: a completed run's summary row is appended to
         summary_path immediately; on restart, any (batch_size, seed) already
         present there (with a matching num_updates) is skipped entirely.
    """
    summary_path = OUT_DIR / "bilevel_scaled_summary.csv"
    traj_path = OUT_DIR / "bilevel_scaled_errors.npz"

    done = set()
    if summary_path.is_file():
        with open(summary_path) as f:
            for row in csv.DictReader(f):
                if int(row["num_updates"]) == num_updates:
                    done.add((row["batch_size"], int(row["seed"])))

    trajectories = dict(np.load(traj_path)) if traj_path.is_file() else {}

    fieldnames = ["batch_size", "seed", "num_updates", "loss0", "lossF", "stable", "seconds"]
    write_header = not summary_path.is_file()
    summary_f = open(summary_path, "a", newline="")
    writer = csv.DictWriter(summary_f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        summary_f.flush()

    total = len(batch_sizes) * len(seeds)
    done_count = 0
    for bs in batch_sizes:
        bs_lab = _bs_label(bs)
        for seed in seeds:
            done_count += 1
            if (bs_lab, seed) in done:
                print(f"[{done_count}/{total}] skip batch_size={bs_lab} seed={seed} (already done)", flush=True)
                continue

            setup = utils_inverse.run_inverse_experiment_setup(
                AGENTS, ACTIVE_AGENTS, mode=MODE, CS3=True, DAMIP=False, GeoMIP=False,
                seed=seed, idx_demo=None,
            )
            group_emis_dicts = utils_inverse.build_group_emis_dicts(
                setup["emis_dict_train_JAX"], setup["eval_sets"],
            )
            ckpt_path = OUT_DIR / f"bilevel_scaled_bs{bs_lab}_seed{seed}.pkl"
            t0 = time.perf_counter()
            result = utils_inverse.optimize_emissions_inverse(
                emis_dict=group_emis_dicts["H-ext"],
                params0=setup["params0"],
                num_updates=num_updates,
                step_size=1000.0,
                momentum=0.9,
                nesterov=True,
                K_inner=400,
                lr_inner=0.05,
                wd_inner=0.01,
                active_agents=ACTIVE_AGENTS,
                init_cond="sine",
                T=477,
                filter_hist=True,
                smoothness_weight=1e-05,
                mode=MODE,
                checkpoint_path=str(ckpt_path),
                checkpoint_every=100,
                resume_if_exists=True,
                preds_every=100,
                batch_size=bs,
            )
            dt = time.perf_counter() - t0
            errors = np.asarray(result["errors"])
            stable = not bool(np.isnan(errors).any() or np.isinf(errors).any())
            print(f"[{done_count}/{total}] batch_size={bs_lab} seed={seed} "
                  f"loss0={errors[0]:.4g} lossF={errors[-1]:.4g} stable={stable} "
                  f"({dt:.1f}s, {len(errors)} steps)", flush=True)

            trajectories[f"bs_{bs_lab}_seed{seed}"] = errors
            np.savez(traj_path, **trajectories)

            writer.writerow({
                "batch_size": bs_lab, "seed": seed, "num_updates": num_updates,
                "loss0": float(errors[0]), "lossF": float(errors[-1]),
                "stable": stable, "seconds": round(dt, 2),
            })
            summary_f.flush()

    summary_f.close()
    print(f"\n=== Scaled bilevel comparison done: {total - len(done)} new runs "
          f"({len(done)} skipped as already-complete). Summary: {summary_path}, "
          f"trajectories: {traj_path} ===")


def _parse_batch_sizes(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        out.append(None if tok.lower() == "full" else int(tok))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["standalone", "bilevel-scaled"], default="standalone")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--batch-sizes", type=str, default="full,16,32,64",
                         help="Comma-separated list for --mode bilevel-scaled; 'full' means batch_size=None")
    parser.add_argument("--num-updates", type=int, default=1000, help="Used only for --mode bilevel-scaled")
    args = parser.parse_args()

    seeds = list(range(args.n_seeds))
    if args.mode == "standalone":
        run_standalone_comparison(seeds, batch_sizes=[None, 16, 32, 64, 128])
    else:
        run_bilevel_scaled(seeds, batch_sizes=_parse_batch_sizes(args.batch_sizes), num_updates=args.num_updates)


if __name__ == "__main__":
    main()
