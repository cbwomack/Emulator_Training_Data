#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Phase 0, Stage 6e Part A (moved up from Phase 1's reviewer-response work):
plot the baseline emulator's own per-step training loss curve, using
train_baseline_emulator's existing (mean_loss, losses) return value - no
new pipeline compute needed beyond running the baseline training itself
(cheap: K MLP-only gradient steps, no outer bilevel loop). Shows whether
the current K=400 training length has actually plateaued (well-trained) or
is still descending steeply (undertrained) - directly relevant to whether
Part B's independent K search matters.

Usage:
    python 6e_baseline_convergence.py
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cmcrameri import cm

import numpy as np
import jax

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import utils_inverse

OUT_DIR = Path("data/SI_results/baseline_hp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGENTS = ["CO2"]
ACTIVE_AGENTS = ("CO2",)
MODE = "FaIR"
HIDDEN_SIZES = [16]
K_CURRENT = 400  # the pipeline's current, un-independently-tuned baseline K
SEEDS = [0, 1, 2, 3, 4]


def main():
    setup = utils_inverse.run_inverse_experiment_setup(
        AGENTS, ACTIVE_AGENTS, mode=MODE, CS3=True, DAMIP=False, GeoMIP=False,
        seed=0, idx_demo=None,
    )
    train_s, _test_s, _stats = utils_inverse.prepare_baseline_data(
        emis_dict_train=setup["emis_dict_train_JAX"], emis_dict_test=setup["emis_dict_train_JAX"], mode=MODE,
    )

    curves = []
    for seed in SEEDS:
        paramsK, (mean_loss, losses), meta = utils_inverse.train_baseline_emulator(
            train_scaled=train_s, key=jax.random.PRNGKey(seed), hidden_sizes=HIDDEN_SIZES,
            K=K_CURRENT, lr=5e-2, weight_decay=1e-2,
        )
        losses_np = np.asarray(losses)
        print(f"seed={seed}: loss0={losses_np[0]:.4g} lossF={losses_np[-1]:.4g} "
              f"min={losses_np.min():.4g} at step {losses_np.argmin()}")
        curves.append(losses_np)
    curves = np.stack(curves, axis=0)  # (n_seeds, K)
    np.save(OUT_DIR / "convergence_curves_K400.npy", curves)

    steps = np.arange(curves.shape[1])
    mean_curve = curves.mean(axis=0)
    lo, hi = curves.min(axis=0), curves.max(axis=0)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.fill_between(steps, lo, hi, color=cm.batlowS(0), alpha=0.25, label="seed min-max (n=5)")
    ax.loglog(steps + 1, mean_curve, color=cm.batlowS(0), lw=2, label="mean")
    ax.set_xlabel("training step")
    ax.set_ylabel("baseline MLP training MSE loss")
    ax.set_title(f"Baseline emulator training convergence (K={K_CURRENT}, lr=0.05, weight_decay=0.01)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.savefig(OUT_DIR / "convergence_K400.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved plot -> {OUT_DIR / 'convergence_K400.png'}")

    # Plateau diagnostic: how much did the loss improve in the last 10% of
    # training vs. the first 10%? A small fraction means it has plateaued.
    tail_frac = int(0.1 * curves.shape[1])
    early_drop = mean_curve[0] - mean_curve[tail_frac]
    late_drop = mean_curve[-tail_frac - 1] - mean_curve[-1]
    print(f"\nPlateau check: loss drop in first 10% of training = {early_drop:.4g}, "
          f"drop in last 10% = {late_drop:.4g} ({100*late_drop/max(early_drop,1e-12):.2f}% of the early rate)")


if __name__ == "__main__":
    main()
