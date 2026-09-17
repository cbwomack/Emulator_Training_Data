#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Generates the LaTeX for SUPPLEMENT_TABLES_FIG7_REVISION_LATEX.md, on the same
first-50-seed slice of data/SI_results/seed_uncertainty/fig7_revision_1000seed.pkl
that scripts/6u_fig7_revision_final_plot.py renders as Figure 7.

Aggregation - the point of this script
--------------------------------------
The first version of this table read each eval set's pre-computed
`results[set]['mean']['global']`, which utils_inverse.
evaluate_emulator_vector_over_multiple_tests builds as

    np.average(errs_global_list, weights=lens)          # lens = n timesteps

i.e. a *timestep-length-weighted mean of raw NRMSE values*, and then took the
percent change of that. That is a legitimate quantity, but it is NOT the average
of the bars Figure 7 draws, and for DECK the two disagree in sign: `2xCO2`'s
baseline NRMSE is ~8.6x `1pctCO2`'s and it runs 250 years vs. 150, so averaging
the NRMSEs first lets `2xCO2` dominate and a config that clearly wins on
`1pctCO2` still reads as a loss.

This script instead aggregates exactly the way the figure does, so the table and
the figure are the same statistic at two levels of detail:

  1. per seed, per scenario, paired on that seed's own baseline run:
         pct = 100 * (NRMSE_base - NRMSE_opt) / NRMSE_base
     - identical to utils_plotting.plot_scenario_difference_bars2's
       `_pct_change_row`, which is what the bars are;
  2. per seed, per eval set: the *unweighted* mean of that set's scenario pcts
     - the average of the bars in that group of the figure;
  3. across the 50 seeds: 25th/50th/75th percentile.

Step 2 is unweighted on purpose: every bar in the figure is one scenario drawn at
one size, so the table's group number has to weight them equally to be the
average of what the reader sees.

The baseline rows follow the same convention - the unweighted mean across the
set's scenarios of baseline NRMSE, not the length-weighted `['mean']['global']` -
so the two halves of the table are averaged the same way and the percent columns
are referenced to the number printed above them.

Usage:
    python scripts/6v_fig7_revision_supplement_table.py            # print LaTeX
    python scripts/6v_fig7_revision_supplement_table.py --compare  # old vs new
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

SEED_CACHE_PATH = "data/SI_results/seed_uncertainty/fig7_revision_1000seed.pkl"
FIRST_N = 50

# Eval sets in figure order, each with its scenarios in figure (left-to-right)
# order. Matches scripts/6u_fig7_revision_final_plot.py's `scenario_keys` /
# `separator_indices` grouping exactly - 16 bars, 7/5/2/2.
EVAL_SETS = {
    "Tier 1": ["historical", "H-ext", "M", "ML", "L", "VLLO-ext", "VLHO"],
    "Tier 2": ["H-ext-OS", "M-ext", "ML-ext", "L-ext", "VLHO-ext"],
    "DECK":   ["1pctCO2", "2xCO2"],
    "CS3":    ["AA", "CT"],
}
CONFIGS = [("optimal_constant", "Opt. Constant"),
           ("optimal_sine",     "Opt. Sine"),
           ("optimal_both",     "Opt. Both")]

PCTS = (25, 50, 75)


def load_seeds():
    with open(SEED_CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    return cache, list(range(FIRST_N))


def scen_pct(cache, seed, cfg, eset, scen):
    """Per-seed, per-scenario paired % change - one bar of Figure 7."""
    b = cache[seed]["baseline"][eset][scen]["global"]
    o = cache[seed][cfg][eset][scen]["global"]
    return 100.0 * (b - o) / b


def set_pct_per_seed(cache, seeds, cfg, eset):
    """Per-seed unweighted mean of that set's scenario % changes."""
    return np.array([np.mean([scen_pct(cache, s, cfg, eset, sc)
                              for sc in EVAL_SETS[eset]])
                     for s in seeds])


def set_baseline_nrmse_per_seed(cache, seeds, eset):
    """Per-seed unweighted mean of that set's scenario baseline NRMSEs."""
    return np.array([np.mean([cache[s]["baseline"][eset][sc]["global"]
                              for sc in EVAL_SETS[eset]])
                     for s in seeds])


def q(arr):
    return [float(np.percentile(arr, p)) for p in PCTS]


def fmt_nrmse(v):
    return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"


def fmt_pct(v):
    return "[" + ", ".join(f"{x:+.1f}\\%" for x in v) + "]"


def emit_latex(cache, seeds):
    lines = []
    lines.append(r"\begin{table}[b!]")
    lines.append(r"    \centering")
    lines.append(r"    \renewcommand{\arraystretch}{1.5}")
    lines.append(r"""    \caption{Numerical results for the revised main text Figure 7 (\gls{mesm} summary), using
    the MESM outputs regenerated for the manuscript revisions. Values are the 25th/50th
    (median)/75th percentile across the first 50 of 1000 stochastic seeds --- an arbitrary,
    unselected sample, not a favorably-chosen subset. \textbf{Baseline} reports its own
    emulator evaluation loss (\gls{nrmse}), averaged with equal weight over the scenarios in
    each evaluation set. Each \textbf{optimized} configuration instead reports \% change in
    skill relative to baseline, aggregated exactly as in the figure: for each seed and each
    scenario, $100 \times (\mathrm{NRMSE}_{\mathrm{baseline}} - \mathrm{NRMSE}_{\mathrm{opt}}) /
    \mathrm{NRMSE}_{\mathrm{baseline}}$, paired on that seed's own baseline run (one bar of
    Figure 7), then averaged with equal weight across the scenarios in the evaluation set, so
    each tabulated value is the mean of the corresponding group of bars. Positive values mean
    the optimized configuration outperforms baseline. The optimized configurations
    (Constant/Sine/Both) refer to the initial-condition strategy used for the upstream
    \gls{scm}-side CO$_2$ emissions optimization that produced each \gls{mesm} training
    trajectory; all three are optimized against the combined Tier 1+Tier 2+DECK+CS3 objective,
    so --- as in the original table --- no cell here is evaluated on a strictly disjoint
    out-of-objective set. No rows are shaded: unlike Figure 6, no cell in this table is a
    direct (in-objective) optimization target.}""")
    lines.append(r"    \begin{tabular}{l l r}")
    lines.append(r"        \textbf{Emulator Config.} & \textbf{Eval. Data} & "
                 r"\textbf{Value [25th, 50th, 75th]}\\")
    lines.append(r"        \hline")

    for i, eset in enumerate(EVAL_SETS):
        name = "Baseline (NRMSE)" if i == 0 else ""
        v = q(set_baseline_nrmse_per_seed(cache, seeds, eset))
        lines.append(f"        {name} & {eset} & {fmt_nrmse(v)} \\\\"
                     if i == 0 else
                     f"         & {eset} & {fmt_nrmse(v)} \\\\")

    for cfg, label in CONFIGS:
        lines.append(r"        \cline{1-3}")
        for i, eset in enumerate(EVAL_SETS):
            v = q(set_pct_per_seed(cache, seeds, cfg, eset))
            head = f"{label} (\\% chg.)" if i == 0 else ""
            lines.append(f"        {head} & {eset} & {fmt_pct(v)} \\\\"
                         if i == 0 else
                         f"         & {eset} & {fmt_pct(v)} \\\\")

    lines.append(r"    \end{tabular}")
    lines.append(r"    \label{tab:supp_num_fig7_revision}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def emit_latex_per_scenario(cache, seeds):
    """Per-scenario companion: one row per bar of Figure 7, fully reconstructible."""
    lines = []
    lines.append(r"\begin{table}[b!]")
    lines.append(r"    \centering")
    lines.append(r"    \renewcommand{\arraystretch}{1.3}")
    lines.append(r"""    \caption{Per-scenario breakdown of the revised main text Figure 7 --- one row per bar.
    \textbf{Baseline \gls{nrmse}} and \textbf{\% change} are defined as in
    Table~\ref{tab:supp_num_fig7_revision}, but reported for each individual scenario rather
    than averaged over an evaluation set; all values are medians across the first 50 of 1000
    stochastic seeds, with the \% changes paired on each seed's own baseline run. Positive \%
    values mean the optimized configuration outperforms baseline.}""")
    lines.append(r"    \begin{tabular}{l l r r r r}")
    lines.append(r"        \textbf{Eval. Data} & \textbf{Scenario} & \textbf{Baseline NRMSE} & "
                 r"\textbf{Constant} & \textbf{Sine} & \textbf{Both}\\")
    lines.append(r"        \hline")
    for eset, scens in EVAL_SETS.items():
        for j, sc in enumerate(scens):
            base = np.median([cache[s]["baseline"][eset][sc]["global"] for s in seeds])
            cells = []
            for cfg, _ in CONFIGS:
                m = np.median([scen_pct(cache, s, cfg, eset, sc) for s in seeds])
                cells.append(f"{m:+.1f}\\%")
            head = eset if j == 0 else ""
            disp = sc.replace("2xCO2", "abrupt-2xCO2").replace("_", "-")
            lines.append(f"        {head} & \\textit{{{disp}}} & {base:.4f} & "
                         + " & ".join(cells) + r" \\")
        lines.append(r"        \cline{1-6}")
    lines.pop()  # trailing cline
    lines.append(r"    \end{tabular}")
    lines.append(r"    \label{tab:supp_num_fig7_revision_scenario}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def compare(cache, seeds):
    """Old (length-weighted mean-NRMSE ratio) vs. new (mean of bars) medians."""
    print(f"{'set':8s} {'config':16s} {'OLD %chg':>10s} {'NEW %chg':>10s} {'delta':>8s}")
    for eset in EVAL_SETS:
        for cfg, label in CONFIGS:
            old = np.median([100.0 * (cache[s]["baseline"][eset]["mean"]["global"]
                                      - cache[s][cfg][eset]["mean"]["global"])
                             / cache[s]["baseline"][eset]["mean"]["global"] for s in seeds])
            new = np.median(set_pct_per_seed(cache, seeds, cfg, eset))
            flag = "  <-- sign flip" if old * new < 0 else ""
            print(f"{eset:8s} {label:16s} {old:+10.1f} {new:+10.1f} {new-old:+8.1f}{flag}")
    print()
    print("baseline NRMSE, length-weighted 'mean' vs. unweighted scenario mean (median over seeds):")
    for eset in EVAL_SETS:
        old = np.median([cache[s]["baseline"][eset]["mean"]["global"] for s in seeds])
        new = np.median(set_baseline_nrmse_per_seed(cache, seeds, eset))
        print(f"  {eset:8s} {old:.4f} -> {new:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true",
                    help="print the old-vs-new aggregation comparison instead of LaTeX")
    ap.add_argument("--per-scenario", action="store_true",
                    help="also emit the per-scenario companion table")
    args = ap.parse_args()

    cache, seeds = load_seeds()
    if args.compare:
        compare(cache, seeds)
        return
    print(emit_latex(cache, seeds))
    if args.per_scenario:
        print()
        print(emit_latex_per_scenario(cache, seeds))


if __name__ == "__main__":
    main()
