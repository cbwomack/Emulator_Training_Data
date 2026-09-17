#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
PREVIEW ONLY - an alternate layout for the Figure 7 revision (first-50-seeds)
plot, requested as a one-off comparison, not a replacement:

  - Row 1: (a) and (b) - the same constant-IC / sine-IC emissions + real-MESM
    ΔT(t) time-series panels scripts/6s_fig7_revision_first50_plot.py draws -
    side by side instead of stacked.
  - Row 2: (c) - the same bar chart, rotated 90 degrees: vertical bars
    instead of horizontal, scenarios along the x-axis, % change along the
    y-axis - increased skill above the zero line, decreased skill below.

Deliberately does NOT call utils_plotting.plot_scenario_difference_bars2 or
modify it - that function's fixed 3-row (stacked) layout is the existing,
published Figure 7 convention and is left untouched. This script duplicates
the relevant drawing logic from that function (verbatim where the layout
doesn't change) so the two can be compared side by side, per user request.

Usage:
    python 6t_fig7_revision_first50_altlayout_preview.py
"""
import os
import sys
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from cmcrameri import cm

import utils_plotting
from utils_plotting import get_global_value
from paths import DATA_DIR, FIGURES_DIR

SEED_CACHE_PATH = "data/SI_results/seed_uncertainty/fig7_revision_1000seed.pkl"
REV_ICS = ["constant", "sine", "both"]
TOP_PANEL_ICS = ["constant", "sine"]
FIRST_N = 50


def load_top_panel_data():
    """Identical to 6s_fig7_revision_first50_plot.py."""
    co2_data = []
    for ic in TOP_PANEL_ICS:
        e = np.loadtxt(f"{DATA_DIR}/MESM/emis_driven/MESM_inputs/opt_all_revision_{ic}.txt",
                       usecols=(2,), skiprows=2)
        co2_data.append(e)

    lat_coords = np.linspace(-88, 88, 46)
    lat_weights = np.cos(np.deg2rad(lat_coords))
    lat_weights = np.maximum(lat_weights, 1e-6)
    lat_weights = lat_weights / np.sum(lat_weights)

    global_mean_temp = []
    for ic in TOP_PANEL_ICS:
        with open(f"data/MESM/emis_driven/zonal_data_mean/optimized_revision/"
                  f"opt_all_revision_{ic}_mean.pkl", "rb") as f:
            zonal = pickle.load(f)
        global_mean_temp.append(np.average(zonal, weights=lat_weights, axis=1))

    return co2_data, global_mean_temp


def main():
    with open(SEED_CACHE_PATH, "rb") as f:
        seed_cache = pickle.load(f)
    seed_idx = list(range(FIRST_N))
    print(f"using seeds {seed_idx[0]}-{seed_idx[-1]} of {len(seed_cache)}")

    seed_baseline_results = [seed_cache[s]["baseline"] for s in seed_idx]
    seed_optimized_results_list = [[seed_cache[s][f"optimal_{ic}"] for s in seed_idx] for ic in REV_ICS]
    baseline_results = seed_baseline_results[0]
    optimized_results_list = [lst[0] for lst in seed_optimized_results_list]

    co2_data, global_mean_temp = load_top_panel_data()

    scenario_keys = ['historical', 'H-ext', 'M', 'ML', 'L', 'VLLO-ext', 'VLHO',
                     'H-ext-OS', 'M-ext', 'ML-ext', 'L-ext', 'VLHO-ext',
                     '2xCO2', '1pctCO2', 'AA', 'CT']
    x_labels = [r'$\it{historical}$', r'$\it{H}$-$\it{ext}$', r'$\it{M}$', r'$\it{ML}$', r'$\it{L}$',
               r'$\it{VLLO}$-$\it{ext}$', r'$\it{VLHO}$', r'$\it{H}$-$\it{ext}$-$\it{OS}$',
               r'$\it{M}$-$\it{ext}$', r'$\it{ML}$-$\it{ext}$', r'$\it{L}$-$\it{ext}$',
               r'$\it{VLHO}$-$\it{ext}$', r'$\it{abrupt}$-$\it{2xCO2}$', r'$\it{1pctCO2}$',
               r'$\it{AA}$', r'$\it{CT}$']
    separator_indices = [6, 11, 13]
    group_labels = ['Priority 1', 'Priority 2', 'DECK', 'CS3']
    legend_labels = ['Const.', 'Sine', 'Both']
    seed_agg = "median_iqr"

    n_total = len(scenario_keys)
    labels = x_labels if x_labels and len(x_labels) == n_total else scenario_keys

    # --- Layout: 2 rows, roughly equal height. Row 1 = (a)/(b) side by side;
    # row 2 = (c), full width, vertical bar chart. ---
    layout = [["Top1", "Top2"], ["Bottom", "Bottom"]]
    fig, axd = plt.subplot_mosaic(
        layout, figsize=(14, 6.76875), constrained_layout=True, height_ratios=[0.9, 1]
    )
    axes_co2 = [axd["Top1"], axd["Top2"]]
    ax_bar = axd["Bottom"]

    t_min = np.min(global_mean_temp)
    t_max = np.max(global_mean_temp)

    panel_titles = [
        r"(a) Optimized emissions and resulting $\overline{\Delta T}(t)$ (const. IC)",
        r"(b) Optimized emissions and resulting $\overline{\Delta T}(t)$ (sine IC)",
    ]
    # (a)/(b) now share both y-axes (same emissions scale, same DeltaT scale) -
    # only (a)'s left axis (Emissions) and (b)'s right axis (DeltaT) carry
    # labels/tick labels; the redundant inner axes are de-cluttered.
    ax_temp_a = None
    for i, co2 in enumerate(co2_data):
        if i == 1:
            axes_co2[i].sharey(axes_co2[0])
        axes_co2[i].plot(np.arange(1750, 2501), co2, lw=1.5, c=cm.actonS(2), label='Emissions')
        axes_co2[i].grid(axis='y', linestyle='--', alpha=0.3, zorder=0, c=cm.actonS(2))
        axes_co2[i].grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
        axes_co2[i].tick_params(axis='y', labelcolor=cm.actonS(2))
        axes_co2[i].set_xlim([1750, 2500])

        ax_temp = axes_co2[i].twinx()
        if i == 0:
            ax_temp_a = ax_temp
        else:
            ax_temp.sharey(ax_temp_a)
        ax_temp.plot(np.arange(1751, 2501), global_mean_temp[i], lw=1.5, ls='--', c=cm.actonS(4),
                    label='Global mean temperature', zorder=0)
        ax_temp.grid(linestyle='--', alpha=0.3, zorder=0, c=cm.actonS(4))
        ax_temp.set_ylim(t_min - 0.75, t_max + 0.75)
        ax_temp.tick_params(axis='y', labelcolor=cm.actonS(4))

        if i == 0:
            axes_co2[i].set_ylabel(r'Emissions [GtCO$_2$/yr]', fontsize=12, c=cm.actonS(2))
            plt.setp(ax_temp.get_yticklabels(), visible=False)  # (a)'s right axis: no DeltaT labels
        else:
            ax_temp.set_ylabel(r"$\overline{\Delta T}(t)$ [$^\circ$C]", fontsize=12, rotation=270,
                               labelpad=13, c=cm.actonS(4))
            plt.setp(axes_co2[i].get_yticklabels(), visible=False)  # (b)'s left axis: no Emissions labels

        axes_co2[i].text(
            0.02, 0.94, panel_titles[i], transform=axes_co2[i].transAxes,
            ha="left", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
        )
        if i == 0:
            lines_1, labels_1 = axes_co2[i].get_legend_handles_labels()
            lines_2, labels_2 = ax_temp.get_legend_handles_labels()
            ax_temp.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=True, loc='lower left',
                          fancybox=True, framealpha=0.8, facecolor='white', edgecolor='#cccccc', fontsize=9)

    # --- Bottom bar chart, rotated 90 degrees: vertical bars, scenarios along
    # x, % change along y - increased skill above the axis, decreased below. ---
    n_opts = len(optimized_results_list)
    total_group_width = 0.8
    bar_width = total_group_width / n_opts
    colors = [cm.osloS(i + 2) for i in range(n_opts)]
    bar_ylim = (-70, 90)  # same range as before, now the y-axis limit

    x_positions = np.arange(n_total)

    def _pct_change_row(base_results, opt_results):
        row = []
        for scen in scenario_keys:
            base_val = get_global_value(base_results, scen)
            opt_val = get_global_value(opt_results, scen)
            if np.isnan(base_val) or np.isnan(opt_val) or base_val == 0:
                row.append(0.0)
            else:
                row.append(((base_val - opt_val) / base_val) * 100)
        return row

    any_seed_err = False
    for opt_idx in range(n_opts):
        seed_opt = seed_optimized_results_list[opt_idx]
        per_seed = np.array([_pct_change_row(b, o) for b, o in zip(seed_baseline_results, seed_opt)])
        any_seed_err = True
        if seed_agg == "median_iqr":
            diff_values = np.median(per_seed, axis=0)
            q1 = np.percentile(per_seed, 25, axis=0)
            q3 = np.percentile(per_seed, 75, axis=0)
            diff_err = np.vstack([diff_values - q1, q3 - diff_values])
        else:
            diff_values = per_seed.mean(axis=0)
            diff_err = per_seed.std(axis=0)

        offset = (opt_idx - n_opts / 2) * bar_width + (bar_width / 2)
        bars = ax_bar.bar(x_positions + offset, diff_values, yerr=diff_err,
                          error_kw=dict(capsize=2, elinewidth=1),
                          label=legend_labels[opt_idx], width=bar_width,
                          color=colors[opt_idx], edgecolor='black', linewidth=0.7, zorder=3)
        for bar, val in zip(bars, diff_values):
            if val < bar_ylim[0] or val > bar_ylim[1]:
                bar.set_hatch('//')

    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(labels, rotation=90, ha='center', va='top', fontsize=10)
    ax_bar.tick_params(axis='x', length=0)
    ax_bar.set_xlim(-0.5, n_total - 0.5)

    ax_bar.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['bottom'].set_visible(False)
    ax_bar.axhline(0, color='#4a5568', linewidth=1.2, zorder=0)
    ax_bar.set_ylim(bar_ylim)

    if separator_indices:
        for idx in separator_indices:
            ax_bar.axvline(idx + 0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.6, zorder=0)

    if group_labels and separator_indices:
        boundaries = [-0.5] + [idx + 0.5 for idx in separator_indices] + [n_total - 0.5]
        for i, label_text in enumerate(group_labels):
            center_x = (boundaries[i] + boundaries[i + 1]) / 2
            # Rotated: group labels move below the (now rotated, tall)
            # scenario-name tick labels, as a second label row - avoids any
            # risk of colliding with row 1 above, and reads naturally as a
            # category-group row under a vertical grouped bar chart.
            ax_bar.text(center_x, -0.40, label_text, transform=ax_bar.get_xaxis_transform(),
                        rotation=0, ha='center', va='top', fontsize=12, fontweight='bold',
                        color='#333333')

    # Panel label, top-left as in the original (unrotated) figure.
    ax_bar.text(0.015, 0.98, r"(c) MESM emulator performance summary", transform=ax_bar.transAxes,
                ha="left", va="top", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

    legend_handles = [
        mpatches.Patch(facecolor=colors[i], edgecolor='black', linewidth=0.7, label=legend_labels[i])
        for i in range(n_opts)
    ]
    if seed_agg == "median_iqr" and any_seed_err:
        legend_handles.append(
            Line2D([0, 1], [0, 0], color='black', linewidth=1, marker='|', markersize=10,
                   markeredgewidth=1, label='Interquartile range')
        )
    # Nudged slightly up and right of the default upper-right corner, just
    # enough to clear the Priority2/DECK separator's dashed line.
    leg = ax_bar.legend(handles=legend_handles, title='Emulator IC', loc='upper right',
                        bbox_to_anchor=(1.012, 1.035), numpoints=2, title_fontsize=11,
                        frameon=True, fancybox=True, framealpha=0.8, facecolor='white',
                        edgecolor='#cccccc', fontsize=10)

    ax_bar.set_xlabel('Scenario', fontsize=13, labelpad=32)  # closer to the group-label row
    y_label = (r'Median performance change' + '\n' + r'from baseline emulator [\%]'
              if seed_agg == "median_iqr" and any_seed_err
              else r'Performance change' + '\n' + r'from baseline emulator [\%]')
    ax_bar.set_ylabel(y_label, fontsize=12)

    figname = "fig07_emic_summary_revision_first50_altlayout_PREVIEW"
    fig.savefig(FIGURES_DIR / f"{figname}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{figname}.png", bbox_inches="tight", dpi=200)
    print(f"saved Figures/{figname}.{{pdf,png}} (preview only - not wired into any pipeline)")


if __name__ == "__main__":
    main()
