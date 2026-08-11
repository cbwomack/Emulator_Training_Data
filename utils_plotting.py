# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5 and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

# Imports

import pickle

import numpy as np

## JAX
import jax
import jax.numpy as jnp

## Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from cmcrameri import cm
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from collections import defaultdict
from sklearn.metrics import r2_score

## Local
from paths import FIGURES_DIR
import run_fair
import utils_FaIR_JAX
import utils_inverse

## Setup plots
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "sans-serif",
  "font.sans-serif": ["Helvetica Light"],
})

# ==================================================================
# Part 5: paper & SI figures
# ==================================================================

def plot_init(res: dict, save: bool = False) -> None:
  """Plot the initial (step-0) CO2 trajectory from an optimize_emissions_inverse result."""
  fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
  ax.plot(res['U_traj'][0]['CO2'], c=cm.batlowWS(1), lw=2)
  ax.set_ylabel(r'Emissions [GtCO$_2$/yr]')
  ax.set_xlabel('Year')
  ax.set_xlim([0, len(res['U_traj'][0]['CO2'])])
  if save:
    plt.savefig(FIGURES_DIR / "init_emis.pdf", transparent=True)

def plot_tier1(years: list, tier1: list, group: list[str], save: bool = False) -> None:
  """Plot one CO2 emissions line per tier-1 scenario in `tier1` (labeled by `group`)."""
  fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
  for i, scen in enumerate(tier1):
    ax.plot(years[i], tier1[i], c=cm.batlowWS(i + 1), lw=2, label=group[i])

  ax.set_ylabel(r'Emissions [GtCO$_2$/yr]')
  ax.set_xlabel('Year')
  ax.set_xlim([1750, 2500])
  ax.legend(loc='upper left', fontsize=14)

  if save:
    plt.savefig(FIGURES_DIR / "tier1.pdf", transparent=True)

def plot_updates(res: dict, save: bool = False) -> None:
  """Plot the CO2 trajectory every 50 outer-loop steps of an optimize_emissions_inverse result, fading with step."""
  fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
  for i, traj in enumerate(res['U_traj']):
    if i % 50 == 0:
      alpha = 0.2 + 0.6 * (i / max(1, len(res['U_traj']) - 1))
      if i in [0, 500]:
        ax.plot(traj['CO2'], alpha=alpha, c=cm.batlowWS(1), lw=2, label=f'Iteration {i}')
      else:
        ax.plot(traj['CO2'], alpha=alpha, c=cm.batlowWS(1), lw=2)

  ax.set_ylabel(r'Emissions [GtCO$_2$/yr]')
  ax.set_xlabel('Year')
  ax.set_xlim([0, len(res['U_traj'][0]['CO2'])])
  ax.legend()
  if save:
    plt.savefig(FIGURES_DIR / "emis_updates.pdf", transparent=True)

def plot_rmse_comparison_single(
    results_list: list[dict],       # List of dictionaries
    baseline_error_list: list[float],     # Single float value
    agents: list[str],
    save: bool = False
) -> None:
    """5-panel NRMSE-vs-update-step comparison, one panel per single-forcing agent experiment."""
    layout = [
        ["Left", "Left", "Right1", "Right2"],
        ["Left", "Left", "Right3", "Right4"]
    ]

    fig, axd = plt.subplot_mosaic(
        layout,
        figsize=(10, 4.5),
        sharey=True,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.001, "hspace": 0.001}
    )
    cmap = cm.batlowS

    axes = [axd["Left"], axd["Right1"], axd["Right2"], axd["Right3"], axd["Right4"]]

    for i, ax in enumerate(axes):
        result = results_list[i]
        baseline_error = baseline_error_list[i]
        errors = jnp.asarray(result["errors"])
        x_err = jnp.arange(errors.shape[0])

        # --- Plotting ---
        ax.loglog(x_err, errors, label="Optimized emulator", lw=2, color=cmap(0))
        ax.axhline(float(baseline_error), ls="--", c=cm.lipariS(5), lw=1.5, label="Baseline emulator\nerror lower bound")

        ax.margins(x=0, y=0)
        #ax.xaxis.set_major_locator(plt.MaxNLocator(, prune='lower'))

        ax.grid(True, alpha=0.3, which="both", ls="-")

        a = agents[i]
        if i > 0:
          x_off, y_off = 0.075, 0.925
        else:
          x_off, y_off = 0.05, 0.95
        ax.text(
          x_off, y_off, fr"{a}-only", transform=ax.transAxes,
          ha="left", va="top", fontsize=16, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.5)
        )

        ax.tick_params(axis='both', which='major', labelsize=12)

        if i in [2, 4]:
            ax.yaxis.tick_right() # Moves ticks to right side
            ax.tick_params(axis='y', labelright=True, labelsize=12)

        # Only add the Y-label to the first plot to reduce clutter
        if i == 0:
            ax.set_ylabel("Emulator error (NRMSE)", fontsize=18)
            ax.legend(loc="best", fontsize=14)

        ax.set_xlim(left=1.1)
        ax.set_ylim([0.01, 1.5])

    fig.supxlabel('Update iteration no.', fontsize=18)

    if save:
      plt.savefig(FIGURES_DIR / 'fig03_single_forcing.pdf')

    return

def plot_rmse_comparison_multi(results: dict, baseline_error: float, save: bool = False) -> None:
    """3-panel figure: NRMSE-vs-step (left) plus optimal WMGHG and aerosol emissions trajectories (right, via _plot_agents)."""
    layout = [
        ["Left", "Left", "Right1", "Right1", "Right1"],
        ["Left", "Left", "Right2", "Right2", "Right2"]
    ]

    fig, axd = plt.subplot_mosaic(
        layout,
        figsize=(15, 5),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.001, "hspace": 0.001}
    )
    cmap = cm.batlowWS

    # --- Plot 1: RMSE (Left) ---
    ax = axd['Left']
    errors = jnp.asarray(results["errors"])
    x_err = jnp.arange(errors.shape[0])

    ax.loglog(x_err, errors, label="Optimized emulator", lw=2, color=cmap(1))
    ax.axhline(float(baseline_error), ls="--", c=cm.lipariS(5), lw=1.5, label="Baseline emulator\nerror lower bound")

    # Styling
    ax.margins(x=0, y=0.2)
    ax.grid(True, alpha=0.3, which="both", ls="-")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("Emulator error (NRMSE)", fontsize=18)
    ax.set_xlabel('Update iteration no.', fontsize=18)
    ax.set_xlim(left=1.1)
    ax.legend(loc="upper right", fontsize=14)

    # Text Box
    _add_textbox(ax, "(a) All agents", 0.025, 0.97)

    # --- Prepare Data for Right Plots ---
    # Extract data once
    traj_data = results['U_traj'][-1]
    years = np.arange(1750, 2501)

    wm_config = {
        'ax': axd["Right1"],
        'agents': ['CO2', 'CH4', 'N2O'],
        'labels': ['CO$_2$ [Gt]', 'CH$_4$ [Mt]', 'N$_2$O [Mt]'],
        'data': [traj_data[a] for a in ['CO2', 'CH4', 'N2O']],
        'title': "(b) WM agents"
    }

    aer_config = {
        'ax': axd["Right2"],
        'agents': ['Sulfur', 'BC'],
        'labels': ['Sulfur [Mt]', 'BC [Mt]'],
        'data': [traj_data[a] for a in ['Sulfur','BC']],
        'title': "(c) AER agents"
    }

    # --- Plot 2 & 3: Trajectories (Right) ---
    # Use helper function to remove redundancy
    _plot_agents(years, wm_config, cmap)
    _plot_agents(years, aer_config, cmap)

    # Final adjustments
    axd["Right1"].xaxis.set_major_locator(plt.MaxNLocator(5))
    axd["Right1"].tick_params(labelbottom=False)
    axd["Right2"].set_xlabel('Year', fontsize=18)
    axd["Right2"].xaxis.set_major_locator(plt.MaxNLocator(5))

    if save:
        plt.savefig(FIGURES_DIR / 'fig05_multi_forcing.pdf')

    return

# --- Helper Functions ---

def _plot_agents(x_data: np.ndarray, config: dict, cmap) -> None:
    """Handles multi-axis plotting, coloring, and unified legends."""
    base_ax = config['ax']
    lines = []

    # Keep track of the last axis used to place text/legend on top
    last_ax = base_ax
    ls = ['-','--','-.']

    for i, (data, label) in enumerate(zip(config['data'], config['labels'])):
        if len(config['labels']) == 3:
            if i == 0:
              color = cmap(i + 1)
            else:
              color = cm.osloS(i + 1)

        else:
          color = cm.actonS(i + 3)

        if i == 0:
            current_ax = base_ax
            spine_key = 'left'
        else:
            current_ax = base_ax.twinx()
            spine_key = 'right'
            # Offset the third spine so it doesn't overlap the second
            if i > 1:
                current_ax.spines["right"].set_position(("axes", 1.0 + (i-1)*0.1))

        ln = current_ax.plot(x_data, data, color=color, label=label, ls=ls[i])
        lines += ln

        # Styling
        #current_ax.set_ylabel(label, color=color)
        current_ax.tick_params(axis='y', colors=color)
        current_ax.spines[spine_key].set_color(color)
        current_ax.grid(True, alpha=0.3, which="both", ls="-", color=color)

        last_ax = current_ax

    # --- FIX 2 & 3: Match Style and Fix Layering ---
    # We add the legend to 'last_ax' (the top layer) so it sits above all lines.
    # We use framealpha, edgecolor, and facecolor to match your text boxes.
    leg = last_ax.legend(
        lines, [l.get_label() for l in lines],
        loc='lower left',
        fancybox=True,      # Rounded corners
        facecolor="white",  # Match text box
        edgecolor="gray",   # Match text box
        framealpha=0.9      # Match text box
    )
    # Force the legend zorder high just to be safe
    leg.set_zorder(105)

    # Add text box to the LAST axis so it sits on top of all lines
    _add_textbox(last_ax, config['title'], 0.02, 0.93)
    base_ax.margins(x=0, y=0.1)

def _add_textbox(ax, text: str, x: float, y: float) -> None:
    """Bold boxed annotation at axes-fraction coords (x, y)."""
    ax.text(
        x, y, text, transform=ax.transAxes,
        ha="left", va="top", fontsize=16, fontweight="bold",
        zorder=100, # Explicitly high zorder
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
    )

def plot_emissions_grid(
    opt_emissions: list, target_emissions: list, years: list, targets: list[str], groups: list, save: bool = False
) -> None:
    """
    Plots a 2x4 grid.
    Top Row: Optimal Emissions (nested list structure)
    Bottom Row: Target Emissions (nested list structure)
    """

    # Setup 2x4 grid
    # sharey='row' ensures all 'Optimal' plots share one scale,
    # and all 'Target' plots share another.
    fig, axes = plt.subplots(
        nrows=2, ncols=3,
        figsize=(14, 6),
        sharex='col',
        sharey=True,
        constrained_layout=True
    )
    cmap = cm.batlowWS

    # Loop over the 4 columns (experiments)
    for col_idx in range(3):
        ax_top = axes[1, col_idx]
        ax_bot = axes[0, col_idx]

        # ---------------------------------------------------
        # 1. Top Row: Optimal Emissions
        # ---------------------------------------------------
        if col_idx < len(opt_emissions):
            # entries_list is the list of time series for this specific experiment
            entries_list = opt_emissions[col_idx]

            # Track max length to set tight x-limits later
            min_t, max_t = np.inf, 0

            # Plot every time series in this experiment's list
            for i, series in enumerate(entries_list):
                if i % 100 == 0:
                  y_data = np.asarray(series).reshape(-1)
                  alpha = 0.2 + 0.6 * (i / max(1, len(entries_list) - 1))
                  if col_idx < 1:
                      x_data = np.arange(0, len(y_data)) + 2024
                  else:
                      x_data = np.arange(0, len(y_data)) + 1750
                  if i in [0, 1000]:
                    ax_top.plot(x_data, y_data, alpha=alpha, lw=1.5, c = cmap(1), label=f'Iteration {i}')
                  else:
                    ax_top.plot(x_data, y_data, alpha=alpha, lw=1.5, c = cmap(1))
                  max_t = max(max_t, x_data[-1])
                  min_t = min(min_t, x_data[0])

            # Apply Stylistic Choices
            ax_top.set_xlim(min_t, max_t - 1)
            ax_top.margins(y=0)

            ax_top.grid(True, alpha=0.3)
        else:
            ax_top.axis('off')

        if col_idx == 0:
          ax_top.legend(loc='upper left')

        # ---------------------------------------------------
        # 2. Bottom Row: Target Emissions
        # ---------------------------------------------------
        if col_idx < len(target_emissions):
            entries_list = target_emissions[col_idx]
            year_list = years[col_idx]
            min_t, max_t = np.inf, 0

            for i, series in enumerate(entries_list):
                x_data = year_list[i]
                y_data = np.asarray(series).reshape(-1)
                # Plotting target with a different style or color if desired (e.g., dashed or C1)
                if col_idx < 1:
                    ax_bot.plot(x_data, y_data, alpha=0.8, lw=1.5, c = cmap(i + 2))
                else:
                  ax_bot.plot(x_data, y_data, alpha=0.8, lw=1.5, c = cmap(i + 2), label=f'{groups[col_idx][i]}')
                max_t = max(max_t, x_data[-1])
                min_t = min(min_t, x_data[0])

            # Apply Stylistic Choices
            ax_bot.set_xlim(min_t, max_t - 1)
            ax_bot.margins(y=0)

            ax_bot.grid(True, alpha=0.3)
        else:
            ax_bot.axis('off')

        if col_idx > 0:
          ax_bot.legend(fontsize=12, loc='upper right')

        ax_bot.text(
              0.035, 0.94, f"Target - {targets[col_idx]}", transform=ax_bot.transAxes,
              ha="left", va="top", fontsize=16, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.5)
            )

    axes[0, 0].set_ylim([-25, 135])
    fig.supylabel(r"Emissions [GtCO$_2$/yr]")
    fig.supxlabel('Year')

    if save:
      plt.savefig(FIGURES_DIR / 'co2_multi_target.pdf')

    return

def plot_co2_sulfur(co2: list, sulfur: list, save: bool = False) -> None:
    """4-row CO2 (left axis) vs. Sulfur (right axis) time series, one row per experiment (Tier 1/DAMIP/GeoMIP/All)."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 5/3*4), sharex=True, sharey=True, constrained_layout=True)

    cmap = cm.batlowWS
    color1 = cmap(1)
    color2 = cmap(2)
    years = np.arange(1750, 2501)

    # Variable to store the first secondary axis (the "anchor")
    first_twin = None
    x_off, y_off = 0.015, 0.24
    experiments = ['Tier 1', 'DAMIP', 'GeoMIP', 'All']
    ylim_co2, ylim_sulfur = -50, -50

    for i, ax in enumerate(axes):
        # --- Left Axis (Primary) ---
        ax.plot(years, co2[i], color=color1)
        ax.spines['left'].set_color(color1)
        ax.tick_params(axis='y', colors=color1)
        ax.grid(True, alpha=0.3, which="both", ls="-", color=color1)

        #ax.set_ylim(ylim_co2)

        # --- Right Axis (Secondary) ---
        ax_temp = ax.twinx()

        # SHAREY LOGIC: Link this new twin axis to the first one created
        if first_twin is None:
            first_twin = ax_temp
        else:
            ax_temp.sharey(first_twin)

        ax_temp.plot(years, sulfur[i], color=color2)
        ax_temp.spines['right'].set_color(color2)
        ax_temp.spines['left'].set_visible(False)
        ax_temp.tick_params(axis='y', colors=color2)
        ax_temp.grid(True, alpha=0.3, which="both", ls="-", color=color2)

        #ax_temp.set_ylim(ylim_sulfur)

        # Apply margins to both axes to be safe
        ax.margins(x=0, y=0.1)
        ax_temp.margins(x=0, y=0.1)

        highlight_opposite_slopes(ax, years, co2[i], sulfur[i], min_length=20)

        ax_temp.text(
          x_off, y_off, f"Target: {experiments[i]}", transform=ax.transAxes,
          ha="left", va="top", fontsize=16, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9,
          zorder=100)
        )

    ax.set_xlabel('Year', fontsize=14)
    fig.supylabel(r'CO$_2$ Emissions [Gt/yr]', color=color1, fontsize=18)

    fig.text(
        1.0, 0.5, 'Sulfur Emissions [Mt/yr]',
        rotation=-90,
        va='center',
        ha='left',
        color=color2,
        fontsize=18
    )

    if save:
      plt.savefig(FIGURES_DIR / 'co2_sulfur_compare.pdf', bbox_inches='tight')

    return

from scipy.ndimage import gaussian_filter1d
def highlight_opposite_slopes(ax, x: np.ndarray, y1: np.ndarray, y2: np.ndarray, min_length: int = 10, sigma: float = 1) -> None:
    """
    Highlights regions with opposite slopes, using smoothing to handle noise.
    """
    # 1. Apply Gaussian Smoothing to ignore high-frequency noise
    # sigma=0 means no smoothing. sigma=2 is usually a good default for 'visual' trends.
    if sigma > 0:
        y1_smooth = gaussian_filter1d(y1, sigma=sigma)
        y2_smooth = gaussian_filter1d(y2, sigma=sigma)
    else:
        y1_smooth = y1
        y2_smooth = y2

    # 2. Calculate slopes on the smoothed data
    d1 = np.diff(y1_smooth)
    d2 = np.diff(y2_smooth)

    # 3. Check for opposite signs
    mask = (d1 * d2) < 0

    padded = np.concatenate(([False], mask, [False]))
    change_indices = np.flatnonzero(padded[:-1] != padded[1:])

    starts = change_indices[::2]
    stops = change_indices[1::2]

    for start, stop in zip(starts, stops):
        if (stop - start) >= min_length:
            ax.axvspan(x[start], x[stop], color='red', alpha=0.15, lw=0, zorder=0)


def plot_stacked_results_ppt(
    # --- Inputs for Top Row (Ground Truth) ---
    target_years: list,            # List of arrays or single array corresponding to target_emissions

    # --- Inputs for Middle Row (Optimized) ---
    opt_emissions_history: list,   # List of arrays: history of optimized emission curves

    # --- Inputs for Bottom Row (Preds vs Truth) ---
    results: dict,                 # Dictionary containing 'preds_traj'
    pred_scenario: str | None = None,      # Specific scenario name to plot (optional)

    # --- Styling Options ---
    opt_start_year: int = 2024,     # Start year for optimized emissions x-axis
    max_lines: int = 11,            # Max lines to plot for fading history
    save: bool = False,
    save_path: str | None = None
) -> None:
    """
    Vertical Stack Plot (3 Rows):
    1. Ground Truth Emissions (Top)
    2. Optimized Emissions History (Middle)
       -> Top and Middle share Y-axis scale and Y-label.
    3. Predictions vs Truth Temperature (Bottom)
       -> Own Y-axis, but shares X-axis with above.
    """

    # 1. Setup Figure and Axes
    # sharex=True ensures all rows share the time axis.
    fig, axes = plt.subplots(nrows=2, figsize=(16, 7), sharey='row',sharex=True, constrained_layout=True)
    ax_opt = axes[0]
    ax_pred = axes[1]

    cmap = cm.batlowWS

    # =========================================================================
    # ROW 2: Optimized Emissions
    # =========================================================================
    entries_list = opt_emissions_history
    n_total = len(entries_list)

    # Logic to select a subset of lines if history is very long
    if n_total > 500:
        indices_to_plot = list(range(0, n_total, 100))
        if (n_total - 1) not in indices_to_plot:
            indices_to_plot.append(n_total - 1)
    else:
        indices_to_plot = range(n_total)

    for i in indices_to_plot:
        series = entries_list[i]
        y_data = np.asarray(series).reshape(-1)
        x_data = np.arange(0, len(y_data)) + opt_start_year

        # Fading alpha logic
        alpha = 0.2 + 0.6 * (i / max(1, n_total - 1))

        if i == 0 or i == 100 or i == n_total - 1:
            c = cmap(1)
            ls = '-'
            if i == 0:
              alpha = 1
              c = cm.naviaS(4)
              ls = '-.'
            ax_opt.plot(x_data, y_data, alpha=alpha, lw=1.5, c=c, label=f'Iteration {i}', ls=ls)
        else:
            ax_opt.plot(x_data, y_data, alpha=alpha, lw=1.5, c=cmap(1))

        #max_t = max(max_t, x_data[-1])

    ax_opt.set_ylim([-30, 95])
    ax_opt.grid(True, alpha=0.3)
    ax_opt.legend(loc='lower left', fontsize=12)
    ax_opt.tick_params(labelbottom=False)

    ax_opt.text(
        0.015, 0.95, "(a) Optimized training emissions", transform=ax_opt.transAxes,
        ha="left", va="top", fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
    )

    # =========================================================================
    # ROW 3: Predictions vs Truth (Temperature)
    # =========================================================================
    preds_traj = results.get("preds_traj", [])

    # Determine Scenario Name
    if pred_scenario is None and len(preds_traj) > 0:
        pred_scenario = preds_traj[0][0][0]
    elif pred_scenario is None:
        pred_scenario = "Unknown"

    def _find_scen_idx(step_list, name):
        for j, (sc, _, _) in enumerate(step_list):
            if sc == name: return j
        return None

    N_all = len(preds_traj)
    if N_all > 0:
        # Fading history logic for predictions
        if N_all <= max_lines:
            sel_pred = np.arange(N_all, dtype=int)
        else:
            sel_pred = np.unique(np.linspace(0, N_all - 1, num=max_lines, dtype=int))

        last_ytrue = None

        for k, i in enumerate(sel_pred):
            step_list = preds_traj[i]
            j = _find_scen_idx(step_list, pred_scenario)
            if j is None: continue

            _, yhat, ytrue = step_list[j]
            yhat, ytrue = jnp.asarray(yhat), jnp.asarray(ytrue)

            alpha = 0.3 + 0.7 * (k / max(1, len(sel_pred) - 1))

            # Plot Prediction
            c = cmap(1)
            ls = '-'
            if i == 0 or i == 2 or i == 20:
              if i == 0:
                alpha = 1
                c = cm.naviaS(4)
                ls='-.'
                lab = 0
              elif i == 2:
                lab = 100
              else:
                lab = 1000
              ax_pred.plot(target_years, yhat, alpha=alpha, color=c, ls=ls, label=f"Emulator iteration {lab}")
            else:
              ax_pred.plot(target_years, yhat, alpha=alpha, color=c, ls=ls)
            last_ytrue = ytrue

        # Plot Truth (Red dashed)
        if last_ytrue is not None:
            ax_pred.plot(target_years, last_ytrue, ls="--", c="C3", lw=2.0, label="SCM-projected")
            # Ensure the shared X-axis covers the full range
            ax_pred.set_xlim(2024, 2500)

    ax_pred.set_xlabel("Year")
    ax_pred.grid(True, alpha=0.3)
    handles, labels = ax_pred.get_legend_handles_labels()
    new_handles = [handles[-1]] + handles[:-1]
    new_labels = [labels[-1]] + labels[:-1]
    ax_pred.legend(new_handles, new_labels, loc="lower right", fontsize=12)

    ax_pred.text(
        0.015, 0.95, f"(b) SCM-projected vs. emulated temperature", transform=ax_pred.transAxes,
        ha="left", va="top", fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
    )

    ax_opt.set_ylabel(r"Emissions [GtCO$_2$/yr]",fontsize=16)
    ax_pred.set_ylabel(r"$\overline{\Delta T}(t)$ [$^\circ$C]",fontsize=16)


    if save:
        plt.savefig(save_path, bbox_inches='tight')

    return

def plot_stacked_results(
    # --- Inputs for Top Row (Ground Truth) ---
    target_emissions: list,        # List of arrays (or nested list)
    target_years: list,            # List of arrays or single array corresponding to target_emissions

    # --- Inputs for Middle Row (Optimized) ---
    opt_emissions_history: list,   # List of arrays: history of optimized emission curves
    opt_temp_history: list,

    # --- Inputs for Bottom Row (Preds vs Truth) ---
    results: dict,                 # Dictionary containing 'preds_traj'
    pred_scenario: str | None = None,      # Specific scenario name to plot (optional)

    # --- Styling Options ---
    opt_start_year: int = 2024,     # Start year for optimized emissions x-axis
    max_lines: int = 11,            # Max lines to plot for fading history
    save: bool = False,
    save_path: str | None = None
) -> None:
    """
    Vertical Stack Plot (3 Rows):
    1. Ground Truth Emissions (Top)
    2. Optimized Emissions History (Middle)
       -> Top and Middle share Y-axis scale and Y-label.
    3. Predictions vs Truth Temperature (Bottom)
       -> Own Y-axis, but shares X-axis with above.
    """

    # 1. Setup Figure and Axes
    # sharex=True ensures all rows share the time axis.
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 6), sharey='row',sharex=True, constrained_layout=True)

    ax_truth = axes[0,0]
    ax_opt = axes[0,1]
    ax_pred = axes[1,0]
    ax_opt_temp = axes[1,1]

    # 2. Link Y-axis for Top and Middle only
    # This forces ax_opt to use the same scale as ax_truth
    ax_opt.sharey(ax_truth)

    cmap = cm.batlowWS

    # =========================================================================
    # ROW 1: Ground Truth Emissions
    # =========================================================================
    min_t, max_t = np.inf, 0

    for i, series in enumerate(target_emissions):
        # Determine X-axis data
        x_data = target_years

        y_data = np.asarray(series)

        # Plot
        ax_truth.plot(x_data, y_data, lw=1.5, c=cm.actonS(2), label=f"Group {i}")

        # Update time bounds
        max_t = max(max_t, x_data[-1])
        min_t = min(min_t, x_data[0])

    ax_truth.grid(True, alpha=0.3)
    # Hide x-labels for top row (redundant due to sharex, but good practice to ensure)
    ax_truth.tick_params(labelbottom=False)

    ax_truth.text(
        0.015, 0.95, r"(a) ScenarioMIP-CMIP7 emissions ($\it{H}$-$\it{ext}$)", transform=ax_truth.transAxes,
        ha="left", va="top", fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
    )

    # =========================================================================
    # ROW 2: Optimized Emissions
    # =========================================================================
    entries_list = opt_emissions_history
    n_total = len(entries_list)

    # Logic to select a subset of lines if history is very long
    if n_total > 500:
        indices_to_plot = list(range(0, n_total, 100))
        if (n_total - 1) not in indices_to_plot:
            indices_to_plot.append(n_total - 1)
    else:
        indices_to_plot = range(n_total)

    for i in indices_to_plot:
        series = entries_list[i]
        y_data = np.asarray(series).reshape(-1)
        x_data = np.arange(0, len(y_data)) + opt_start_year

        # Fading alpha logic
        alpha = 0.2 + 0.6 * (i / max(1, n_total - 1))

        if i == 0 or i == 100 or i == n_total - 1:
            c = cmap(1)
            ls = '-'
            if i == 0:
              alpha = 1
              c = cm.naviaS(4)
              ls = '-.'
            ax_opt.plot(x_data, y_data, alpha=alpha, lw=1.5, c=c, label=f'Iteration {i}', ls=ls)
        else:
            ax_opt.plot(x_data, y_data, alpha=alpha, lw=1.5, c=cmap(1))

        max_t = max(max_t, x_data[-1])

    ax_opt.grid(True, alpha=0.3)
    ax_opt.legend(loc='lower left', fontsize=12)
    ax_opt.tick_params(labelbottom=False)

    ax_opt.text(
        0.015, 0.95, "(b) Optimized training emissions", transform=ax_opt.transAxes,
        ha="left", va="top", fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
    )

    # =========================================================================
    # ROW 3: Predictions vs Truth (Temperature)
    # =========================================================================
    preds_traj = results.get("preds_traj", [])

    # Determine Scenario Name
    if pred_scenario is None and len(preds_traj) > 0:
        pred_scenario = preds_traj[0][0][0]
    elif pred_scenario is None:
        pred_scenario = "Unknown"

    def _find_scen_idx(step_list, name):
        for j, (sc, _, _) in enumerate(step_list):
            if sc == name: return j
        return None

    N_all = len(preds_traj)
    if N_all > 0:
        # Fading history logic for predictions
        if N_all <= max_lines:
            sel_pred = np.arange(N_all, dtype=int)
        else:
            sel_pred = np.unique(np.linspace(0, N_all - 1, num=max_lines, dtype=int))

        last_ytrue = None

        for k, i in enumerate(sel_pred):
            step_list = preds_traj[i]
            j = _find_scen_idx(step_list, pred_scenario)
            if j is None: continue

            _, yhat, ytrue = step_list[j]
            yhat, ytrue = jnp.asarray(yhat), jnp.asarray(ytrue)

            alpha = 0.3 + 0.7 * (k / max(1, len(sel_pred) - 1))

            # Plot Prediction
            c = cmap(1)
            ls = '-'
            if i == 0 or i == 2 or i == 20:
              if i == 0:
                alpha = 1
                c = cm.naviaS(4)
                ls='-.'
                lab = 0
              elif i == 2:
                lab = 100
              else:
                lab = 1000
              ax_pred.plot(target_years, yhat, alpha=alpha, color=c, ls=ls, label=f"Emulator iteration {lab}")
            else:
              ax_pred.plot(target_years, yhat, alpha=alpha, color=c, ls=ls)
            last_ytrue = ytrue

        # Plot Truth (Red dashed)
        if last_ytrue is not None:
            ax_pred.plot(target_years, last_ytrue, ls="--", c="C3", lw=2.0, label="SCM-projected")
            # Ensure the shared X-axis covers the full range
            ax_pred.set_xlim(min_t, max(max_t, len(last_ytrue)))

    ax_pred.set_xlabel("Year")
    ax_pred.grid(True, alpha=0.3)
    handles, labels = ax_pred.get_legend_handles_labels()
    new_handles = [handles[-1]] + handles[:-1]
    new_labels = [labels[-1]] + labels[:-1]
    ax_pred.legend(new_handles, new_labels, loc="lower right", fontsize=12)

    ax_pred.text(
        0.015, 0.95, f"(c) SCM-projected vs. emulated temperature", transform=ax_pred.transAxes,
        ha="left", va="top", fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
    )

    # =========================================================================
    # ROW 4: Optimized temperature
    # =========================================================================
    entries_list = opt_temp_history
    n_total = len(entries_list)

    # Logic to select a subset of lines if history is very long
    if n_total > 500:
        indices_to_plot = list(range(0, n_total, 100))
        if (n_total - 1) not in indices_to_plot:
            indices_to_plot.append(n_total - 1)
    else:
        indices_to_plot = range(n_total)

    for i in indices_to_plot:
        series = entries_list[i]
        y_data = np.asarray(series).reshape(-1)
        x_data = np.arange(0, len(y_data)) + opt_start_year

        # Fading alpha logic
        alpha = 0.1 + 0.7 * (i / max(1, n_total - 1))

        if i == 0 or i == 100 or i == n_total - 1:
            c = cmap(1)
            ls = '-'
            if i == 0:
              alpha = 1
              c = cm.naviaS(4)
              ls = '-.'
            ax_opt_temp.plot(x_data, y_data, alpha=alpha, lw=1.5, c=c, ls=ls)
        else:
            ax_opt_temp.plot(x_data, y_data, alpha=alpha, lw=1.5, c=cmap(1))

        max_t = max(max_t, x_data[-1])

    ax_opt_temp.set_xlabel("Year")
    ax_opt_temp.grid(True, alpha=0.3)
    #ax_opt_temp.legend(loc='lower left', fontsize=12)

    ax_opt_temp.text(
        0.015, 0.95, "(d) Temperature from optimized training emissions", transform=ax_opt_temp.transAxes,
        ha="left", va="top", fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.95)
    )

    # =========================================================================
    # Shared Y-Label for Top Two Rows
    # =========================================================================
    # We place text on the figure relative to the axes positions.
    # roughly centered vertically between row 0 and row 1
    # -0.05 is an offset to the left of the axes
    ax_truth.set_ylabel(r"Emissions [GtCO$_2$/yr]",fontsize=16)

    ax_pred.set_ylabel(r"$\overline{\Delta T}(t)$ [$^\circ$C]",fontsize=16)


    if save:
        plt.savefig(save_path, bbox_inches='tight')

    return

def plot_single_heatmap(baseline_results: dict,
                        optimized_results: dict,
                        train_scenarios: list[str],
                        test_scenarios: list[str],
                        training_paths: list[str],
                        weights: list[int],
                        vmax: float,
                        cmap: str = cm.lajolla_r,
                        long_title: str = '',
                        save: bool=False,
                        figname: str='') -> None:
  """
  Draws a comparison heatmap: Baseline vs Optimized results.

  Rows (Y-axis): Test Sets
  Cols (X-axis): Baseline (Col 0) + Optimized Results per Training Path (Cols 1..N)

  Parameters
  ----------
  baseline_results : dict
    Errors indexed as baseline_results[test_set]['mean'].
  optimized_results : dict
    Errors indexed as optimized_results[training_path]['metrics'][test_set]['mean'].
  train_scenarios : list[str]
    List of training path keys (determines columns 1 to N).
  test_scenarios : list[str]
    List of test set keys (determines rows).
  ax : matplotlib.axes.Axes
    Target axis for the heat-map.
  vmax : float
    Colour-scale maximum (min is fixed at 0).
  cmap : str, optional
    Matplotlib/SNS colour-map (default "Reds").
  long_title : str, optional
    Sub-plot title.
  add_cbar : bool, optional
    Add colour-bar on this axis when True.

  Returns
  -------
  None
  """
  # Dimensions: Rows = 1 (Avg.) + N (Test Sets), Columns = 1 (Baseline) + N (Training Scenarios)
  n_test = len(test_scenarios)
  n_rows = 1 + n_test
  n_cols = 1 + len(train_scenarios)

  fig, ax = plt.subplots(figsize=(8 * n_cols / 5, 6 * n_rows / 4), sharex='col', constrained_layout=True)

  # Instantiate data array
  data = np.empty((n_rows, n_cols))

  # --- Fill Column 0: Baseline Results ---
  for i, scen_test in enumerate(test_scenarios):
    try:
      # Access: baseline_results[test_set]['mean']
      value = baseline_results[scen_test]['mean']
    except KeyError:
      value = np.nan
    data[i, 0] = value

  # --- Fill Columns 1 to N: Optimized Results ---
  for j, path in enumerate(training_paths):
    col_idx = j + 1  # Offset by 1 because col 0 is baseline
    for i, scen_test in enumerate(test_scenarios):
      try:
        value = optimized_results[path]['metrics'][scen_test]['mean']
      except KeyError:
        value = np.nan
      data[i, col_idx] = value

  w_arr = np.array(weights)
  for col in range(n_cols):
    # Slice the column data corresponding to the test scenarios (exclude the last empty row)
    col_data = data[:n_test, col]

    # Calculate weighted average
    # Note: If col_data contains NaNs, the result will be NaN.
    # Use np.ma.average if you wish to ignore NaNs, but standard np.average is used here.
    try:
      avg_val = np.average(col_data, weights=w_arr)
    except Exception:
      avg_val = np.nan

    data[n_test, col] = avg_val

  # Plot the heatmap
  sns.heatmap(
    data,
    ax=ax,
    cmap=cmap,
    vmin=0,
    vmax=vmax,
    linewidth=0.5,
    annot=True,
    fmt=".2g",
    cbar=True,
    cbar_kws={"label": r"Mean NRMSE"}
  )

  # Configure labels and title
  ax.set_title(long_title)

  # Generate dynamic labels based on inputs
  # X-labels: "Baseline" followed by the training scenario names
  x_tick_labels = ['Baseline'] + train_scenarios
  # Y-labels: The test scenario names
  y_tick_labels = test_scenarios + ['Avg.']
  ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
  ax.set_yticklabels(y_tick_labels, rotation=0) # Typically horizontal for Y-axis looks better

  fig.supxlabel('Emulator configuration')
  fig.supylabel('Test dataset')

  if save:
    plt.savefig(FIGURES_DIR / f'{figname}.pdf')

  return

def plot_grouped_improvement_bars(baseline_results: dict = None,
                                  optimized_results: dict = None,
                                  train_scenarios: list[str] = None,
                                  test_scenarios: list[str] = None,
                                  x_labels: list[str] = None,
                                  leg_labels: list[str] = None,
                                  weights: list[int] = None,
                                  ax: plt.Axes = None,
                                  long_title: str = '',
                                  show_legend: bool = True,
                                  show_xlabel: bool = True,
                                  save: bool = False,
                                  figname: str = '',
                                  n_plots: int=1,
                                  seed_baseline_results: list[dict] = None,
                                  seed_optimized_results: list[dict] = None) -> None:
    """
    Grouped horizontal bar chart of % NRMSE improvement (optimized vs. baseline), grouped by test scenario.

    `seed_baseline_results`/`seed_optimized_results` (both optional, default None):
    when both are given (equal-length lists, one baseline/optimized result dict
    per seed - each shaped exactly like `baseline_results`/`optimized_results`),
    the bars show the mean % improvement across seeds with error bars (+/- 1 std),
    computing pct-improvement per seed against THAT seed's own baseline (not a
    single shared one). `baseline_results`/`optimized_results` are ignored in this
    mode. Leaving both at None (the default) reproduces the exact prior
    single-point-estimate behavior - fully backward-compatible.
    """

    # 0. Handle Axis Creation (Backward Compatibility)
    # ----------------------------------------------
    is_standalone = False
    if ax is None:
        is_standalone = True
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    # 1. Organize Data
    # ----------------
    n_test = len(test_scenarios)
    n_opts = len(train_scenarios)
    w_arr = np.array(weights)

    seed_mode = seed_baseline_results is not None and seed_optimized_results is not None

    def _pct_improvement_for(baseline_res, optimized_res):
        base_errors = np.zeros(n_test)
        opt_errors = np.zeros((n_test, n_opts))

        for i, test_key in enumerate(test_scenarios):
            try:
                base_errors[i] = baseline_res[test_key]['mean']
            except KeyError:
                base_errors[i] = np.nan

        for j, train_key in enumerate(train_scenarios):
            for i, test_key in enumerate(test_scenarios):
                try:
                    opt_errors[i, j] = optimized_res[train_key][test_key]['mean']
                except KeyError:
                    opt_errors[i, j] = np.nan

        avg_base_error = np.average(base_errors, weights=w_arr)
        avg_opt_errors = np.average(opt_errors, axis=0, weights=w_arr)

        pct_improvement = (base_errors[:, None] - opt_errors) / base_errors[:, None]
        avg_improvement = (avg_base_error - avg_opt_errors) / avg_base_error

        return np.vstack([pct_improvement, avg_improvement]) * 100

    pct_std = None
    if seed_mode:
        n_seeds = len(seed_baseline_results)
        stacked = np.stack([
            _pct_improvement_for(seed_baseline_results[s], seed_optimized_results[s])
            for s in range(n_seeds)
        ], axis=0)
        plot_data = stacked.mean(axis=0)
        pct_std = stacked.std(axis=0)
    else:
        plot_data = _pct_improvement_for(baseline_results, optimized_results)

    row_labels = leg_labels + ['Avg.']
    n_rows = len(row_labels)

    # 3. Plotting Setup
    # -----------------
    x_positions = np.arange(n_opts)
    total_group_width = 0.7
    bar_width = total_group_width / n_rows
    limit = -60

    # 4. Draw Grouped Bars
    # --------------------
    hatch_pattern = '//'
    for i in range(n_rows):
        row_values = plot_data[i]
        label = row_labels[i]

        offset = (i - n_rows / 2) * bar_width + (bar_width / 2)
        if i < 4:
          color=cm.actonS(i+2)
          alpha=0.5
        else:
          color=cm.lipariS(5)
          alpha=1

        bars = ax.bar(x_positions + offset, row_values,
                      width=bar_width,
                      yerr=pct_std[i] if pct_std is not None else None,
                      capsize=2,
                      label=label,
                      edgecolor='black',
                      linewidth=0.7,
                      color=color,
                      zorder=3,
                      alpha=alpha)

        for bar, val in zip(bars, row_values):
            # Formatting value to 1 decimal place
            label_text = f"{val:.1f}"

            # Determine Y position
            if val < 0:
                if val <= -60:
                    bar.set_hatch(hatch_pattern)
                # Negative: Place just above the x-axis (0 line)
                y_pos = 2  # Fixed small offset above 0
                va = 'bottom'
            else:
                # Positive: Place just above the bar
                y_pos = val + 2
                va = 'bottom'

            ax.text(bar.get_x() + bar.get_width() / 2,
                    y_pos,
                    label_text,
                    ha='center',
                    va=va,
                    fontsize=7,        # Slightly smaller font to fit nicely
                    color=color,       # Match the bar color
                    fontweight='bold',
                    zorder=4)

            #if bar.get_height() < limit:
            #  ax.plot(bar.get_x() + bar.get_width() / 2, limit, marker='d', color='white', markeredgecolor='black',
            #    markersize=10, clip_on=False, zorder=10)

    # 5. Styling
    # ----------
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('#4a5568')

    # X-Axis Ticks & Labels
    ax.set_xticks(x_positions)

    # Only show X-tick labels if requested (e.g., usually only on the bottom plot)
    if show_xlabel:
        ax.set_xticklabels(x_labels, fontsize=14, fontweight='bold')
        if n_plots == 2:
            ax.tick_params(axis='x', pad=65, length=0)
        elif n_plots == 5:
            ax.tick_params(axis='x', pad=75, length=0)
    else:
        ax.set_xticklabels([]) # Hide labels
        ax.tick_params(axis='x', length=0)

    if len(x_positions) > 1:
        separators = (x_positions[:-1] + x_positions[1:]) / 2
        for x in separators:
            ax.axvline(x, color='black', linestyle='--', linewidth=0.8, alpha=0.6, zorder=0)

    # Y-Axis
    if is_standalone:
      ax.set_ylabel(r'Change from baseline [\%]', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    # Legend (Conditional)
    if show_legend and not is_standalone:
        # Place legend outside to the right, slightly aligned to top
        if n_plots == 2:
          ax.legend(title='Evaluation Dataset',
                    loc='lower left',
                    bbox_to_anchor=(0, -0.45),
                    ncol=3,
                    frameon=True,
                    fancybox=True,
                    framealpha=0.8,
                    facecolor='white',
                    edgecolor='#cccccc',
                    fontsize=12)
        elif n_plots == 5:
          ax.legend(title='Evaluation Dataset',
                    loc='lower left',
                    bbox_to_anchor=(0, -0.025),
                    ncol=3,
                    frameon=True,
                    fancybox=True,
                    framealpha=0.8,
                    facecolor='white',
                    edgecolor='#cccccc',
                    fontsize=12)

    if is_standalone:
      ax.legend(fontsize=8)

    if long_title:
        ax.set_title(long_title, fontsize=14, pad=10, loc='left')

    # 6. Finalize (Only if running standalone)
    # ----------------------------------------
    if is_standalone:
        ax.set_xlabel('Emulator Configuration', fontsize=14)
        if save:
            plt.savefig(FIGURES_DIR / f'{figname}.pdf', bbox_inches='tight')
        plt.show()

def plot_vertical_stacked_bars(baseline_results_list: list[dict],
                               optimized_results_list: list[dict],
                               train_scenarios: list[str],
                               test_scenarios: list[str],
                               x_labels: list[str],
                               leg_labels: list[str],
                               weights: list[int],
                               titles: list[str] = None,
                               save: bool = False,
                               figname: str = 'stacked_comparison',
                               seed_baseline_results_list: list[list[dict] | None] = None,
                               seed_optimized_results_list: list[list[dict] | None] = None) -> None:
    """
    Creates N vertical subplots using the plot_grouped_improvement_bars logic.
    Assumes baseline_results_list and optimized_results_list have the same length.

    `seed_baseline_results_list`/`seed_optimized_results_list` (optional, default
    None): one entry per panel, each either None (that panel plots as a normal
    single point estimate) or a list of per-seed result dicts (that panel plots
    mean +/- seed spread via plot_grouped_improvement_bars' seed mode) - lets
    different panels use different modes in the same figure (e.g. a CO2-only
    panel with seed spread next to a Multi-agent panel that isn't multi-seeded
    yet). Leaving both at None (the default) is fully backward-compatible.
    """

    n_plots = len(baseline_results_list)

    # Create the figure with vertical subplots
    # Height scales with number of plots to maintain aspect ratio
    fig, axes = plt.subplots(nrows=n_plots, ncols=1,
                             figsize=(16, 3 * n_plots),
                             sharey=False, # Y-scales might differ between datasets
                             constrained_layout=True)

    # Ensure axes is iterable even if n_plots=1
    if n_plots == 1: axes = [axes]

    for i, ax in enumerate(axes):
        # Determine specific inputs for this subplot
        curr_base = baseline_results_list[i]
        curr_opt = optimized_results_list[i]
        curr_title = titles[i] if titles and i < len(titles) else ''
        curr_seed_base = seed_baseline_results_list[i] if seed_baseline_results_list else None
        curr_seed_opt = seed_optimized_results_list[i] if seed_optimized_results_list else None

        # Determine layout flags
        is_first = (i == 0)
        is_last = (i == n_plots - 1)

        # Call the refactored plotting function on the specific axis
        plot_grouped_improvement_bars(
            baseline_results=curr_base,
            optimized_results=curr_opt,
            train_scenarios=train_scenarios,
            test_scenarios=test_scenarios,
            x_labels=x_labels,
            leg_labels=leg_labels,
            weights=weights,
            ax=ax,
            long_title=curr_title,
            show_legend=is_first,
            show_xlabel=is_last,
            save=False,
            seed_baseline_results=curr_seed_base,
            seed_optimized_results=curr_seed_opt,
            n_plots=n_plots
        )

        ax.set_ylim([-60, 100])
        if n_plots == 2:
          if i == 0:
            ax.text(
              0.01, 0.975, r"(a) CO$_2$-only", transform=ax.transAxes,
              ha="left", va="top", fontsize=14, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
            )
          elif i == 1:
            ax.text(
              0.01, 0.975, r"(b) Multi-agent", transform=ax.transAxes,
              ha="left", va="top", fontsize=14, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
            )
        elif n_plots == 5:
          if i == 0:
            ax.text(
              0.01, 0.975, r"(a) CO$_2$-only", transform=ax.transAxes,
              ha="left", va="top", fontsize=14, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
            )
          elif i == 1:
            ax.text(
              0.01, 0.975, r"(b) CH$_4$-only", transform=ax.transAxes,
              ha="left", va="top", fontsize=14, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
            )
          elif i == 2:
            ax.text(
              0.01, 0.975, r"(c) N$_2$O-only", transform=ax.transAxes,
              ha="left", va="top", fontsize=14, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
            )
          elif i == 3:
            ax.text(
              0.01, 0.975, r"(d) Sulfur-only", transform=ax.transAxes,
              ha="left", va="top", fontsize=14, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
            )
          elif i == 4:
            ax.text(
              0.01, 0.975, r"(e) BC-only", transform=ax.transAxes,
              ha="left", va="top", fontsize=14, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
            )

    if n_plots == 5:
      fig.get_layout_engine().set(h_pad=0.2)

    # Add a global X-axis label at the bottom of the figure
    fig.supxlabel('Emulator Configuration', fontsize=16)
    fig.supylabel(r'Performance change from baseline emulator [\%]', fontsize=16)

    if save:
        plt.savefig(FIGURES_DIR / f'{figname}.pdf', bbox_inches='tight')

    plt.show()


def get_global_value(data_dict: dict, scenario_name: str) -> float:
    """
    Helper to search for a scenario name within the nested dictionary structure
    {'scenario_set': {'scenario': {'global': value}}} and return the global value.
    """
    for set_key, scenarios in data_dict.items():
        if scenario_name in scenarios:
            try:
                return scenarios[scenario_name]['global']
            except KeyError:
                return np.nan
    return np.nan

def plot_scenario_difference_bars2(baseline_results: dict,
                                  optimized_results_list: list[dict],
                                  scenario_keys: list[str],
                                  legend_labels: list[str],
                                  co2_data: list[np.array],
                                  global_mean_temp: list[np.array],
                                  x_labels: list[str] = None,
                                  separator_indices: list[int] = None,
                                  group_labels: list[str] = None,
                                  save: bool = False,
                                  figname: str = 'scenario_differences') -> None:
    """
    Per-scenario bar chart of global NRMSE across baseline + multiple optimized
    variants, with optional group separators/labels. Used by 5a_paper_plots.ipynb.
    """

    # 1. Setup Data & Layout
    # ----------------------
    n_total = len(scenario_keys)

    if x_labels and len(x_labels) == n_total:
        labels = x_labels
    else:
        labels = scenario_keys

    layout = [
        ["Top1"],
        ["Top2"],
        ["Bottom"]
    ]

    fig, axd = plt.subplot_mosaic(
        layout,
        figsize=(8.7, 19.5),
        constrained_layout=True,
        height_ratios=[1, 1, 5]
    )

    axd["Top2"].sharex(axd["Top1"])
    axd["Top2"].sharey(axd["Top1"])
    axd["Top1"].tick_params(labelbottom=False)

    axes_co2 = [axd["Top1"], axd['Top2']]
    ax_bar = axd["Bottom"]

    t_min = np.min(global_mean_temp)
    t_max = np.max(global_mean_temp)

    # --- Top Time-Series Plots ---
    for i, co2 in enumerate(co2_data):
      axes_co2[i].plot(np.arange(1750, 2501), co2, lw=1.5, c=cm.actonS(2), label='Emissions')
      axes_co2[i].grid(axis='y', linestyle='--', alpha=0.3, zorder=0, c=cm.actonS(2))
      axes_co2[i].grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
      axes_co2[i].tick_params(axis='y', labelcolor=cm.actonS(2))

      ax_temp = axes_co2[i].twinx()
      ax_temp.plot(np.arange(1751, 2501), global_mean_temp[i], lw=1.5, ls='--', c=cm.actonS(4), label='Global mean temperature', zorder=0)
      ax_temp.grid(linestyle='--', alpha=0.3, zorder=0, c=cm.actonS(4))
      ax_temp.set_ylim(t_min - 0.75, t_max + 0.75)
      ax_temp.tick_params(axis='y', labelcolor=cm.actonS(4))

      axes_co2[i].set_ylabel(r'Emissions [GtCO$_2$/yr]',  fontsize=16, c=cm.actonS(2))
      ax_temp.set_ylabel(r"$\overline{\Delta T}(t)$ [$^\circ$C]", fontsize=16, rotation=270, labelpad=15, c=cm.actonS(4))

      if i == 0:
        axes_co2[i].text(
              0.02, 0.94, r"(a) Optimized emissions and resulting $\overline{\Delta T}(t)$ (const. IC)", transform=axes_co2[i].transAxes,
              ha="left", va="top", fontsize=16, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
            )
        lines_1, labels_1 = axes_co2[i].get_legend_handles_labels()
        lines_2, labels_2 = ax_temp.get_legend_handles_labels()
        ax_temp.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=True, loc='lower left',
                    fancybox=True, framealpha=0.8, facecolor='white', edgecolor='#cccccc', fontsize=14)
      else:
        axes_co2[i].text(
              0.02, 0.94, r"(b) Optimized emissions and resulting $\overline{\Delta T}(t)$ (sine IC)", transform=axes_co2[i].transAxes,
              ha="left", va="top", fontsize=16, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
            )

    axes_co2[0].set_xlim([1750, 2500])

    # --- Bottom Horizontal Bar Chart ---
    n_opts = len(optimized_results_list)
    total_group_width = 0.8
    bar_width = total_group_width / n_opts
    colors = [cm.osloS(i + 2) for i in range(n_opts)]

    y_positions = np.arange(n_total)
    ax_bar.invert_yaxis()

    # 2. Plotting Loop for Bar Chart
    # ------------------------------
    for opt_idx, opt_dict in enumerate(optimized_results_list):
        diff_values = []
        for scen in scenario_keys:
            base_val = get_global_value(baseline_results, scen)
            opt_val = get_global_value(opt_dict, scen)

            if np.isnan(base_val) or np.isnan(opt_val) or base_val == 0:
                diff_values.append(0)
            else:
                pct_change = ((base_val - opt_val) / base_val) * 100
                diff_values.append(pct_change)

        offset = (opt_idx - n_opts / 2) * bar_width + (bar_width / 2)

        bars = ax_bar.barh(y_positions + offset,
                   diff_values,
                   label=legend_labels[opt_idx],
                   height=bar_width,
                   color=colors[opt_idx],
                   edgecolor='black',
                   linewidth=0.7,
                   zorder=3)

        # Annotation Loop
        for bar, val in zip(bars, diff_values):
            if val < -70:
                bar.set_hatch('//')

            label_text = f"{val:.1f}"

            if val < 0:
                x_pos = 2
            else:
                x_pos = val + 2

            ax_bar.text(x_pos,
                        bar.get_y() + bar.get_height() / 2,
                        label_text,
                        ha='left',
                        va='center',
                        fontsize=12,
                        color=colors[opt_idx],
                        fontweight='bold',
                        zorder=4)

    # 3. Styling & User Requests
    # --------------------------
    ax_bar.set_yticks(y_positions)
    ax_bar.set_yticklabels(labels, ha='right', va='center')
    ax_bar.tick_params(axis='y', length=0, labelsize=14)

    ax_bar.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(False)

    # Draw a custom vertical line at x=0 to act as the baseline for the bars
    ax_bar.axvline(0, color='#4a5568', linewidth=1.2, zorder=0)

    ax_bar.set_xlim([-70, 90])
    if separator_indices:
        for idx in separator_indices:
            # Place the line halfway between the specified index and the next one (idx + 0.5)
            ax_bar.axhline(idx + 0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.6, zorder=0)

    if group_labels and separator_indices:
        boundaries = [-0.5] + [idx + 0.5 for idx in separator_indices] + [n_total - 0.5]

        for i, label_text in enumerate(group_labels):
            center_y = (boundaries[i] + boundaries[i+1]) / 2
            ax_bar.text(1.02, center_y, label_text,
                        transform=ax_bar.get_yaxis_transform(),
                        rotation=270,
                        ha='left',
                        va='center',
                        fontsize=18,
                        fontweight='bold',
                        color='#333333')

    ax_bar.text(
              0.02, 0.99, r"(c) MESM emulator performance summary", transform=ax_bar.transAxes,
              ha="left", va="top", fontsize=16, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
            )

    ax_bar.legend(title='Emulator IC',
                    loc='lower right',
                    title_fontsize=16,
                    frameon=True,
                    fancybox=True,
                    framealpha=0.8,
                    facecolor='white',
                    edgecolor='#cccccc',
                    fontsize=14)

    ax_bar.set_ylabel('Scenario', fontsize=18)
    fig.supxlabel(r'Performance change from baseline emulator [\%]', fontsize=18)

    if save:
        plt.savefig(FIGURES_DIR / f'{figname}.pdf', bbox_inches='tight')


def plot_individual_effects_summary(
    y_true_ind_effects: dict,
    y_hat_baseline: dict,
    y_hat_ind_effects: dict,
    train_scenarios_ind_effects: list[str],
    ppt: bool = True,
    save: bool = False,
    figname: str = 'individual_effects_ppt4',
) -> None:
  """Plot the multi-agent "individual effects" summary figure.

  Left column: SCM-projected vs. baseline- and optimal-emulator ("Opt. All") predicted
  temperature trajectories, one row per single-driver scenario (M-GHG, M-aer, G6sulfur).
  Right panel: emulator-fit scatter/regression across all three scenarios pooled together,
  with R^2 scores per emulator configuration.

  Only 'Opt. All' is drawn against the baseline (the other entries of
  train_scenarios_ind_effects are computed upstream but intentionally skipped here, matching
  the original figure design).
  """
  i_ppt = 0

  layout = [
          ["Left1", "Left1", "Left1", "Left1", "Left1", "Right", "Right", "Right"],
          ["Left2", "Left2", "Left2", "Left2", "Left2", "Right", "Right", "Right"],
          ["Left3", "Left3", "Left3", "Left3", "Left3", "Right", "Right", "Right"],
      ]

  plot_map = {"Left1":"M_GHG", "Left2":"M_AER", "Left3":"G6sulfur"}
  color_map = {'Opt. Tier 1':cm.actonS(2), 'Opt. DAMIP':cm.actonS(4), 'Opt. GeoMIP': cm.actonS(6), 'Opt. All':cm.osloS(2)}
  labels = [r"(a) Medium emissions, greenhouse gases only (DAMIP: $\it{M}$-$\it{GHG}$)",
            r"(b) Medium emissions, aerosols only (DAMIP: $\it{M}$-$\it{aer}$)",
            r"(c) High emissions with sulfur injection (GeoMIP: $\it{G6sulfur}$)"]
  plot_len = len(y_true_ind_effects["All"]["M_GHG"])
  x_vals = np.arange(1850, 2151)

  if ppt:
      figsize = (16.81, 7)
  else:
      figsize = (17.8, 7)

  fig, ax_dict = plt.subplot_mosaic(
      layout,
      figsize=figsize,
      constrained_layout=True,
      gridspec_kw={"wspace": 0.001, "hspace": 0.001}
  )

  for j, ax_label in enumerate(plot_map):
      ax_plot = ax_dict[ax_label]
      scen_plot = plot_map[ax_label]

      ax_plot.plot(x_vals, y_true_ind_effects['All'][scen_plot][100:plot_len], label='SCM-projected', c='black', ls='--', lw=2, alpha = 0.8)
      ax_plot.plot(x_vals, y_hat_baseline['All'][scen_plot][100:plot_len], label='Baseline Em.', lw=2, ls="-", c=cm.lipariS(5))

      for i, train in enumerate(reversed(train_scenarios_ind_effects)):
          train_label = train
          if train in ['Opt. Tier 2','Opt. DECK', 'Opt. CS3']:
              continue
          if train not in ['Opt. All']:
              continue
          alpha = 0.6
          zorder = 0
          ls = (0, (5, 4))
          if train == 'Opt. All':
              alpha = 1
              zorder = 10
              ls = '-'
          elif train == 'Opt. Tier 1':
              train_label = 'Opt. Prio. 1'
          ax_plot.plot(x_vals, y_hat_ind_effects[train]['All'][scen_plot][100:plot_len], alpha=alpha, label=train_label, zorder=zorder, lw=2, ls=ls, color=color_map[train])

      if ax_label != "Left3":
          ax_plot.sharex(ax_dict["Left3"])
          ax_plot.sharey(ax_dict["Left3"])
          ax_plot.tick_params(labelbottom=False)

      ax_plot.grid(linestyle='--', alpha=0.3, zorder=0)

      ax_plot.text(
              0.015, 0.915, labels[j], transform=ax_plot.transAxes,
              ha="left", va="top", fontsize=14, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
            )
      ax_plot.set_ylim([-4.2, 5.2])

  ax_plot.set_xlim([1850, 2150])
  ax_plot.set_xlabel("Year")

  ax_dict["Left2"].legend(ncol=2,
                    frameon=True,
                    fancybox=True,
                    framealpha=0.8,
                    facecolor='white',
                    edgecolor='#cccccc',
                    fontsize=12)

  fig.supylabel(r'Temperature anomaly [$^\circ$C]', fontsize=16)

  ax_plot = ax_dict['Right']

  scenario_legend_handles = {}
  marker_legend_handles = []
  scenarios = ["M_GHG","M_AER","G6sulfur"]
  model_performance = defaultdict(lambda: {'true': [], 'pred': [], 'color': None})

  step = 15
  size = 35
  markers = ['o','s','d']
  base_color = cm.lipariS(5)

  for j, scen_plot in enumerate(scenarios):
      y_true = y_true_ind_effects['All'][scen_plot][100:plot_len]
      y_pred_base = y_hat_baseline['All'][scen_plot][100:plot_len]
      ax_plot.scatter(y_true[::step], y_pred_base[::step],
                      marker=markers[j], color=base_color, s=size,
                      facecolor='white', linewidths=1, alpha=0.4)

      model_performance['Baseline Em.']['true'].extend(y_true)
      model_performance['Baseline Em.']['pred'].extend(y_pred_base)
      model_performance['Baseline Em.']['color'] = base_color

      for i, train in enumerate(reversed(train_scenarios_ind_effects)):
              if train in ['Opt. Tier 2','Opt. DECK', 'Opt. CS3']:
                  continue
              y_pred = y_hat_ind_effects[train]['All'][scen_plot][100:plot_len]

              model_performance[train]['true'].extend(y_true)
              model_performance[train]['pred'].extend(y_pred)
              model_performance[train]['color'] = color_map[train]

  emulator_handles = [Line2D([0], [0], ls='--', color='k', lw=2, label=f"Ideal fit (1:1)")]
  for name, data in model_performance.items():
      markeredgewidth = 1
      ls = (0, (5, 4))
      if name not in ['Baseline Em.','Opt. All']:
          continue
      if name == 'Opt. All' or name == 'Baseline Em.':
          markeredgewidth = 1.75
          ls = '-'
      elif name == 'Opt. Tier 1':
          name = 'Opt. Prio. 1'
      sns.regplot(
          x=np.array(data['true']),
          y=np.array(data['pred']),
          scatter=False,
          ci=95,
          color=data['color'],
          ax=ax_plot,
          truncate=False,
          line_kws={'linewidth': 1.5, 'zorder': 11, 'linestyle':ls}
      )
      score = r2_score(data['true'], data['pred'])
      handle = Line2D([0], [0], ls=ls, color=data['color'], lw=2, label=f"{name} ($R^2={score:.2f}$)")
      emulator_handles.append(handle)

  scenario_handles = []
  scen_labels = [r'$\it{M}$-$\it{GHG}$',r'$\it{M}$-$\it{aer}$',r'$\it{G6sulfur}$']
  for marker, label in zip(markers, scen_labels):
      handle = Line2D([0], [0], marker=marker, color='w', label=label,
                      markerfacecolor='white', markeredgecolor='k', markersize=8)
      scenario_handles.append(handle)

  scen_legend = ax_plot.legend(handles=scenario_handles, loc='lower right', title='Scenario', fontsize=12, title_fontsize=12)
  ax_plot.add_artist(scen_legend)

  emu_legend = ax_plot.legend(handles=emulator_handles,
                              loc='upper left', fontsize=12,
                              bbox_to_anchor=(0.019893604239317852,
                                              0.6345526951619422,
                                              0.43059349211779896,
                                              0.2872407910360216),
                              title='Emulator configuration',
                              mode='expand',
                              title_fontsize=12,
                              borderaxespad=0)
  shift = [65, 2, 26, 0, 3, 6]
  for i, text in enumerate(emu_legend.get_texts()):
      text.set_horizontalalignment('right')
      text.set_x(shift[i])

  ax_plot.grid(linestyle='--', alpha=0.3, zorder=0)
  ax_plot.set_ylabel(r'Emulated temperature anomaly [$^\circ$C]')
  ax_plot.set_xlabel(r'SCM-projected temperature anomaly [$^\circ$C]')
  ax_plot.axline((0, 0), slope=1, color='black', alpha=0.8, linestyle='--', linewidth=1.5, zorder=0)
  ax_plot.set_xlim([-1.5, 5.175])
  ax_plot.set_ylim([-4.5, 6.5])
  final_handles = list(scenario_legend_handles.values()) + marker_legend_handles

  ax_plot.text(
              0.03, 0.975, '(d) Emulator fit', transform=ax_plot.transAxes,
              ha="left", va="top", fontsize=14, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
            )

  if save:
      plt.savefig(FIGURES_DIR / f'{figname}.pdf')

# ==================================================================
# Part 1: FaIR scenario plots (moved from run_fair.py)
# ==================================================================

def plot_emissions(emis_dict: dict, agent: str, experiment_id: str, MIP: str = 'ScenarioMIP_tier1') -> None:
    """Plot per-scenario emissions time series for one agent (colors/line styles from run_fair.colors)."""
    fig, ax = plt.subplots(figsize=(14,4), constrained_layout=True)
    for tag in emis_dict.keys():
        if MIP in ['ScenarioMIP_tier1','ScenarioMIP_tier2','GeoMIP']:
            if tag == 'historical':
                years = np.arange(1750, 2024)
                ls = '-'
            elif 'ext' not in tag:
                years = np.arange(2024, 2151)
                ls = '-'
            else:
                years = np.arange(2024, 2501)
                ls = '--'
        elif MIP == 'DECK':
            if 'abrupt' in tag:
                years = np.arange(1750, 2051)
                ls = '--'
            elif '1pct' in tag:
                years = np.arange(1750, 1901)
                ls = '-'
        elif MIP == 'CS3':
            if tag == 'historical':
                years = np.arange(1750, 2006)
                ls = '-'
            else:
                years = np.arange(2006, 2151)
                ls = '-'
        elif MIP == 'Optimal':
            years = np.arange(1750, 2501)
            ls = '-'
        else:
            raise ValueError(f'Error: type {MIP} not recognized.')

        if MIP == 'DECK':
            ax.semilogy(years, emis_dict[tag][agent], label=tag, ls=ls, lw=2, c=run_fair.colors[tag])
        else:
            ax.plot(years, emis_dict[tag][agent], label=tag, ls=ls, lw=2, c=run_fair.colors[tag])

    units = {'CO2':'Gt',
             'CH4':'Mt',
             'N2O':'Mt',
             'Sulfur':'Mt',
             'BC':'Mt'}

    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{agent} emissions ({units[agent]})')
    ax.set_title(f'{experiment_id} scenarios')
    #ax.set_xlim([1750,2500])
    plt.grid(True, alpha=0.3)

    return

def plot_delT(delT_dict: dict, scen_to_plot: list, experiment_id: str, MIP: str = 'ScenarioMIP') -> None:
    """Plot per-scenario GMST anomaly time series (colors/line styles from run_fair.colors)."""
    fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)
    for tag in scen_to_plot:
        if MIP in ['ScenarioMIP','GeoMIP']:
            if tag == 'historical':
                years = np.arange(1750, 2024)
                ls = '-'
            elif 'ext' not in tag:
                years = np.arange(2024, 2151)
                ls = '-'
            else:
                years = np.arange(2024, 2501)
                ls = '--'
        elif MIP == 'DECK':
            if 'abrupt' in tag:
                years = np.arange(1750, 2051)
                ls = '--'
            elif '1pct' in tag:
                years = np.arange(1750, 1901)
                ls = '-'
        elif MIP == 'CS3':
            if tag == 'historical':
                years = np.arange(1750, 2006)
                ls = '-'
            else:
                years = np.arange(2006, 2151)
                ls = '-'
        elif MIP == 'Optimal':
            years = np.arange(1750, 2501)
            ls = '-'
        else:
            raise ValueError(f'Error: type {MIP} not recognized.')

        ax.plot(years, delT_dict[tag], label=tag, ls=ls, lw=2, c=run_fair.colors[tag])

    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel(r'$\overline{\Delta T}(t)$')
    ax.set_title(f'{experiment_id} scenarios')

    return

# ==================================================================
# Part 2a: JAX SCM calibration plots (moved from utils_FaIR_JAX.py)
# ==================================================================

def plot_FaIR_v_JAX(delT_dict_FaIR: dict, delT_dict_JAX: dict) -> None:
    """Overlay FaIR (dashed) vs. JAX SCM (solid) GMST per scenario, for calibration sanity checks."""
    fig, ax = plt.subplots(constrained_layout=True)
    for i, scen in enumerate(delT_dict_JAX):
        if scen in utils_FaIR_JAX.needs_historical:
            ax.plot(delT_dict_JAX[scen][274:], c=f"C{i}", label=scen)
            ax.plot(delT_dict_FaIR[scen], ls='--', c=f"C{i}", label=scen)
        else:
            ax.plot(delT_dict_JAX[scen], c=f"C{i}", label=scen)
            ax.plot(delT_dict_FaIR[scen], ls='--', c=f"C{i}", label=scen)

    ax.set_xlabel('Year')
    ax.set_ylabel(r'$\overline{\Delta T}(t)$ [$^\circ$ C]')
    #fig.legend()

def plot_calibration_results(
    conc_dict: dict | None,
    emis_dict: dict | None,
    target_dict: dict,
    theta0: jnp.ndarray,
    theta_opt: jnp.ndarray,
    dt: float = 0.1,
    calib: str = "Climate",
    mode: str = 'FaIR',
    base_params: dict | None = None,
) -> None:
    """
    Plots the model performance before and after optimization.

    Args:
        conc_dict: {scenario: (3, T)} Input concentrations (Used in 'Climate' mode).
        emis_dict: {scenario: (5, T)} Input emissions (Used in both modes).
        target_dict: {scenario: (T,)} Target data.
                     If mode='Climate', this is Temperature (K).
                     If mode='Carbon', this is CO2 Concentration (ppm).
        mode: "Climate" (Conc -> Temp) or "Carbon" (Emis -> Conc).
        base_params: params dict supplying every theta field not tuned during
            calibration. Defaults to utils_FaIR_JAX.FAIR_PARAMS; pass
            utils_FaIR_JAX.MESM_PARAMS when plotting an MESM calibration
            (matches the base_params now threaded through
            utils_FaIR_JAX.calibrate_carbon_cycle/calibrate_climate_sensitivity).
    """
    if base_params is None:
        base_params = utils_FaIR_JAX.FAIR_PARAMS
    # 1. Prepare Data
    # Use target_dict for keys/lengths as it is required in both modes
    scenario_names = list(target_dict.keys())
    n_scens = len(scenario_names)

    # Determine maximum time length
    if calib == 'Climate':
      max_len = max([target_dict[s].shape[0] for s in scenario_names])
    else:
      max_len = max([target_dict[s].shape[1] for s in scenario_names])

    # Initialize padded arrays
    # Conc: (N_scen, 3, Max_T), Emis: (N_scen, 5, Max_T), Target: (N_scen, Max_T)
    conc_matrix = jnp.zeros((n_scens, 3, max_len))
    emis_matrix = jnp.zeros((n_scens, 5, max_len))
    target_matrix = jnp.zeros((n_scens, max_len))
    loss_mask = jnp.zeros((n_scens, max_len))

    # Fill arrays
    for i, name in enumerate(scenario_names):
        # Handle inputs based on availability
        if conc_dict is not None and name in conc_dict:
            c_data = conc_dict[name]
            conc_matrix = conc_matrix.at[i, :, :c_data.shape[1]].set(c_data)

        if emis_dict is not None and name in emis_dict:
            e_data = emis_dict[name]
            emis_matrix = emis_matrix.at[i, :, :e_data.shape[1]].set(e_data)

        t_data = target_dict[name]
        if calib == 'Climate':
          curr_len = t_data.shape[0]
          target_matrix = target_matrix.at[i, :curr_len].set(t_data)
        else:
          curr_len = t_data.shape[1]
          target_matrix = target_matrix.at[i, :curr_len].set(t_data[0,:])
        loss_mask = loss_mask.at[i, :curr_len].set(1.0)

    # 2. Setup Vmap Inputs
    T_steps = max_len
    years_template = jnp.arange(T_steps, dtype=jnp.float32)
    years_batch = jnp.tile(years_template, (n_scens, 1))

    params_initial = utils_FaIR_JAX.params_from_theta(theta0, base_params)
    params_optimized = utils_FaIR_JAX.params_from_theta(theta_opt, base_params)

    # 3. Generate Predictions based on Mode
    if calib == "Carbon":
        # Emissions -> Concentrations (CO2)
        # vmap over simulate_temp(years, emissions, params, dt)
        vmap_model = jax.vmap(utils_FaIR_JAX.simulate_temp, in_axes=(0, 0, None, None, None))

        # Helper to extract just the CO2 ppm from the full output dict
        def get_preds(params):
            res_dict = vmap_model(years_batch, emis_matrix, mode, params, dt)
            return res_dict["Catm_ppm"]

        preds_init = get_preds(params_initial)
        preds_opt = get_preds(params_optimized)
        ylabel = r'$CO_2$ Concentration (ppm)'

    elif calib == "Climate":
        # Concentrations -> Temperature
        # vmap over simulate_temp_prescribed_conc(years, conc, emis, params, dt)
        vmap_model = jax.vmap(utils_FaIR_JAX.simulate_temp_prescribed_conc, in_axes=(0, 0, 0, None, None))

        preds_init = vmap_model(years_batch, conc_matrix, emis_matrix, params_initial, dt)
        preds_opt = vmap_model(years_batch, conc_matrix, emis_matrix, params_optimized, dt)
        ylabel = r'$\Delta T$ (K)'

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 4. Plotting
    fig, axes = plt.subplots(1, n_scens, figsize=(6 * n_scens, 5), sharey=True)
    if n_scens == 1: axes = [axes]

    for i, name in enumerate(scenario_names):
        ax = axes[i]
        actual_len = int(jnp.sum(loss_mask[i]))
        years = jnp.arange(actual_len)

        # Plot Data
        ax.plot(years, target_matrix[i, :actual_len], color='black', label='Target (Data)', lw=2)
        ax.plot(years, preds_init[i, :actual_len], color='tab:blue', linestyle=':', label='SCM (Pre-Opt)', lw=2)
        ax.plot(years, preds_opt[i, :actual_len], color='tab:red', linestyle='--', label='SCM (Post-Opt)', lw=2)

        ax.set_title(f'Scenario: {name}')
        ax.set_xlabel('Years')
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    plt.tight_layout()
    plt.show()

def plot_model_comparison(emis_dict: dict, target_dict: dict, theta0: jnp.ndarray, mode: str = 'FaIR', dt: float = 0.1) -> None:
    """
    Plots a comparison between:
    1. The 'True' target temperature data.
    2. The simulation using optimized parameters (theta0).
    3. The simulation using default parameters for the specified mode (e.g., MESM defaults).

    Args:
        emis_dict: {scenario: (5, T)} Input emissions.
        target_dict: {scenario: (T,)} Target Temperature data.
        theta0: Optimized parameter vector.
        mode: 'FaIR' or 'MESM'. Determines the default parameters compared against.
    """
    # 1. Parameter Setup
    # Select the correct base parameters for the mode to reconstruct theta0
    if mode == 'FaIR':
        base_params = utils_FaIR_JAX.FAIR_PARAMS
    elif mode == 'MESM':
        base_params = utils_FaIR_JAX.MESM_PARAMS
    else:
        raise ValueError(f"Unknown mode: {mode}")

    params_opt = utils_FaIR_JAX.params_from_theta(theta0, base_params=base_params)

    # 2. Prepare Data (Padding and Batching)
    scenario_names = list(target_dict.keys())
    n_scens = len(scenario_names)

    # Determine maximum time length (looking at target data)
    lengths = [target_dict[s].shape[0] for s in scenario_names]
    max_len = max(lengths)

    # Initialize padded arrays
    emis_matrix = jnp.zeros((n_scens, 5, max_len))
    target_matrix = jnp.zeros((n_scens, max_len))
    loss_mask = jnp.zeros((n_scens, max_len))

    for i, name in enumerate(scenario_names):
        # Emissions
        if name in emis_dict:
            e_data = emis_dict[name]
            # Clamp to max_len if necessary
            curr_len_e = min(e_data.shape[1], max_len)
            emis_matrix = emis_matrix.at[i, :, :curr_len_e].set(e_data[:, :curr_len_e])

        # Targets
        t_data = target_dict[name]
        curr_len_t = min(t_data.shape[0], max_len)
        target_matrix = target_matrix.at[i, :curr_len_t].set(t_data[:curr_len_t])

        # Store valid length for plotting
        loss_mask = loss_mask.at[i, :curr_len_t].set(1.0)

    # 3. Setup Vmap Inputs
    years_template = jnp.arange(max_len, dtype=jnp.float32)
    years_batch = jnp.tile(years_template, (n_scens, 1))

    # 4. Run Simulations
    # Vmap signature: (years, emis, mode, params, dt)
    vmap_model = jax.vmap(utils_FaIR_JAX.simulate_temp, in_axes=(0, 0, None, None, None))

    # Run A: Optimized Parameters
    res_opt = vmap_model(years_batch, emis_matrix, mode, params_opt, dt)
    preds_opt = res_opt["GMST"]

    # Run B: Default Mode Parameters
    # We pass params=None so simulate_temp uses the defaults for 'mode'
    res_default = vmap_model(years_batch, emis_matrix, mode, None, dt)
    preds_default = res_default["GMST"]

    # 5. Plotting
    fig, axes = plt.subplots(1, n_scens, figsize=(6 * n_scens, 5), sharey=True)
    if n_scens == 1: axes = [axes]

    for i, name in enumerate(scenario_names):
        ax = axes[i]
        actual_len = int(jnp.sum(loss_mask[i]))
        years = jnp.arange(actual_len)

        # Plot Truth
        ax.plot(years, target_matrix[i, :actual_len], color='black', label='Target (Data)', lw=2)

        # Plot Default
        ax.plot(years, preds_opt[i, :actual_len], color='tab:red', linestyle='--',
                label='Default (FaIR)', lw=2)

        # Plot MESM calibration
        ax.plot(years, preds_default[i, :actual_len], color='tab:blue', linestyle=':',
                label=f'Calibrated ({mode})', lw=2)

        ax.set_title(f'Scenario: {name}')
        ax.set_xlabel('Years')
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(r'$\Delta T$ (K)')
    axes[0].legend()
    plt.tight_layout()
    plt.show()

# ==================================================================
# Part 3/4: inverse-optimization & emulator evaluation plots (moved from utils_inverse.py)
# ==================================================================

def plot_mlp_predictions(params: list[dict], Xs: jnp.ndarray, y: jnp.ndarray, metric: str = "NRMSE", title_prefix: str = "MLP fit") -> None:
    """Plot an MLP's prediction vs. truth for one scenario, titled with its NRMSE."""
    yhat = utils_inverse.mlp_forward(params, Xs)
    loss_val = utils_inverse._nrmse(yhat, y)

    plt.figure(figsize=(8,3))
    plt.plot(np.asarray(y),    label="truth", alpha=0.8)
    plt.plot(np.asarray(yhat), label="pred",  alpha=0.8)
    plt.legend(); plt.xlabel("time step"); plt.ylabel("target")
    plt.title(f"{title_prefix} - {metric.upper()}: {loss_val:.4f}")
    plt.tight_layout(); plt.show()

def plot_inverse_results(
    results: dict,
    baseline_error: float,                  # 1) dashed reference line
    active_agents: tuple | None = None,              # 4) only plot these agents’ emissions
    agent_units: dict | None = None,                # optional: {'CO2':'GtCO₂/yr','CH4':'MtCH₄/yr'}
    max_lines: int = 11,                    # 7) up to 12 emissions curves
    pred_scenario: str | None = None,
    baseline_preds: jnp.ndarray | None = None,
) -> None:
    """
    Multipanel plot of an optimize_emissions_inverse `results` dict:
      (a) NRMSE vs update step
      (b) Optimal emissions profiles (one subplot per active agent)
      (c) Training temperature trajectory
      (d) Predictions vs Truth
    """
    errors = jnp.asarray(results["errors"])
    U_traj = results["U_traj"]
    preds_traj = results.get("preds_traj", [])

    # --- helpers to read U_traj --------------------------------
    def _agents_in_state(state):
        if isinstance(state, (tuple, list)) and len(state) == 2:
            return ("CO2", "CH4")
        if isinstance(state, dict):
            return tuple(state.keys())
        raise ValueError("Unrecognized U_traj state format.")

    def _get_series(state, agent):
        if isinstance(state, (tuple, list)):
            if agent == "CO2": return jnp.asarray(state[0]).reshape(-1)
            if agent == "CH4": return jnp.asarray(state[1]).reshape(-1)
            raise KeyError(f"Agent {agent} not found in tuple state.")
        return jnp.asarray(state[agent]).reshape(-1)

    # Figure out which agents exist in this run
    present_agents = _agents_in_state(U_traj[0])
    if active_agents is None:
        active_agents = present_agents
    else:
        active_agents = tuple(a for a in active_agents if a in present_agents)

    # Units
    if agent_units is None:
        agent_units = {"CO2": "Gt/yr", "CH4": "Mt/yr", "N2O": "Mt/yr", "Sulfur":"Mt/yr", "BC": "Mt/yr"}

    n_update_steps = int(results.get("updates_done", max(0, errors.shape[0] - 1)))

    # years length
    if len(active_agents) == 0:
        probe_agent = present_agents[0]
    else:
        probe_agent = active_agents[0]
    T = int(_get_series(U_traj[-1], probe_agent).shape[0])

    # --- DYNAMIC SUBPLOTS ---
    # Rows: 1 (NRMSE) + N (Agents) + 1 (TrainTemp) + 1 (Preds)
    n_agents = len(active_agents)
    n_rows = 3 + n_agents

    # Adjust figure height based on number of rows to maintain aspect ratio
    fig_height = 2.5 * n_rows
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=1,
        figsize=(9, fig_height),
        constrained_layout=True
    )

    # Handle single axis case (unlikely but safe)
    if n_rows == 1: axes = [axes]

    # Assign axes
    ax_err = axes[0]
    ax_emis_list = axes[1 : 1 + n_agents] # Slice for emissions
    ax_train = axes[-2]
    ax_pred = axes[-1]

    # Share x-axis between all emissions plots and the training temperature
    for ax in ax_emis_list:
        ax.sharex(ax_train)

    # ----- Panel (a): scaled RMSE vs update step --------------------------------
    x_err = jnp.arange(errors.shape[0])
    ax_err.semilogy(x_err, errors, label="NRMSE")
    ax_err.axhline(float(baseline_error), ls="--", c='r', lw=1.5, label=f"Baseline emulator (avg.)")
    ax_err.set_xlim(0, n_update_steps)
    ax_err.set_xlabel("Update step")
    ax_err.set_ylabel("NRMSE")
    ax_err.text(
        0.015, 0.93, "NRMSE vs. Update Step", transform=ax_err.transAxes,
        ha="left", va="top", fontsize=16, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
    )
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best")

    # ----- Panel (b): Optimal emissions (One subplot per agent) -----------------
    n_total = len(U_traj)
    if n_total <= max_lines:
        sel_steps = np.arange(n_total, dtype=int)
    else:
        sel_steps = np.unique(np.linspace(0, n_total - 1, num=max_lines, dtype=int))

    # Iterate over agents and their corresponding axes
    for idx, ag in enumerate(active_agents):
        ax_curr = ax_emis_list[idx]
        has_any = False

        for i in sel_steps:
            state = U_traj[i]
            alpha = 0.3 + 0.7 * (i / max(1, len(U_traj) - 1))
            series = _get_series(state, ag)
            ax_curr.plot(series, alpha=alpha, label=f"Step {i}")
            has_any = True

        ax_curr.set_xlim(0, T)
        unit = agent_units.get(ag, "units/yr")
        ax_curr.set_ylabel(f"{ag} ({unit})")

        ax_curr.text(
            0.015, 0.93, f"Optimal {ag}", transform=ax_curr.transAxes,
            ha="left", va="top", fontsize=16, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.5)
        )

        ax_curr.grid(True, alpha=0.3)
        # Turn off x-tick labels for all emissions plots (shared with bottom)
        ax_curr.tick_params(axis="x", labelbottom=False)

        if has_any and idx == 0:
            # Only put legend on the first emissions plot to avoid clutter
            ax_curr.legend(ncol=3, fontsize=8, loc="best")

    # ----- Panel (c): Training temperature trajectory -------------------------
    train_temp_traj = results.get("train_temp_traj", [])
    if len(train_temp_traj) > 0:
        M_all = len(train_temp_traj)
        if M_all <= max_lines:
            sel_train = np.arange(M_all, dtype=int)
        else:
            sel_train = np.unique(np.linspace(0, M_all - 1, num=max_lines, dtype=int))

        for k, i in enumerate(sel_train):
            temp_list = train_temp_traj[i]
            if temp_list is None or len(temp_list) == 0: continue
            y_train = np.asarray(temp_list[0])
            if y_train.ndim > 1: y_plot = y_train[:, 0]
            else: y_plot = y_train

            alpha = 0.3 + 0.7 * (k / max(1, len(sel_train) - 1))
            ax_train.plot(y_plot, alpha=alpha, label=f"Step {i}")

        ax_train.set_xlabel("Year")
        ax_train.set_ylabel(r"$\overline{\Delta T}(t)$ ($^\circ$C)")
        ax_train.grid(True, alpha=0.3)
        ax_train.set_xlim(0, T)
        ax_train.text(
            0.015, 0.93, r"$\overline{\Delta T}(t)$ from Optimal Emissions", transform=ax_train.transAxes,
            ha="left", va="top", fontsize=16, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.5)
        )

    # ----- Panel (d): Predictions vs Truth ------------------------------------
    if pred_scenario is None:
        target_scen = preds_traj[0][0][0] if len(preds_traj) > 0 else "Unknown"
    else:
        target_scen = pred_scenario

    def _find_scen_idx(step_list, name):
        for j, (sc, _, _) in enumerate(step_list):
            if sc == name: return j
        return None

    N_all = len(preds_traj)
    if N_all > 0:
        if N_all <= max_lines:
            sel_pred = np.arange(N_all, dtype=int)
        else:
            sel_pred = np.unique(np.linspace(0, N_all - 1, num=max_lines, dtype=int))

        last_ytrue = None
        for k, i in enumerate(sel_pred):
            step_list = preds_traj[i]
            j = _find_scen_idx(step_list, target_scen)
            if j is None: continue
            _, yhat, ytrue = step_list[j]
            yhat, ytrue = jnp.asarray(yhat), jnp.asarray(ytrue)

            alpha = 0.3 + 0.7 * (k / max(1, len(sel_pred) - 1))
            ax_pred.plot(yhat, alpha=alpha, label=(f"Step {i}"))
            last_ytrue = ytrue

        if last_ytrue is not None:
            ax_pred.plot(last_ytrue, ls="--", c="C3", label=f"truth: {target_scen}")
            if baseline_preds is not None:
                ax_pred.plot(baseline_preds, ls="-.", c="C2", label=f"Baseline Emulator")
            ax_pred.set_xlim(0, int(last_ytrue.shape[0]) - 1)

    ax_pred.text(
        0.015, 0.93, f"Predictions vs Truth ({target_scen})", transform=ax_pred.transAxes,
        ha="left", va="top", fontsize=16, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.5)
    )
    ax_pred.set_xlabel("Year")
    ax_pred.set_ylabel(r"$\overline{\Delta T}(t)$ ($^\circ$C)")
    ax_pred.grid(True, alpha=0.3)
    ax_pred.legend(ncol=3, loc="best", fontsize=8)

    return

def plot_baseline_pred_delT(baseline_results: dict, baseline_pred_delT: dict, ground_truth_delT: dict) -> None:
    """Grid of per-scenario truth-vs-prediction GMST plots (titled with NRMSE) for the 'All' eval set."""
    test_set = 'All'
    N_scens = len(baseline_results[test_set])
    rows = int(np.ceil(N_scens / 3))
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5*rows), constrained_layout=True)
    axes = axes.ravel()

    for i, scen in enumerate(baseline_results[test_set]):
        if scen == 'mean':
            continue
        ax = axes[i]
        ax.plot(ground_truth_delT[test_set][scen],  label="truth", alpha=0.9)
        ax.plot(baseline_pred_delT[test_set][scen],  label="prediction", ls="--", alpha=0.9)
        ax.set_title(f"{scen} - NRMSE={baseline_results[test_set][scen]:.3f}")
        ax.set_xlabel("time step")
        ax.set_ylabel("GMST (K)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    return

def plot_zonal_predictions(results: dict, preds: dict, truths: dict, lat_coords: np.ndarray, eval_set: str = 'Tier 1') -> "matplotlib.figure.Figure":
    """
    Plots baseline emulator predictions vs ground truth at t=0, T/2, and T.
    Includes a secondary axis (twinx) showing the Zonal NRMSE profile.
    """
    scenarios = list(preds[eval_set].keys())
    n_scens = len(scenarios)

    # 1. Determine Shared Axis Limits
    # Flatten all arrays to find global min/max for Temperature
    all_p = np.concatenate([p.flatten() for p in preds[eval_set].values()])
    all_t = np.concatenate([t.flatten() for t in truths[eval_set].values()])
    t_min, t_max = min(all_p.min(), all_t.min()), max(all_p.max(), all_t.max())

    # Determine global limits for NRMSE (secondary axis)
    all_nrmse = []
    for s in scenarios:
        if 'zonal' in results[eval_set][s]:
            all_nrmse.append(results[eval_set][s]['zonal'])
    if all_nrmse:
        all_nrmse = np.concatenate(all_nrmse)
        n_min, n_max = 0, max(all_nrmse.max() * 1.1, 0.1) # Start at 0, add buffer
    else:
        n_min, n_max = 0, 1

    # 2. Setup Plot
    fig, axes = plt.subplots(n_scens, 3, sharey='row',
                             figsize=(16, 4.5 * n_scens),
                             constrained_layout=True)
    if n_scens == 1: axes = np.expand_dims(axes, 0) # Ensure 2D array

    # 3. Plotting Loop
    for i, scen in enumerate(scenarios):
        y_pred = preds[eval_set][scen]
        y_true = truths[eval_set][scen]

        # Metrics
        glob_nrmse_t_series = results[eval_set][scen]['global_t']
        zonal_nrmse = results[eval_set][scen]['zonal'] # (N_lat,)

        T = y_true.shape[0]
        time_steps = [0, T//2, T-1]

        for j, t in enumerate(time_steps):
            ax = axes[i, j]

            # --- Primary Axis: Temperature ---
            ax.plot(lat_coords, y_true[t], color=cm.batlowS(0), ls='-', label='MESM temp. anomaly', lw=1.5, alpha=0.8)
            ax.plot(lat_coords, y_pred[t], color=cm.lajollaS(2), ls='--', label='Emulated temp. anomaly', lw=1.5, alpha=0.9)

            ax.set_ylim(t_min, t_max)
            ax.set_xlabel("Latitude")
            if j == 0:
                ax.set_ylabel("Temp. anomaly [K]")
                ax.text(-0.25, 0.5, scen, transform=ax.transAxes,
                        rotation=90, va='center', ha='right', fontsize=14, fontweight='bold')

            # --- Secondary Axis: NRMSE ---
            #ax2 = ax.twinx()
            #ax2.plot(lat_coords, zonal_nrmse, 'g:', label='Zonal NRMSE', lw=1.5, alpha=0.6)
            #ax2.set_ylim(n_min, n_max)
            #ax2.tick_params(axis='y', labelcolor='tab:green')

            #if j == 2:
            #    ax2.set_ylabel("Zonal NRMSE", color='tab:green')
            #else:
            #    ax2.set_yticklabels([]) # Hide ticks on inner plots

            # Titles and Legends
            current_glob_nrmse = glob_nrmse_t_series[t]
            if j == 0:
                ax.set_title('Scenario start', fontsize=16)
            elif j == 1:
                ax.set_title('Scenario middle', fontsize=16)
            elif j == 2:
                ax.set_title('Scenario end', fontsize=16)
            #ax.set_title(f"Year {t} | Global NRMSE: {current_glob_nrmse:.4f}")

            if j == 0:
                # Combined legend
                lines, labels = ax.get_legend_handles_labels()
                #lines2, labels2 = ax2.get_legend_handles_labels()
                #ax.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=8)
                ax.legend(lines, labels, loc='upper left', fontsize=14)

            ax.set_xlim([-89, 89])
            ax.grid(True, alpha=0.3)

    return fig

def plot_comparison_results(
    result_paths: list[str],
    column_titles: list[str],
    baseline_errors: list[float],
    active_agents: tuple | None = None,
    agent_units: dict | None = None,
    max_lines: int = 11,
    save_path: str | None = None,
) -> None:
    """
    Comparison plot for multiple optimize_emissions_inverse checkpoints, used
    by supplementary_notebooks/SI_plots.ipynb's sensitivity-sweep figures
    (initial condition / architecture / feature window).
    Columns = datasets (defined by result_paths). Rows = 1 (NRMSE) + N (agents).

    Args:
        result_paths: File paths to pickle files containing 'errors'/'U_traj'.
        column_titles: Text box label for each column (top left).
        baseline_errors: Baseline error value (dashed line) per column.
    """

    # --- 1. Load Data ---
    loaded_results = []
    for path in result_paths:
        with open(path, 'rb') as f:
            loaded_results.append(pickle.load(f))

    n_cols = len(loaded_results)
    if len(column_titles) != n_cols:
        raise ValueError("Length of column_titles must match result_paths.")
    if len(baseline_errors) != n_cols:
        raise ValueError("Length of baseline_errors must match result_paths.")

    # --- 2. Helpers (Internal) ---
    def _agents_in_state(state):
        if isinstance(state, (tuple, list)) and len(state) == 2:
            return ("CO2", "CH4")
        if isinstance(state, dict):
            return tuple(state.keys())
        raise ValueError("Unrecognized U_traj state format.")

    def _get_series(state, agent):
        if isinstance(state, (tuple, list)):
            if agent == "CO2": return jnp.asarray(state[0]).reshape(-1)
            if agent == "CH4": return jnp.asarray(state[1]).reshape(-1)
            raise KeyError(f"Agent {agent} not found in tuple state.")
        return jnp.asarray(state[agent]).reshape(-1)

    # --- 3. Determine Layout based on First Dataset ---
    # (Assumes all datasets have roughly consistent agents, or uses active_agents to filter)
    first_res = loaded_results[0]
    present_agents = _agents_in_state(first_res["U_traj"][0])

    if active_agents is None:
        active_agents = present_agents
    else:
        active_agents = tuple(a for a in active_agents if a in present_agents)

    if agent_units is None:
        agent_units = {"CO2": "Gt/yr", "CH4": "Mt/yr", "N2O": "Mt/yr", "Sulfur":"Mt/yr", "BC": "Mt/yr"}

    n_agents = len(active_agents)
    n_rows = 1 + n_agents  # 1 for NRMSE, N for agents

    # Calculate T (time) from the first dataset/first agent to set x-limits
    # (Assumes all datasets span the same timeframe, which is standard for comparisons)
    probe_agent = active_agents[0] if len(active_agents) > 0 else present_agents[0]
    T = int(_get_series(first_res["U_traj"][-1], probe_agent).shape[0])

    # --- 4. Create Subplots ---
    fig_height = 2.5 * n_rows
    fig_width = 4.0 * n_cols

    # sharey='row' ensures all NRMSE plots share scale, and all CO2 plots share scale, etc.
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(fig_width, fig_height),
        sharey='row',
        sharex='row',
        constrained_layout=True
    )

    # Ensure axes is always 2D array [row, col] even if n_cols=1 or n_rows=1
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # --- 5. Plotting Loop ---
    for col_idx, results in enumerate(loaded_results):

        errors = jnp.asarray(results["errors"])
        U_traj = results["U_traj"]
        n_update_steps = int(results.get("updates_done", max(0, errors.shape[0] - 1)))

        # --- Row 0: NRMSE ---
        ax_err = axes[0, col_idx]
        x_err = jnp.arange(errors.shape[0])

        ax_err.loglog(x_err, errors, c=cm.batlowWS(1), label="Optimized emulator")
        ax_err.axhline(float(baseline_errors[col_idx]), ls="--", c=cm.lipariS(5), lw=1.5, label="Baseline emulator\nerror lower bound")

        ax_err.set_xlim(0, n_update_steps)
        if col_idx == 0:
            ax_err.set_ylabel("NRMSE")

        # Add the text box (Dataset Title) to the top plot of the column
        ax_err.text(
            0.03, 0.95, column_titles[col_idx], transform=ax_err.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
        )
        ax_err.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_err.legend(loc="upper right", fontsize=9)

        # --- Rows 1..N: Emissions ---
        n_total = len(U_traj)
        if n_total <= max_lines:
            sel_steps = np.arange(n_total, dtype=int)
        else:
            sel_steps = np.unique(np.linspace(0, n_total - 1, num=max_lines, dtype=int))

        for row_offset, ag in enumerate(active_agents):
            row_idx = 1 + row_offset
            ax_curr = axes[row_idx, col_idx]

            has_any = False
            for i in sel_steps:
                state = U_traj[i]
                # Check if this agent exists in this specific dataset's state
                # (Handles cases where datasets might have slightly different agent keys)
                try:
                    series = _get_series(state, ag)
                    alpha = 0.2 + 0.6 * (i / max(1, len(U_traj) - 1))
                    if i in [0, 100, 1000]:
                        if i == 0:
                            c = cm.lipariS(5)
                            alpha = 1
                        else:
                            c = cm.batlowWS(1)
                        ax_curr.plot(series, alpha=alpha, c=c, label=f"Step {i}")
                    else:
                        ax_curr.plot(series, alpha=alpha, c=cm.batlowWS(1))
                    has_any = True
                except KeyError:
                    continue

            ax_curr.set_xlim(0, T)
            ax_curr.grid(True, alpha=0.3)

            # Label Y-axis only for the first column
            if col_idx == 0:
                unit = agent_units.get(ag, "units/yr")
                ax_curr.set_ylabel(f"{ag} ({unit})")

            # Add Legend only on the first agent plot of the first column to avoid clutter
            if col_idx == 0 and row_offset == 0 and has_any:
                ax_curr.legend(fontsize=8, loc="lower left")

        # Set X-label on the bottom-most plot of this column
        axes[-1, col_idx].set_xlabel("Year")

    if save_path is not None:
        plt.savefig(FIGURES_DIR / f'{save_path}.pdf')

    return