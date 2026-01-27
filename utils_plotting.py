# Imports

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

## Setup plots
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "sans-serif",
  "font.sans-serif": ["Helvetica Light"],
})

def plot_init(res, save=False):
  fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
  ax.plot(res['U_traj'][0]['CO2'], c=cm.batlowWS(1), lw=2)
  ax.set_ylabel(r'Emissions [GtCO$_2$/yr]')
  ax.set_xlabel('Year')
  ax.set_xlim([0, len(res['U_traj'][0]['CO2'])])
  if save:
    plt.savefig("Figures/init_emis.pdf", transparent=True)

def plot_tier1(years, tier1, group, save=False):
  fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
  for i, scen in enumerate(tier1):
    ax.plot(years[i], tier1[i], c=cm.batlowWS(i + 1), lw=2, label=group[i])

  ax.set_ylabel(r'Emissions [GtCO$_2$/yr]')
  ax.set_xlabel('Year')
  ax.set_xlim([1750, 2500])
  ax.legend(loc='upper left', fontsize=14)

  if save:
    plt.savefig("Figures/tier1.pdf", transparent=True)

def plot_updates(res, save=False):
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
    plt.savefig("Figures/emis_updates.pdf", transparent=True)

def plot_rmse_comparison_single(
    results_list,       # List of dictionaries
    baseline_error_list,     # Single float value
    agents,
    save=False
):
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
        ax.axhline(float(baseline_error), ls="--", c='r', lw=1.5, label="Baseline emulator\nerror lower bound")

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
            ax.set_ylabel("Evaluation loss (NRMSE)", fontsize=18)
            ax.legend(loc="best", fontsize=14)

        ax.set_xlim(left=1.1)

    fig.supxlabel('Update iteration no.', fontsize=18)

    if save:
      plt.savefig('Figures/single_forcing_optimization.pdf')

    return

def plot_rmse_comparison_multi(results, baseline_error, save=False):
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
    ax.axhline(float(baseline_error), ls="--", c='r', lw=1.5, label="Baseline emulator\nerror lower bound")

    # Styling
    ax.margins(x=0, y=0.2)
    ax.grid(True, alpha=0.3, which="both", ls="-")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("Evaluation loss (NRMSE)", fontsize=18)
    ax.set_xlabel('Update iteration no.', fontsize=18)
    ax.set_xlim(left=1.1)
    ax.legend(loc="best", fontsize=14)

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
        plt.savefig('Figures/multi_forcing_optimization.pdf')

    return

# --- Helper Functions ---

def _plot_agents(x_data, config, cmap):
    """Handles multi-axis plotting, coloring, and unified legends."""
    base_ax = config['ax']
    lines = []

    # Keep track of the last axis used to place text/legend on top
    last_ax = base_ax

    for i, (data, label) in enumerate(zip(config['data'], config['labels'])):
        color = cmap(i + 1)

        if i == 0:
            current_ax = base_ax
            spine_key = 'left'
        else:
            current_ax = base_ax.twinx()
            spine_key = 'right'
            # Offset the third spine so it doesn't overlap the second
            if i > 1:
                current_ax.spines["right"].set_position(("axes", 1.0 + (i-1)*0.1))

        ln = current_ax.plot(x_data, data, color=color, label=label)
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

def _add_textbox(ax, text, x, y):
    ax.text(
        x, y, text, transform=ax.transAxes,
        ha="left", va="top", fontsize=16, fontweight="bold",
        zorder=100, # Explicitly high zorder
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
    )

def plot_emissions_grid(opt_emissions, target_emissions, years, targets, groups, save=False):
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
      plt.savefig('Figures/co2_multi_target.pdf')

    return

def plot_co2_sulfur(co2, sulfur, save=False):
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
      plt.savefig('Figures/co2_sulfur_compare.pdf', bbox_inches='tight')

    return

from scipy.ndimage import gaussian_filter1d
def highlight_opposite_slopes(ax, x, y1, y2, min_length=10, sigma=1):
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


def plot_stacked_results(
    # --- Inputs for Top Row (Ground Truth) ---
    target_emissions,        # List of arrays (or nested list)
    target_years,            # List of arrays or single array corresponding to target_emissions

    # --- Inputs for Middle Row (Optimized) ---
    opt_emissions_history,   # List of arrays: history of optimized emission curves
    opt_temp_history,

    # --- Inputs for Bottom Row (Preds vs Truth) ---
    results,                 # Dictionary containing 'preds_traj'
    pred_scenario=None,      # Specific scenario name to plot (optional)

    # --- Styling Options ---
    opt_start_year=2024,     # Start year for optimized emissions x-axis
    max_lines=11,            # Max lines to plot for fading history
    save_path=None
):
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
        ax_truth.plot(x_data, y_data, lw=1.5, c=cmap(i + 2), label=f"Group {i}")

        # Update time bounds
        max_t = max(max_t, x_data[-1])
        min_t = min(min_t, x_data[0])

    ax_truth.grid(True, alpha=0.3)
    # Hide x-labels for top row (redundant due to sharex, but good practice to ensure)
    ax_truth.tick_params(labelbottom=False)

    ax_truth.text(
        0.015, 0.95, "(a) Ground truth emissions", transform=ax_truth.transAxes,
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

        if i == 0 or i == n_total - 1:
            ax_opt.plot(x_data, y_data, alpha=alpha, lw=1.5, c=cmap(1), label=f'Iteration {i}')
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
            ax_pred.plot(target_years, yhat, alpha=alpha, color=cmap(1))
            last_ytrue = ytrue

        # Plot Truth (Red dashed)
        if last_ytrue is not None:
            ax_pred.plot(target_years, last_ytrue, ls="--", c="C3", lw=2.0, label=r"Ground truth $\overline{\Delta T}$")
            # Ensure the shared X-axis covers the full range
            ax_pred.set_xlim(min_t, max(max_t, len(last_ytrue)))

    ax_pred.set_xlabel("Year")
    ax_pred.grid(True, alpha=0.3)
    ax_pred.legend(loc="lower left", fontsize=12)

    ax_pred.text(
        0.015, 0.95, f"(c) Predictions vs. ground truth", transform=ax_pred.transAxes,
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

        if i == 0 or i == n_total - 1:
            ax_opt_temp.plot(x_data, y_data, alpha=alpha, lw=1.5, c=cmap(1), label=f'Iteration {i*50}')
        else:
            ax_opt_temp.plot(x_data, y_data, alpha=alpha, lw=1.5, c=cmap(1))

        max_t = max(max_t, x_data[-1])

    ax_opt_temp.set_xlabel("Year")
    ax_opt_temp.grid(True, alpha=0.3)
    ax_opt_temp.legend(loc='lower left', fontsize=12)

    ax_opt_temp.text(
        0.015, 0.95, "(d) Optimized training temperature", transform=ax_opt_temp.transAxes,
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

    ax_pred.set_ylabel(r"$\overline{\Delta T}(t)$ ($^\circ$C)",fontsize=16)


    if save_path:
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
    plt.savefig(f'Figures/{figname}.pdf')

  return

def plot_grouped_improvement_bars(baseline_results: dict,
                                  optimized_results: dict,
                                  train_scenarios: list[str],
                                  test_scenarios: list[str],
                                  weights: list[int],
                                  ax: plt.Axes = None,  # <--- NEW ARGUMENT
                                  long_title: str = '',
                                  show_legend: bool = True, # <--- NEW ARGUMENT
                                  show_xlabel: bool = True, # <--- NEW ARGUMENT
                                  save: bool = False,
                                  figname: str = '') -> None:

    # 0. Handle Axis Creation (Backward Compatibility)
    # ----------------------------------------------
    is_standalone = False
    if ax is None:
        is_standalone = True
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    # 1. Organize Data (Your logic preserved)
    # ---------------------------------------
    n_test = len(test_scenarios)
    n_opts = len(train_scenarios)

    base_errors = np.zeros(n_test)
    opt_errors = np.zeros((n_test, n_opts))

    for i, test_key in enumerate(test_scenarios):
        try:
            base_errors[i] = baseline_results[test_key]['mean']
        except KeyError:
            base_errors[i] = np.nan

    for j, train_key in enumerate(train_scenarios):
        for i, test_key in enumerate(test_scenarios):
            try:
                opt_errors[i, j] = optimized_results[train_key][test_key]['mean']
            except KeyError:
                opt_errors[i, j] = np.nan

    w_arr = np.array(weights)
    avg_base_error = np.average(base_errors, weights=w_arr)
    avg_opt_errors = np.average(opt_errors, axis=0, weights=w_arr)

    # 2. Calculate Percent Improvement
    # --------------------------------
    pct_improvement = (base_errors[:, None] - opt_errors) / base_errors[:, None]
    avg_improvement = (avg_base_error - avg_opt_errors) / avg_base_error

    plot_data = np.vstack([pct_improvement, avg_improvement])
    plot_data = plot_data * 100

    row_labels = test_scenarios + ['Avg.']
    n_rows = len(row_labels)

    # 3. Plotting Setup
    # -----------------
    x_positions = np.arange(n_opts)
    total_group_width = 0.7
    bar_width = total_group_width / n_rows
    limit = -60

    # 4. Draw Grouped Bars
    # --------------------
    for i in range(n_rows):
        row_values = plot_data[i]
        label = row_labels[i]

        offset = (i - n_rows / 2) * bar_width + (bar_width / 2)
        #color = brewer2_light(i)
        if i < 4:
          color=cm.actonS(i+2)
          alpha=0.5
        else:
          color=cm.lipariS(5)
          alpha=1

        bars = ax.bar(x_positions + offset, row_values,
                      width=bar_width,
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
        ax.set_xticklabels(train_scenarios, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', pad=65, length=0)
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
        ax.legend(title='Evaluation Dataset',
                  loc='lower left',
                  bbox_to_anchor=(0, -0.45),
                  ncol=3, # Optional: Makes legend horizontal if preferred
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
            plt.savefig(f'Figures/{figname}.pdf', bbox_inches='tight')
        plt.show()

def plot_vertical_stacked_bars(baseline_results_list: list[dict],
                               optimized_results_list: list[dict],
                               train_scenarios: list[str],
                               test_scenarios: list[str],
                               weights: list[int],
                               titles: list[str] = None,
                               save: bool = False,
                               figname: str = 'stacked_comparison') -> None:
    """
    Creates N vertical subplots using the plot_grouped_improvement_bars logic.
    Assumes baseline_results_list and optimized_results_list have the same length.
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

        # Determine layout flags
        is_first = (i == 0)
        is_last = (i == n_plots - 1)

        # Call the refactored plotting function on the specific axis
        plot_grouped_improvement_bars(
            baseline_results=curr_base,
            optimized_results=curr_opt,
            train_scenarios=train_scenarios,
            test_scenarios=test_scenarios,
            weights=weights,
            ax=ax,                       # Pass the subplot axis
            long_title=curr_title,
            show_legend=is_first,        # Only show legend on the top plot
            show_xlabel=is_last,         # Only show x-labels on the bottom plot
            save=False                   # Never save inside the loop
        )

        ax.set_ylim([-60, 100])
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

    # Add a global X-axis label at the bottom of the figure
    fig.supxlabel('Emulator Configuration', fontsize=16)
    fig.supylabel(r'Performance change from baseline emulator [\%]', fontsize=16)

    if save:
        plt.savefig(f'Figures/{figname}.pdf', bbox_inches='tight')

    plt.show()



brewer2_light_rgb = np.divide([(102, 194, 165),
                               (252, 141,  98),
                               (141, 160, 203),
                               (231, 138, 195),
                               (166, 216,  84),
                               (255, 217,  47),
                               (229, 196, 148),
                               (179, 179, 179),
                               (202, 178, 214)],255)
brewer2_light = mcolors.ListedColormap(brewer2_light_rgb)