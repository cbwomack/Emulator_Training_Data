# Imports

import numpy as np

## JAX
import jax
import jax.numpy as jnp

## Plotting
import matplotlib.pyplot as plt
from cmcrameri import cm

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

def plot_rmse_comparison(
    results_list,       # List of dictionaries
    baseline_error_list,     # Single float value
    agents,
    save=False
):
    """
    Creates a 1x5 multipanel plot of Panel (a) (Scaled RMSE vs update step).
    - Shares Y-axis across all subplots.
    - Compares up to 5 runs side-by-side.
    """

    # 1. Setup 1x5 Figure with shared Y-axis
    # constrained_layout ensures labels don't overlap
    fig, axes = plt.subplots(
        nrows=1, ncols=5,
        figsize=(20, 4),
        sharey=True,
        constrained_layout=True
    )

    # Ensure axes is iterable even if, for some reason, ncols was 1
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    # 2. Iterate through the axes and the results list
    for i, ax in enumerate(axes):
        # Check if we have a result for this subplot index
        if i < len(results_list):
            result = results_list[i]
            baseline_error = baseline_error_list[i]
            errors = jnp.asarray(result["errors"])
            x_err = jnp.arange(errors.shape[0])

            # --- Plotting ---
            cmap = cm.batlowS
            ax.semilogy(x_err, errors, label="Optimized emulator", lw=2, color=cmap(0))
            ax.axhline(float(baseline_error), ls="--", c='r', lw=1.5, label="Baseline emulator", color=cmap(1))

            ax.margins(x=0, y=0)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))

            # --- Formatting ---
            if i == 2:
              ax.set_xlabel("Update iteration no.", fontsize=18)
            ax.grid(True, alpha=0.3, which="both", ls="-")

            a = agents[i]
            ax.text(
              0.05, 0.95, fr"{a} only", transform=ax.transAxes,
              ha="left", va="top", fontsize=16, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.5)
            )

            # Only add the Y-label to the first plot to reduce clutter
            if i == 0:
                ax.set_ylabel("Scaled RMSE", fontsize=18)
                # Add legend only to the first plot
                ax.legend(loc="best", fontsize=14)

        else:
            # If we have fewer than 5 results, hide the empty axes
            ax.axis('off')

    if save:
      plt.savefig('Figures/single_forcing_optimization.pdf')

    return

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
                if i % 50 == 0:
                  y_data = np.asarray(series).reshape(-1)
                  alpha = 0.2 + 0.6 * (i / max(1, len(y_data) - 1))
                  if col_idx < 1:
                      x_data = np.arange(0, len(y_data)) + 2024
                  else:
                      x_data = np.arange(0, len(y_data)) + 1750
                  if i in [0, 500]:
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

    axes[0, 0].set_ylim([-30, 130])
    fig.supylabel(r"Emissions [GtCO$_2$/yr]")
    fig.supxlabel('Year')

    if save:
      plt.savefig('Figures/co2_multi_target.pdf')

    return