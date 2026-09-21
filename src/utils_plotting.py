# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5, Claude Sonnet 5, and Gemini 3.1 Pro.
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
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

## Local
from paths import DATA_DIR, FIGURES_DIR
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

def plot_fig01_init_emissions(res: dict, save: bool = False) -> None:
  """Plot the initial (step-0) CO2 trajectory from an optimize_emissions_inverse result."""
  fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
  ax.plot(res['U_traj'][0]['CO2'], c=cm.batlowWS(1), lw=2)
  ax.set_ylabel(r'Emissions [GtCO$_2$/yr]')
  ax.set_xlabel('Year')
  ax.set_xlim([0, len(res['U_traj'][0]['CO2'])])
  if save:
    plt.savefig(FIGURES_DIR / "fig01a_init_emissions.pdf", transparent=True)

def plot_fig01_tier1_scenarios(years: list, tier1: list, group: list[str], save: bool = False) -> None:
  """Plot one CO2 emissions line per tier-1 scenario in `tier1` (labeled by `group`)."""
  fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
  for i, scen in enumerate(tier1):
    ax.plot(years[i], tier1[i], c=cm.batlowWS(i + 1), lw=2, label=group[i])

  ax.set_ylabel(r'Emissions [GtCO$_2$/yr]')
  ax.set_xlabel('Year')
  ax.set_xlim([1750, 2500])
  ax.legend(loc='upper left', fontsize=14)

  if save:
    plt.savefig(FIGURES_DIR / "fig01b_tier1_scenarios.pdf", transparent=True)

def plot_fig01_emissions_updates(res: dict, save: bool = False) -> None:
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
    plt.savefig(FIGURES_DIR / "fig01c_emissions_updates.pdf", transparent=True)

def _preds_stride(results: dict, n_preds: int) -> int:
    """Outer iterations between consecutive preds_traj entries.
    """
    # Checked against None rather than truthiness: `errors` may be a JAX array,
    # whose truth value raises for more than one element.
    errors = results.get("errors")
    n_err = 0 if errors is None else len(errors)
    if n_preds > 1 and n_err > 1 and (n_err - 1) % (n_preds - 1) == 0:
        return (n_err - 1) // (n_preds - 1)
    meta = results.get("meta") or {}
    return int(meta.get("preds_every", 50))


def _highlight_indices(sel, middle: str = "midpoint") -> tuple:
    """First / middle / last of a subsampled index list.
    """
    sel = np.asarray(sel)
    if sel.size == 0:
        return ()
    mid = 1 if middle == "second" else sel.size // 2
    return (int(sel[0]), int(sel[min(mid, sel.size - 1)]), int(sel[-1]))


AGGREGATION_DEFAULT = "median"


def _aggregate_seeds(stacked, aggregation: str = AGGREGATION_DEFAULT,
                     mean_band: str = "std"):
    """Collapse a (n_seeds,...) stack to (centre, low, high) for plotting.
    """
    stacked = np.asarray(stacked)
    if aggregation == "median":
        return (np.median(stacked, axis=0),
                np.percentile(stacked, 25, axis=0),
                np.percentile(stacked, 75, axis=0))
    if aggregation == "mean":
        centre = stacked.mean(axis=0)
        if mean_band == "minmax":
            return centre, stacked.min(axis=0), stacked.max(axis=0)
        spread = stacked.std(axis=0)
        return centre, centre - spread, centre + spread
    raise ValueError(f"unknown aggregation {aggregation!r} (expected 'median' or 'mean')")


def _aggregation_label(aggregation: str, n: int) -> str:
    """Legend text naming the convention, so the figure is self-describing."""
    return f"median, IQR, n={n}" if aggregation == "median" else f"mean, n={n}"


def plot_fig03_single_forcing(
    results_list: list[dict],       # List of dictionaries
    baseline_error_list: list[float],     # Single float value
    agents: list[str],
    save: bool = False,
    seed_errors_list: list[list[dict] | None] = None,
    seed_baseline_error_list: list[list[float] | None] = None,
    aggregation: str = AGGREGATION_DEFAULT,
) -> None:
    """5-panel NRMSE-vs-update-step comparison, one panel per single-forcing agent experiment.
    """
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
        seed_errs = seed_errors_list[i] if seed_errors_list else None
        seed_base = seed_baseline_error_list[i] if seed_baseline_error_list else None

        # --- Plotting: optimized-emulator trajectory ---
        if seed_errs is not None:
            _traj = [np.asarray(r["errors"]) for r in seed_errs]
            _lens = sorted({t.shape[0] for t in _traj})
            if len(_lens) > 1:
                raise ValueError(
                    f"panel {i} ({agents[i] if i < len(agents) else '?'}) has seeds at "
                    f"different trajectory lengths {_lens}: this family is only "
                    f"partially migrated, or its regeneration jobs are still running. "
                    f"Wait for the family to finish before plotting - mixing lengths "
                    f"within one panel would average different run stages together."
                )
            stacked = np.stack(_traj, axis=0)
            centre, lo, hi = _aggregate_seeds(stacked, aggregation, mean_band="minmax")
            x_err = np.arange(centre.shape[0])
            ax.loglog(x_err, centre, lw=2, color=cmap(0),
                      label="Optimized emulator (median)")
            ax.fill_between(x_err, lo, hi, color=cmap(0), alpha=0.2, linewidth=0)
        else:
            result = results_list[i]
            errors = jnp.asarray(result["errors"])
            x_err = jnp.arange(errors.shape[0])
            ax.loglog(x_err, errors, label="Optimized emulator (median)", lw=2, color=cmap(0))

        # --- Plotting: baseline lower bound ---
        if seed_base is not None:
            base_arr = np.asarray(seed_base, dtype=float)
            b_centre, b_lo, b_hi = _aggregate_seeds(base_arr, aggregation, mean_band="minmax")
            ax.axhline(float(b_centre), ls="--", c=cm.lipariS(5), lw=1.5,
                       label="Baseline emulator\nerror lower bound (median)")
            if float(b_hi) > float(b_lo):
                ax.axhspan(float(b_lo), float(b_hi), color=cm.lipariS(5), alpha=0.15, linewidth=0)
        else:
            baseline_error = baseline_error_list[i]
            ax.axhline(float(baseline_error), ls="--", c=cm.lipariS(5), lw=1.5, label="Baseline emulator (median)")

        ax.margins(x=0, y=0)
        #ax.xaxis.set_major_locator(plt.MaxNLocator(prune='lower'))

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
            
            # Fetch existing handles and labels to append the custom IQR patch
            handles, labels = ax.get_legend_handles_labels()
            iqr_patch = mpatches.Patch(color='gray', alpha=0.2, label='Interquartile range')
            
            handles.append(iqr_patch)
            labels.append('Interquartile range')
            
            ax.legend(handles=handles, labels=labels, loc="lower left", fontsize=12)

        ax.set_xlim(left=1.1)
        ax.set_ylim([0.01, 1.5])

    fig.supxlabel('Update iteration no.', fontsize=18)

    if save:
      plt.savefig(FIGURES_DIR / 'fig03_single_forcing.pdf')

    return

def plot_fig05_multi_forcing(results: dict, baseline_error: float, save: bool = False,
                               seed_errors: list[dict] = None,
                               seed_baseline_errors: list[float] = None,
                               aggregation: str = AGGREGATION_DEFAULT) -> None:
    """3-panel figure: NRMSE-vs-step (left) plus optimal WMGHG and aerosol emissions trajectories (right, via _plot_agents).
    """
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
    if seed_errors is not None:
        stacked = np.stack([np.asarray(r["errors"]) for r in seed_errors], axis=0)
        centre, lo, hi = _aggregate_seeds(stacked, aggregation, mean_band="minmax")
        x_err = np.arange(centre.shape[0])
        ax.loglog(x_err, centre, lw=2, color=cmap(1),
                  label=f"Optimized emulator (median)")
        ax.fill_between(x_err, lo, hi, color=cmap(1), alpha=0.2, linewidth=0)
    else:
        errors = np.asarray(results["errors"])
        x_err = np.arange(errors.shape[0])
        ax.loglog(x_err, errors, label="Optimized emulator", lw=2, color=cmap(1))

    if seed_baseline_errors is not None:
        b_centre, b_lo, b_hi = _aggregate_seeds(
            np.asarray(seed_baseline_errors, dtype=float), aggregation, mean_band="minmax")
        stat = "median" if aggregation == "median" else "mean"
        ax.axhline(float(b_centre), ls="--", c=cm.lipariS(5), lw=1.5,
                   label=f"Baseline emulator\nerror lower bound ({stat})")
        if float(b_hi) > float(b_lo):
            ax.axhspan(float(b_lo), float(b_hi), color=cm.lipariS(5), alpha=0.15, linewidth=0)
    else:
        ax.axhline(float(baseline_error), ls="--", c=cm.lipariS(5), lw=1.5,
                   label="Baseline emulator\nerror lower bound")

    # Styling
    ax.margins(x=0, y=0.2)
    ax.grid(True, alpha=0.3, which="both", ls="-")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("Emulator error (NRMSE)", fontsize=18)

    # Fetch existing handles and labels to append the custom IQR patch
    handles, labels = ax.get_legend_handles_labels()
    iqr_patch = mpatches.Patch(color='gray', alpha=0.2, label='Interquartile range')
    
    handles.append(iqr_patch)
    labels.append('Interquartile range')
    
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=14)


    ax.set_xlabel('Update iteration no.', fontsize=18)
    ax.set_xlim(left=1.1)

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

    leg = last_ax.legend(
        lines, [l.get_label() for l in lines],
        loc='lower left',
        fancybox=True,      # Rounded corners
        facecolor="white",  # Match text box
        edgecolor="gray",   # Match text box
        framealpha=0.9,      # Match text box
        fontsize=12
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



def plot_fig02_co2_example(
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

        hl_pred = _highlight_indices(sel_pred, middle="second")
        pred_stride = _preds_stride(results, N_all)

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
            if i in hl_pred:
              lab = i * pred_stride
              if i == hl_pred[0]:
                alpha = 1
                c = cm.naviaS(4)
                ls='-.'
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
                                  seed_optimized_results: list[dict] = None,
                                  aggregation: str = AGGREGATION_DEFAULT) -> None:
    """
    Grouped horizontal bar chart of % NRMSE improvement (optimized vs. baseline), grouped by test scenario.
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

    pct_err = None
    if seed_mode:
        n_seeds = len(seed_baseline_results)
        stacked = np.stack([
            _pct_improvement_for(seed_baseline_results[s], seed_optimized_results[s])
            for s in range(n_seeds)
        ], axis=0)
        plot_data, lo, hi = _aggregate_seeds(stacked, aggregation, mean_band="std")
        pct_err = np.stack([np.clip(plot_data - lo, 0, None),
                            np.clip(hi - plot_data, 0, None)], axis=0)
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

    # 3.5 Information-leakage shading (in-objective evaluation sets)
    # ----------------------------------------------------------------
    LEAK_COLOR = "0.55"
    LEAK_ALPHA = 0.35
    for j, train_key in enumerate(train_scenarios):
        trained_on = train_key.removeprefix('Opt. ')
        for i in range(n_rows):
            is_avg = (i == n_test)
            leaked = trained_on == 'All' or (not is_avg and test_scenarios[i] == trained_on)
            if not leaked:
                continue
            offset = (i - n_rows / 2) * bar_width + (bar_width / 2)
            x_center = x_positions[j] + offset
            ax.axvspan(x_center - bar_width / 2, x_center + bar_width / 2,
                       color=LEAK_COLOR, alpha=LEAK_ALPHA, zorder=1, lw=0)

    PARTIAL_LEAK_COLOR = "0.85"
    for j, train_key in enumerate(train_scenarios):
        trained_on = train_key.removeprefix('Opt. ')
        if trained_on == 'All':
            continue
        offset = (n_test - n_rows / 2) * bar_width + (bar_width / 2)
        x_center = x_positions[j] + offset
        ax.axvspan(x_center - bar_width / 2, x_center + bar_width / 2,
                   color=PARTIAL_LEAK_COLOR, alpha=LEAK_ALPHA, zorder=1, lw=0)

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
                      yerr=pct_err[:, i] if pct_err is not None else None,
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
                if val <= -80:
                    bar.set_hatch(hatch_pattern)
                # Negative: Place just above the x-axis (0 line)
                y_pos = 2  # Fixed small offset above 0
                va = 'bottom'
            else:
                # Positive: Place just above the bar
                y_pos = val + 2
                va = 'bottom'
            """
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y_pos,
                    label_text,
                    ha='center',
                    va=va,
                    fontsize=7,        # Slightly smaller font to fit nicely
                    color=color,       # Match the bar color
                    fontweight='bold',
                    zorder=4)
            """
        

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
    if show_legend:
        # Fetch the existing handles (the colored bars) and labels
        handles, labels = ax.get_legend_handles_labels()
        
        # If we plotted error bars, add a custom black line to represent the IQR
        if seed_mode:
            # Create a proxy artist that natively renders as a vertical error bar
            iqr_proxy = ax.errorbar([np.nan], [np.nan], yerr=[1], color='black', 
                                    capsize=3, elinewidth=1.5, capthick=1.5, fmt='none')
            
            handles.append(iqr_proxy)
            labels.append('Interquartile range')

        if not is_standalone:
            # Place legend outside to the right, slightly aligned to top
            if n_plots == 2:
              ax.legend(handles=handles, labels=labels,
                        title='Evaluation Dataset',
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
              ax.legend(handles=handles, labels=labels,
                        title='Evaluation Dataset',
                        loc='lower left',
                        bbox_to_anchor=(0, -0.025),
                        ncol=3,
                        frameon=True,
                        fancybox=True,
                        framealpha=0.8,
                        facecolor='white',
                        edgecolor='#cccccc',
                        fontsize=12)

    if is_standalone and show_legend:
      ax.legend(handles=handles, labels=labels, fontsize=8)

    if long_title:
        ax.set_title(long_title, fontsize=14, pad=10, loc='left')

    # 6. Finalize (Only if running standalone)
    # ----------------------------------------
    if is_standalone:
        ax.set_xlabel('Emulator Configuration', fontsize=14)
        if save:
            plt.savefig(FIGURES_DIR / f'{figname}.pdf', bbox_inches='tight')
        plt.show()

def _plot_vertical_stacked_bars(baseline_results_list: list[dict],
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
                               seed_optimized_results_list: list[list[dict] | None] = None,
                               aggregation: str = AGGREGATION_DEFAULT) -> None:
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
            aggregation=aggregation,
            n_plots=n_plots
        )

        ax.set_ylim([-80, 120])
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
    fig.supylabel(r'Median performance change from baseline emulator [\%]', fontsize=16)

    if save:
        plt.savefig(FIGURES_DIR / f'{figname}.pdf', bbox_inches='tight')

    plt.show()


def plot_fig04_scm_summary(*args, figname: str = 'fig04_scm_summary', **kwargs) -> None:
    """Figure 4: emulator skill change from baseline, CO2-only and multi-agent."""
    return _plot_vertical_stacked_bars(*args, figname=figname, **kwargs)


def plot_SI_extended_results(*args, figname: str = 'SI_extended_results', **kwargs) -> None:
    """Supplement: the same bar layout as Figure 4, per single forcing agent."""
    return _plot_vertical_stacked_bars(*args, figname=figname, **kwargs)


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


# --- Figure 7: MESM (EMIC) zonal-emulator summary ---------------------------

FIG07_SEED_CACHE = "data/SI_results/seed_uncertainty/fig7_seed_spread_MESM.pkl"
FIG07_ICS = ["constant", "sine", "both"]
FIG07_TOP_PANEL_ICS = ["constant", "sine"]
FIG07_N_SEEDS = 50


def _fig07_top_panel_data() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Emissions MESM was driven with, and the area-weighted global mean of the
    zonal temperature it returned, for the constant and sine initial guesses."""
    co2_data = [
        np.loadtxt(f"{DATA_DIR}/MESM/emis_driven/MESM_inputs/opt_all_{ic}.txt",
                   usecols=(2,), skiprows=2)
        for ic in FIG07_TOP_PANEL_ICS
    ]

    lat_weights = np.cos(np.deg2rad(np.linspace(-88, 88, 46)))
    lat_weights = np.maximum(lat_weights, 1e-6)
    lat_weights = lat_weights / np.sum(lat_weights)

    global_mean_temp = []
    for ic in FIG07_TOP_PANEL_ICS:
        with open(f"{DATA_DIR}/MESM/emis_driven/zonal_data_mean/optimized/"
                  f"opt_all_{ic}_mean.pkl", "rb") as f:
            zonal = pickle.load(f)
        global_mean_temp.append(np.average(zonal, weights=lat_weights, axis=1))

    return co2_data, global_mean_temp


def plot_fig07_emic_summary(seed_idx: list[int] | None = None, save: bool = False,
                            figname: str = "fig07_emic_summary"):
    """
    Figure 7: optimized emissions and MESM temperature response for the constant
    and sine initial guesses (a, b), over a bar chart of emulator skill change
    from baseline per scenario (c).

    Args:
        seed_idx: seeds to aggregate. Defaults to the first 50 of the 1000-seed
            sweep - an arbitrary, unselected sample rather than a chosen subset.
        save: write the figure to Figures/ as both PDF and PNG.
        figname: stem of the output file.

    Returns:
        (fig, axd) from plt.subplot_mosaic, so a notebook cell can display it.
    """
    with open(FIG07_SEED_CACHE, "rb") as f:
        seed_cache = pickle.load(f)
    if seed_idx is None:
        seed_idx = list(range(FIG07_N_SEEDS))

    seed_baseline_results = [seed_cache[s]["baseline"] for s in seed_idx]
    seed_optimized_results_list = [[seed_cache[s][f"optimal_{ic}"] for s in seed_idx]
                                   for ic in FIG07_ICS]

    co2_data, global_mean_temp = _fig07_top_panel_data()

    scenario_keys = ['historical', 'H-ext', 'M', 'ML', 'L', 'VLLO-ext', 'VLHO',
                     'H-ext-OS', 'M-ext', 'ML-ext', 'L-ext', 'VLHO-ext',
                     '2xCO2', '1pctCO2', 'AA', 'CT']
    labels = [r'$\it{historical}$', r'$\it{H}$-$\it{ext}$', r'$\it{M}$', r'$\it{ML}$', r'$\it{L}$',
              r'$\it{VLLO}$-$\it{ext}$', r'$\it{VLHO}$', r'$\it{H}$-$\it{ext}$-$\it{OS}$',
              r'$\it{M}$-$\it{ext}$', r'$\it{ML}$-$\it{ext}$', r'$\it{L}$-$\it{ext}$',
              r'$\it{VLHO}$-$\it{ext}$', r'$\it{abrupt}$-$\it{2xCO2}$', r'$\it{1pctCO2}$',
              r'$\it{AA}$', r'$\it{CT}$']
    separator_indices = [6, 11, 13]
    group_labels = ['Priority 1', 'Priority 2', 'DECK', 'CS3']
    legend_labels = ['Const.', 'Sine', 'Both']
    n_total = len(scenario_keys)

    # Row 1 is (a)/(b) side by side and 10% shorter than row 2, the full-width
    # bar chart (c).
    fig, axd = plt.subplot_mosaic(
        [["Top1", "Top2"], ["Bottom", "Bottom"]],
        figsize=(14, 6.76875), constrained_layout=True, height_ratios=[0.9, 1]
    )
    axes_co2 = [axd["Top1"], axd["Top2"]]
    ax_bar = axd["Bottom"]

    t_min = np.min(global_mean_temp)
    t_max = np.max(global_mean_temp)
    panel_titles = [
        r"(a) Optimized emissions and resulting $\overline{\Delta T}(t)$ (const. initial guess)",
        r"(b) Optimized emissions and resulting $\overline{\Delta T}(t)$ (sine initial guess)",
    ]

    # (a) and (b) share both y-scales, so only (a)'s left axis and (b)'s right
    # axis carry labels.
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
            plt.setp(ax_temp.get_yticklabels(), visible=False)
        else:
            ax_temp.set_ylabel(r"$\overline{\Delta T}(t)$ [$^\circ$C]", fontsize=12, rotation=270,
                               labelpad=13, c=cm.actonS(4))
            plt.setp(axes_co2[i].get_yticklabels(), visible=False)

        axes_co2[i].text(
            0.02, 0.94, panel_titles[i], transform=axes_co2[i].transAxes,
            ha="left", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
        )
        if i == 0:
            lines_1, labels_1 = axes_co2[i].get_legend_handles_labels()
            lines_2, labels_2 = ax_temp.get_legend_handles_labels()
            ax_temp.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=True, loc='lower left',
                           fancybox=True, framealpha=0.8, facecolor='white',
                           edgecolor='#cccccc', fontsize=9)

    n_opts = len(seed_optimized_results_list)
    bar_width = 0.8 / n_opts
    colors = [cm.osloS(i + 2) for i in range(n_opts)]
    bar_ylim = (-70, 90)
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

    for opt_idx in range(n_opts):
        seed_opt = seed_optimized_results_list[opt_idx]
        per_seed = np.array([_pct_change_row(b, o) for b, o in zip(seed_baseline_results, seed_opt)])
        diff_values = np.median(per_seed, axis=0)
        q1 = np.percentile(per_seed, 25, axis=0)
        q3 = np.percentile(per_seed, 75, axis=0)
        diff_err = np.vstack([diff_values - q1, q3 - diff_values])

        offset = (opt_idx - n_opts / 2) * bar_width + (bar_width / 2)
        bars = ax_bar.bar(x_positions + offset, diff_values, yerr=diff_err,
                          error_kw=dict(capsize=2, elinewidth=1),
                          label=legend_labels[opt_idx], width=bar_width,
                          color=colors[opt_idx], edgecolor='black', linewidth=0.7, zorder=3)
        # Bars past the y-limit are hatched rather than clipped silently.
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

    for idx in separator_indices:
        ax_bar.axvline(idx + 0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.6, zorder=0)

    boundaries = [-0.5] + [idx + 0.5 for idx in separator_indices] + [n_total - 0.5]
    for i, label_text in enumerate(group_labels):
        center_x = (boundaries[i] + boundaries[i + 1]) / 2
        ax_bar.text(center_x, -0.40, label_text, transform=ax_bar.get_xaxis_transform(),
                    rotation=0, ha='center', va='top', fontsize=12, fontweight='bold',
                    color='#333333')

    ax_bar.text(0.015, 0.98, r"(c) MESM emulator performance summary", transform=ax_bar.transAxes,
                ha="left", va="top", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

    legend_handles = [
        mpatches.Patch(facecolor=colors[i], edgecolor='black', linewidth=0.7, label=legend_labels[i])
        for i in range(n_opts)
    ]
    legend_handles.append(
        Line2D([0, 1], [0, 0], color='black', linewidth=1, marker='|', markersize=10,
               markeredgewidth=1, label='Interquartile range')
    )
    ax_bar.legend(handles=legend_handles, title='Emulator initial guess', loc='upper right',
                  bbox_to_anchor=(1.012, 1.035), numpoints=2, title_fontsize=11,
                  frameon=True, fancybox=True, framealpha=0.8, facecolor='white',
                  edgecolor='#cccccc', fontsize=10)

    ax_bar.set_xlabel('Scenario', fontsize=13, labelpad=32)
    ax_bar.set_ylabel(r'Median performance change' + '\n' + r'from baseline emulator [\%]',
                      fontsize=12)

    if save:
        fig.savefig(FIGURES_DIR / f"{figname}.pdf", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / f"{figname}.png", bbox_inches="tight", dpi=200)

    return fig, axd

def _fig6_seed_stack(seed_cache, getter, scen_plot, lo_idx, hi_idx):
    """Stack one trajectory across seeds -> (n_seeds, n_times).

    `getter` pulls the per-seed dict down to the {eval_set: {scen: array}} level,
    so baseline and per-train-scenario curves share one code path.
    """
    rows = []
    for seed in sorted(seed_cache):
        d = getter(seed_cache[seed])
        if d is None:
            continue
        rows.append(np.asarray(d['All'][scen_plot][lo_idx:hi_idx]))
    return np.stack(rows) if rows else None


def _fig6_ood_seed_stack(ood_seed_traj, tag, label):
    """(n_seeds, n_times) OOD trajectory stack for one (scenario, emulator label), or None."""
    if not ood_seed_traj:
        return None
    return ood_seed_traj.get(tag, {}).get(label)


def plot_ood_r2_forest(
    ax,
    r2_table: list[dict],
    color_map: dict,
    scenarios: list[str],
    baseline_color=None,
    display_label: dict | None = None,
    dmin: float = -1.0,
    dmax: float = 1.0,
    title: str | None = None,
    scenario_shade: dict[str, str] | None = None,
    text_ax=None,
    group_labels: list[tuple[str, int]] | None = None,
    legend_valign_group: str | None = None,
    within_group_gap: float = 0.8,
    between_group_gap: float = 1.4,
) -> None:
  """Median R^2 + IQR forest plot across scenarios x emulator configs.
  """
  baseline_color = baseline_color if baseline_color is not None else cm.lipariS(5)
  display_label = display_label or {}
  scenario_shade = scenario_shade or {}
  SHADE_COLORS = {"dark": "0.55", "light": "0.85"}
  SHADE_ALPHA = 0.35
  emulators = ["Baseline Em."] + list(color_map.keys())
  series_color = {"Baseline Em.": baseline_color, **color_map}

  stats = {}
  for row in r2_table:
      stats[(row["scenario"], row["emulator"])] = (
          float(row["r2_median"]), float(row["r2_p25"]), float(row["r2_p75"]))

  n_emu = len(emulators)
  dodge = np.linspace(-0.32, 0.32, n_emu)

  def _clip(v):
      return max(dmin, min(dmax, v))

  if group_labels:
      group_of_idx = []
      for name, n_rows in group_labels:
          group_of_idx.extend([name] * n_rows)
  else:
      group_of_idx = [None] * len(scenarios)

  cum = [0.0] * len(scenarios)
  for i in range(1, len(scenarios)):
      if not group_labels:
          gap = 1.0  # no group structure given - old uniform spacing
      else:
          same_group = group_of_idx[i] == group_of_idx[i - 1]
          gap = within_group_gap if same_group else between_group_gap
      cum[i] = cum[i - 1] + gap
  row_y_vals = [cum[-1] - c for c in cum]  # index 0 (list order) is the topmost row

  row_half = (within_group_gap if group_labels else 1.0) / 2

  for i, scen in enumerate(scenarios):
      row_y = row_y_vals[i]
      shade = scenario_shade.get(scen)
      if shade is not None:
          if not group_labels:
              top_ext = bottom_ext = row_half
          else:
              top_ext = (within_group_gap / 2 if (i > 0 and group_of_idx[i] == group_of_idx[i - 1])
                         else between_group_gap / 2)
              bottom_ext = (within_group_gap / 2 if (i < len(scenarios) - 1 and group_of_idx[i] == group_of_idx[i + 1])
                            else between_group_gap / 2)
          ax.axhspan(row_y - bottom_ext, row_y + top_ext, color=SHADE_COLORS[shade], alpha=SHADE_ALPHA, zorder=0)
      for j, emu in enumerate(emulators):
          st = stats.get((scen, emu))
          if st is None:
              continue
          med, lo, hi = st
          y = row_y + dodge[j]
          color = series_color[emu]
          if med >= dmin:
              ax.plot([_clip(lo), _clip(hi)], [y, y], color=color, lw=2.5,
                      alpha=0.55, solid_capstyle="round", zorder=2)
              ax.plot(_clip(med), y, "o", color=color, ms=5.5, zorder=3,
                      markeredgecolor="white", markeredgewidth=0.5)
          else:
              ax.plot(_clip(med), y, "o", color=color, ms=5.5, zorder=3,
                      markeredgecolor="white", markeredgewidth=0.5)
              ax.text(_clip(med) + 0.03, y, f"{med:.1f}", fontsize=7,
                      va="center", ha="left", color=color)

  if group_labels:
      for i in range(1, len(scenarios)):
          y_div = (row_y_vals[i - 1] + row_y_vals[i]) / 2
          same_group = group_of_idx[i] == group_of_idx[i - 1]
          if same_group:
              ax.axhline(y_div, color="0.82", lw=0.7, alpha=0.8, zorder=0.5)
          else:
              ax.axhline(y_div, color="0.35", lw=1.2, alpha=0.9, zorder=0.5)

  scenario_display = {"M_GHG": "M-GHG", "M_AER": "M-aer",
                       "H-ext-Maer": "H-ext-M-aer", "H-ext-VLaer": "H-ext-VL-aer",
                       "H-ext-Laer": "H-ext-L-aer"}
  ax.set_yticks(row_y_vals)
  ax.set_yticklabels([r"\textit{" + scenario_display.get(s, s).replace("_", r"\_") + "}"
                       for s in scenarios])
  ax.set_ylim(row_y_vals[-1] - row_half, row_y_vals[0] + row_half)
  ax.set_xlim(dmin - 0.05, dmax + 0.05)
  ax.set_xticks([dmin, dmin / 2, 0, dmax / 2, dmax])
  ax.set_xticklabels([rf"$\leq${dmin:g}"] + [f"{t:g}" for t in [dmin / 2, 0, dmax / 2, dmax]])
  ax.axvline(0, color="0.6", lw=1, ls="--", zorder=1)
  ax.set_xlabel(r"Median $R^2$")
  ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)

  if text_ax is not None and group_labels:
      text_ax.sharey(ax)
      text_ax.axis("off")
      idx = 0  # index into scenarios/row_y_vals of this group's first row
      for name, n_rows in group_labels:
          center_y = (row_y_vals[idx] + row_y_vals[idx + n_rows - 1]) / 2
          text_ax.text(0.5, center_y, name, transform=text_ax.get_yaxis_transform(),
                       rotation=270, ha="center", va="center",
                       fontsize=11, fontweight="bold", color="#333333")
          idx += n_rows

  handles = [Line2D([0], [0], marker="o", color=series_color[emu], lw=2.5,
                    label=display_label.get(emu, emu), markersize=5.5)
             for emu in emulators]

  handles.append(Line2D([0], [0], color="0.5", lw=2.5, alpha=0.55,
                        solid_capstyle="round", label="Interquartile range"))

  legend_x, legend_y_below_title = 0.03, 0.90
  if title is not None:
      title_artist = ax.text(
              0.03, 0.985, title, transform=ax.transAxes,
              ha="left", va="top", fontsize=14, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
            )

      ax.figure.canvas.draw()
      renderer = ax.figure.canvas.get_renderer()
      bbox_disp = title_artist.get_bbox_patch().get_window_extent(renderer=renderer)
      bbox_axes = bbox_disp.transformed(ax.transAxes.inverted())
      legend_x, legend_y_below_title = bbox_axes.x0, bbox_axes.y0 - 0.015

  legend_loc, legend_anchor = "upper left", (legend_x, legend_y_below_title)
  if group_labels and legend_valign_group is not None:

      idx = 0
      center_y = None
      for name, n_rows in group_labels:
          if name == legend_valign_group:
              center_y = (row_y_vals[idx] + row_y_vals[idx + n_rows - 1]) / 2
              break
          idx += n_rows
      if center_y is not None:
          y0, y1 = ax.get_ylim()
          legend_loc, legend_anchor = "center left", (legend_x, (center_y - y0) / (y1 - y0))

  ax.legend(handles=handles, loc=legend_loc, bbox_to_anchor=legend_anchor,
            fontsize=11, title="Emulator configuration",
            title_fontsize=11, framealpha=0.85)


def plot_fig06_individual_effects(
    y_true_ind_effects: dict,
    y_hat_baseline: dict,
    y_hat_ind_effects: dict,
    train_scenarios_ind_effects: list[str],
    ppt: bool = True,
    save: bool = False,
    figname: str = 'individual_effects_ppt4',
    seed_cache: dict | None = None,
    aggregation: str = AGGREGATION_DEFAULT,
    ood_scenarios: dict | None = None,
    ood_seed_traj: dict | None = None,
    r2_table: list[dict] | None = None,
) -> None:
  """Plot the multi-agent "individual effects" summary figure
  """
  i_ppt = 0
  layout = [
      ["YLabel", "Left1", "Right", "Text"],
      ["YLabel", "Left2", "Right", "Text"],
      ["YLabel", "Left3", "Right", "Text"],
      ["YLabel", ".",      "Right", "Text"],  # spacer: harmonized-group break (c -> d)
      ["YLabel", "Left4", "Right", "Text"],
      ["YLabel", "Left5", "Right", "Text"],
      ["YLabel", ".",      "Right", "Text"],  # spacer: harmonized-group break (e -> f)
      ["YLabel", "Left6", "Right", "Text"],
  ]


  color_map = {'Opt. Tier 1':cm.bamakoS(15), 'Opt. DAMIP':cm.hawaiiS(3), 'Opt. GeoMIP': cm.budaS(3), 'Opt. All':cm.osloS(2)}
  baseline_color = cm.lipariS(5)
  display_label = {'Opt. Tier 1': 'Opt. Prio. 1', 'Baseline Em.': 'Baseline Em.',
                    'Opt. DAMIP': 'Opt. DAMIP', 'Opt. GeoMIP': 'Opt. GeoMIP', 'Opt. All': 'Opt. All'}

  panel_kind = {"Left1": "in_obj", "Left2": "in_obj", "Left3": "in_obj",
                "Left4": "ood", "Left5": "ood", "Left6": "ood"}
  panel_scenario = {"Left1": "M_GHG", "Left2": "M_AER", "Left3": "G6sulfur",
                     "Left4": "H-ext-VLaer", "Left5": "ssp534-over", "Left6": "esm-bell-2000PgC"}

  ood_xlim = {"H-ext-VLaer": (2024, 2100), "ssp534-over": (2024, 2100),
              "esm-bell-2000PgC": (0, 200)}
  ood_relative_x = {"esm-bell-2000PgC"}
  panel_labels = {
      "Left1": r"(a) Medium GHG emissions, greenhouse gases only (DAMIP: $\it{M}$-$\it{GHG}$)",
      "Left2": r"(b) Medium GHG emissions, aerosols only (DAMIP: $\it{M}$-$\it{aer}$)",
      "Left3": r"(c) High GHG emissions with sulfur injection (GeoMIP: $\it{G6sulfur}$)",
      "Left4": r"(d) High GHG emissions, very-low aerosols (RAMIP analog: $\it{H}$-$\it{ext}$-$\it{VL}$-$\it{aer}$)",
      "Left5": r"(e) Overshoot scenario (ScenarioMIP-CMIP6: $\it{ssp534}$-$\it{over}$)",
      "Left6": r"(f) piControl-branched CO$_2$-only bell-shaped emissions (ZECMIP: $\it{esm}$-$\it{bell}$-$\it{2000PgC}$)",
  }
  plot_len = len(y_true_ind_effects["All"]["M_GHG"])
  x_vals = np.arange(1850, 2151)

  if ppt:
      figsize = (16.81, 7)
  else:
      figsize = (17.8, 12.8)

  fig, ax_dict = plt.subplot_mosaic(
      layout,
      figsize=figsize,
      constrained_layout=True,
      gridspec_kw={
          "width_ratios": [0.35, 4.75, 3, 0.25],
          "height_ratios": [1, 1, 1, 0.03, 1, 1, 0.03, 1],
      }
  )

  fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.0, wspace=0.0, hspace=0.0)
  ax_dict["Text"].axis("off")
  ax_dict["YLabel"].axis("off")

  has_ood = bool(ood_scenarios and ood_seed_traj)
  if not has_ood:
      print("plot_fig06_individual_effects: no OOD data supplied - Left4-6 left blank. "
            "Pass ood_scenarios/ood_seed_traj from utils_inverse.load_fig6_ood_data.")

  y_min, y_max = np.inf, -np.inf
  def _track(*arrays):
      nonlocal y_min, y_max
      for arr in arrays:
          arr = np.asarray(arr)
          if arr.size == 0:
              continue
          y_min = min(y_min, float(np.nanmin(arr)))
          y_max = max(y_max, float(np.nanmax(arr)))

  for ax_label, scen_plot in panel_scenario.items():
      ax_plot = ax_dict[ax_label]
      kind = panel_kind[ax_label]

      if kind == "ood" and not has_ood:
          continue

      if kind == "in_obj":
          truth = y_true_ind_effects['All'][scen_plot][100:plot_len]
          ax_plot.plot(x_vals, truth, label='SCM-projected', c='black', ls='--', lw=2, alpha = 0.8)
          _track(truth)

          if seed_cache:
              b_stack = _fig6_seed_stack(
                  seed_cache, lambda r: r['y_hat_baseline'], scen_plot, 100, plot_len)
              if b_stack is not None:
                  b_c, b_lo, b_hi = _aggregate_seeds(b_stack, aggregation, mean_band="minmax")
                  ax_plot.fill_between(x_vals, b_lo, b_hi, color=cm.lipariS(5), alpha=0.18,
                                       lw=0, zorder=1)
                  ax_plot.plot(x_vals, b_c, label='Baseline Em.', lw=2, ls="-", c=cm.lipariS(5))
                  _track(b_lo, b_hi)
              else:
                  base = y_hat_baseline['All'][scen_plot][100:plot_len]
                  ax_plot.plot(x_vals, base, label='Baseline Em.', lw=2, ls="-", c=cm.lipariS(5))
                  _track(base)
          else:
              base = y_hat_baseline['All'][scen_plot][100:plot_len]
              ax_plot.plot(x_vals, base, label='Baseline Em.', lw=2, ls="-", c=cm.lipariS(5))
              _track(base)

          for train in reversed(train_scenarios_ind_effects):
              train_label = train
              if train in ['Opt. Tier 2','Opt. DECK', 'Opt. CS3']:
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
              o_stack = (_fig6_seed_stack(seed_cache, lambda r: r['y_hat'].get(train),
                                          scen_plot, 100, plot_len)
                         if seed_cache else None)
              if o_stack is not None:
                  o_c, o_lo, o_hi = _aggregate_seeds(o_stack, aggregation, mean_band="minmax")
                  ax_plot.fill_between(x_vals, o_lo, o_hi, color=color_map[train],
                                       alpha=0.18, lw=0, zorder=zorder)
                  ax_plot.plot(x_vals, o_c, alpha=alpha, label=train_label, zorder=zorder + 1,
                               lw=2, ls=ls, color=color_map[train])
                  _track(o_lo, o_hi)
              else:
                  opt = y_hat_ind_effects[train]['All'][scen_plot][100:plot_len]
                  ax_plot.plot(x_vals, opt, alpha=alpha, label=train_label, zorder=zorder, lw=2, ls=ls, color=color_map[train])
                  _track(opt)

          ax_plot.set_xlim([1850, 2150])
          if ax_label in ("Left1", "Left2"):
              ax_plot.sharex(ax_dict["Left3"])
              ax_plot.tick_params(axis="x", bottom=False, labelbottom=False)

      else:  # kind == "ood"
          tag = scen_plot
          scen = ood_scenarios.get(tag)
          if scen is None:
              continue
          years = scen["years"]
          x_plot = (years - years.min()) if tag in ood_relative_x else years
          xlim = ood_xlim[tag]
          vis = (x_plot >= xlim[0]) & (x_plot <= xlim[1])

          ax_plot.plot(x_plot, scen["y_scm"], label='SCM-projected', c='black',
                       ls='--', lw=2, alpha=0.8)
          _track(scen["y_scm"][vis])

          b_stack = _fig6_ood_seed_stack(ood_seed_traj, tag, "Baseline Em.")
          if b_stack is not None:
              b_c, b_lo, b_hi = _aggregate_seeds(b_stack, aggregation, mean_band="minmax")
              ax_plot.fill_between(x_plot, b_lo, b_hi, color=baseline_color, alpha=0.18, lw=0, zorder=1)
              ax_plot.plot(x_plot, b_c, label='Baseline Em.', lw=2, ls="-", c=baseline_color)
              _track(b_lo[vis], b_hi[vis])

          for train in reversed(train_scenarios_ind_effects):
              if train in ['Opt. Tier 2', 'Opt. DECK', 'Opt. CS3']:
                  continue
              train_label = display_label.get(train, train)
              alpha, zorder, ls = (1, 10, '-') if train == 'Opt. All' else (0.85, 5, (0, (5, 4)))
              o_stack = _fig6_ood_seed_stack(ood_seed_traj, tag, train)
              if o_stack is None:
                  continue
              o_c, o_lo, o_hi = _aggregate_seeds(o_stack, aggregation, mean_band="minmax")
              ax_plot.fill_between(x_plot, o_lo, o_hi, color=color_map[train], alpha=0.18, lw=0, zorder=zorder)
              ax_plot.plot(x_plot, o_c, alpha=alpha, label=train_label, zorder=zorder + 1,
                           lw=2, ls=ls, color=color_map[train])
              _track(o_lo[vis], o_hi[vis])

          ax_plot.set_xlim(xlim)
          if ax_label == "Left4":
              ax_plot.tick_params(axis="x", bottom=False, labelbottom=False)

      ax_plot.grid(linestyle='--', alpha=0.3, zorder=0)
      ax_plot.text(
              0.015, 0.915, panel_labels[ax_label], transform=ax_plot.transAxes,
              ha="left", va="top", fontsize=11, fontweight="bold", zorder=20,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
            )

  if np.isfinite(y_min) and np.isfinite(y_max):
      data_range = y_max - y_min
      bottom_pad = 0.04 * data_range
      top_pad = 0.16 * data_range
      shared_ylim = [y_min - bottom_pad, y_max + top_pad]
      for ax_label in panel_scenario:
          ax_dict[ax_label].set_ylim(shared_ylim)


  ax_dict["Left3"].set_xlabel("Year")
  ax_dict["Left5"].set_xlabel("Year")
  ax_dict["Left6"].set_xlabel("Years since piControl branch")

  handles, legend_labels = ax_dict["Left1"].get_legend_handles_labels()
  ax_dict["Left2"].legend(handles, legend_labels, ncol=2,
                    loc='upper right',
                    frameon=True,
                    fancybox=True,
                    framealpha=0.8,
                    facecolor='white',
                    edgecolor='#cccccc',
                    fontsize=11,
                    handlelength=1.5,
                    handletextpad=0.5,
                    columnspacing=1.0)

  ax_dict["YLabel"].text(0.3, 0.5, r'Temperature anomaly [$^\circ$C]',
                          transform=ax_dict["YLabel"].transAxes,
                          rotation=90, ha='center', va='center', fontsize=16)

  # -- Right column: median R^2 + IQR forest plot across all scenarios/configs --
  if r2_table:
      scenario_groups = [
          ("DAMIP",                 ["M_GHG", "M_AER"],                            "dark"),
          ("GeoMIP",                ["G6sulfur"],                                  "dark"),
          ("RAMIP analogs (CMIP7)", ["H-ext-Maer", "H-ext-VLaer", "H-ext-Laer"], "light"),
          ("CMIP5",                 ["rcp45", "rcp85"],                            None),
          ("CMIP6",                 ["ssp370-lowNTCF", "ssp370-126aer", "ssp534-over"], None),
          ("ZECMIP",                ["esm-bell-1000PgC", "esm-bell-2000PgC"],      None),
      ]
      ood_forest_scenarios = [s for _, scens, _ in scenario_groups for s in scens]
      scenario_shade = {s: shade for _, scens, shade in scenario_groups if shade is not None for s in scens}
      plot_ood_r2_forest(
          ax_dict['Right'], r2_table, color_map,
          scenarios=ood_forest_scenarios,
          scenario_shade=scenario_shade,
          baseline_color=baseline_color, display_label=display_label,
          title='(g) Emulator skill summary',
          text_ax=ax_dict['Text'],
          group_labels=[(name, len(scens)) for name, scens, _ in scenario_groups],
          legend_valign_group="CMIP5",
      )
  else:
      print("plot_fig06_individual_effects: no r2_table supplied - right column left blank. "
            "Pass r2_table from utils_inverse.load_fig6_ood_data.")

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
    base_params = utils_FaIR_JAX.params_for_mode(mode)

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

        # Plot calibrated (theta0 / params_opt)
        ax.plot(years, preds_opt[i, :actual_len], color='tab:red', linestyle='--',
                label=f'Calibrated ({mode})', lw=2)

        # Plot mode's own uncalibrated default params (params=None)
        ax.plot(years, preds_default[i, :actual_len], color='tab:blue', linestyle=':',
                label=f'Default ({mode})', lw=2)

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
            ax_pred.plot(yhat, alpha=alpha, label=f"Step {i * _preds_stride(results, N_all)}")
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


def plot_comparison_results(
    result_paths: list[str] | None = None,
    column_titles: list[str] = None,
    baseline_errors: list[float] | None = None,
    active_agents: tuple | None = None,
    agent_units: dict | None = None,
    max_lines: int = 11,
    save_path: str | None = None,
    seed_result_paths: list[list[str]] | None = None,
    seed_baseline_errors: list[list[float]] | None = None,
    representative_seed: int = 0,
    aggregation: str = AGGREGATION_DEFAULT,
) -> None:
    """
    Comparison plot for multiple optimize_emissions_inverse checkpoints, used
    by pipeline/08_plotting/SI_plots.ipynb's sensitivity-sweep figures
    (initial condition / architecture / feature window).
    Columns = datasets (defined by result_paths). Rows = 1 (NRMSE) + N (agents).

    Args:
        result_paths: File paths to pickle files containing 'errors'/'U_traj'
            (single-run mode). Ignored for a column where the corresponding
            entry of `seed_result_paths` is given.
        column_titles: Text box label for each column (top left).
        baseline_errors: Baseline error value (dashed line) per column,
            single-run mode.
        seed_result_paths: Optional, one entry per column - a list of
            per-seed checkpoint paths (same experiment/condition, varying
            seed). When given for a column, the NRMSE row plots the
            median trajectory with a shaded IQR band (matching Figure 3's
            plot_fig03_single_forcing convention) instead of a single
            line, using the penalty-corrected NRMSE
            (utils_inverse.load_inverse_ckpt_nrmse_only), not a checkpoint's
            raw 'errors'. Leaving this None (the default) is fully
            backward-compatible - reproduces the exact prior single-run
            output.
        seed_baseline_errors: Optional, one entry per column - a list of
            per-seed baseline NRMSE floats, paired with `seed_result_paths`.
        representative_seed: In seed mode, which seed's emissions trajectory
            drives the bottom rows (rows 1..N). These stay single-seed even
            in seed mode - overlaying 50 realized trajectories per column
            would be unreadable, matching this repo's existing precedent
            that illustrative trajectory panels are not seed-aggregated
            (e.g. Figure 2's H-ext visualization).
        aggregation: 'median' (default, IQR band) or 'mean' (min-max band),
            only used when `seed_result_paths` is given for a column.
    """
    n_cols = len(seed_result_paths) if seed_result_paths is not None else len(result_paths)
    if column_titles is None or len(column_titles) != n_cols:
        raise ValueError("column_titles must be given and match the number of columns.")

    # --- 1. Load Data ---
    # Per column: the single-run dict used for rows 1..N (emissions
    # trajectories) always, plus (in seed mode) the per-seed NRMSE
    # trajectories/baselines used for row 0 instead of the single-run values.
    loaded_results = []
    seed_nrmse_by_col = [None] * n_cols
    seed_baseline_by_col = [None] * n_cols
    for col_idx in range(n_cols):
        if seed_result_paths is not None and seed_result_paths[col_idx] is not None:
            paths = seed_result_paths[col_idx]
            rep_path = paths[representative_seed]
            with open(rep_path, 'rb') as f:
                loaded_results.append(pickle.load(f))
            seed_nrmse_by_col[col_idx] = [
                utils_inverse.load_inverse_ckpt_nrmse_only(p)["errors"] for p in paths
            ]
            seed_baseline_by_col[col_idx] = seed_baseline_errors[col_idx]
        else:
            with open(result_paths[col_idx], 'rb') as f:
                loaded_results.append(pickle.load(f))

    if seed_result_paths is None and baseline_errors is not None and len(baseline_errors) != n_cols:
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

        if seed_nrmse_by_col[col_idx] is not None:
            _lens = sorted({int(t.shape[0]) for t in seed_nrmse_by_col[col_idx]})
            if len(_lens) > 1:
                raise ValueError(
                    f"column {col_idx} ({column_titles[col_idx]!r}) has seeds at different "
                    f"trajectory lengths {_lens}: this sweep is only partially migrated, or "
                    f"its regeneration jobs are still running."
                )
            stacked = np.stack([np.asarray(t) for t in seed_nrmse_by_col[col_idx]], axis=0)
            centre, lo, hi = _aggregate_seeds(stacked, aggregation, mean_band="minmax")
            x_err = np.arange(centre.shape[0])
            stat = "median" if aggregation == "median" else "mean"
            ax_err.loglog(x_err, centre, c=cm.batlowWS(1), lw=2, label=f"Optimized emulator ({stat})")
            ax_err.fill_between(x_err, lo, hi, color=cm.batlowWS(1), alpha=0.2, linewidth=0)

            b_centre, b_lo, b_hi = _aggregate_seeds(
                np.asarray(seed_baseline_by_col[col_idx], dtype=float), aggregation, mean_band="minmax")
            ax_err.axhline(float(b_centre), ls="--", c=cm.lipariS(5), lw=1.5,
                           label=f"Baseline emulator\nerror lower bound ({stat})")
            if float(b_hi) > float(b_lo):
                ax_err.axhspan(float(b_lo), float(b_hi), color=cm.lipariS(5), alpha=0.15, linewidth=0)
            n_update_steps = int(x_err[-1])

            if col_idx == 0:
                handles, labels = ax_err.get_legend_handles_labels()
                iqr_patch = mpatches.Patch(color='gray', alpha=0.2, label='Interquartile range')
                handles.append(iqr_patch)
                labels.append('Interquartile range')
                ax_err.legend(handles=handles, labels=labels, loc="upper right", fontsize=9)
        else:
            x_err = jnp.arange(errors.shape[0])
            ax_err.loglog(x_err, errors, c=cm.batlowWS(1), label="Optimized emulator")
            ax_err.axhline(float(baseline_errors[col_idx]), ls="--", c=cm.lipariS(5), lw=1.5, label="Baseline emulator\nerror lower bound")
            if col_idx == 0:
                ax_err.legend(loc="upper right", fontsize=9)

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

        # --- Rows 1..N: Emissions ---
        n_total = len(U_traj)
        if n_total <= max_lines:
            sel_steps = np.arange(n_total, dtype=int)
        else:
            sel_steps = np.unique(np.linspace(0, n_total - 1, num=max_lines, dtype=int))

        # Highlighted/labelled steps, derived from the selection rather than
        # hardcoded, so the final iterate is always one of them.
        hl_steps = _highlight_indices(sel_steps)

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
                    if i in hl_steps:
                        if i == hl_steps[0]:
                            c = cm.lipariS(5)
                            alpha = 1
                        else:
                            c = cm.batlowWS(1)
                        # i indexes U_traj directly, so it IS the outer
                        # iteration number and needs no stride conversion.
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

# ==================================================================
# Figure 6 OOD extension
# ==================================================================

OOD_GROUP_COLORS = {
    "cmip7_ramip":     "#011959",  # navy
    "co2_only":        "#226061",  # teal
    "other_generation": "#828231", # olive
    "cmip6_aerosol":   "#dd954d",  # orange
}
OOD_GROUP_LABELS = {
    "cmip7_ramip": "CMIP7 RAMIP analogue",
    "co2_only": r"CO$_2$-only idealized",
    "other_generation": "Other scenario generation",
    "cmip6_aerosol": "CMIP6 aerosol perturbation",
}
OOD_LINESTYLES = ["-", "--", ":"]

_OOD_PANEL_SPECS = [
    ("CO2", r"CO$_2$ [GtCO$_2$/yr]"),
    ("CH4", r"CH$_4$ [MtCH$_4$/yr]"),
    ("N2O", r"N$_2$O [MtN$_2$O/yr]"),
    ("Sulfur", r"Sulfur [MtSO$_2$/yr]"),
    ("BC", "BC [MtBC/yr]"),
    ("GMST", r"SCM $\Delta T$ [$^\circ$C]"),
]


def _ood_scenario_styles(scen):
    """Stable per-scenario style: colour from its group, linestyle from its index
    within that group. Shared by every the out-of-objective build figure so a scenario keeps one
    identity across them."""
    style, per_group = {}, {}
    for tag, d in scen.items():
        g = d["group"]
        i = per_group.get(g, 0)
        per_group[g] = i + 1
        style[tag] = dict(color=OOD_GROUP_COLORS[g], ls=OOD_LINESTYLES[i % len(OOD_LINESTYLES)])
    return style

def plot_scm_mesm_fidelity_grid(panels, ncols=3, save=None, figsize=None):
    """One panel per scenario: the recalibrated SCM
    (mode='MESM') global-mean-temperature output against real MESM
    ground truth (data/MESM/emis_driven/zonal_data_mean/, area-weighted to a
    global mean). Companion figure to pipeline/07_results/07e_scm_mesm_fidelity.py.

    `panels` - list of dicts, one per scenario, plotted in the given order:
        {
          "set_name": str,          # "Tier 1" / "Tier 2" / "DECK" / "CS3"
          "scenario": str,
          "x": array (T,),          # calendar year, or simulation year for
                                     # idealized (DECK/CS3) experiments - see
                                     # xlabel
          "xlabel": str,
          "scm": array (T,),
          "mesm": array (T,),
          "nrmse": float,
          "r2": float,
        }
    """
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (6.0 * ncols, 3.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    c_mesm, c_scm = "black", cm.osloS(2)

    y_min = min(min(np.nanmin(p["mesm"]), np.nanmin(p["scm"])) for p in panels)
    y_max = max(max(np.nanmax(p["mesm"]), np.nanmax(p["scm"])) for p in panels)
    data_range = y_max - y_min
    shared_ylim = [y_min - 0.05 * data_range, y_max + 0.18 * data_range]

    handles = [Line2D([0], [0], color=c_mesm, lw=2, label="MESM global average"),
               Line2D([0], [0], color=c_scm, ls="--", lw=2, label="MESM-calibrated SCM")]

    def _italic_scenario(name):
        return "-".join(rf"$\it{{{part}}}$" for part in name.split("-"))

    for i, (ax, p) in enumerate(zip(axes, panels)):
        ax.plot(p["x"], p["mesm"], color=c_mesm, lw=2, alpha=0.85, zorder=5,
                label="MESM global average")
        ax.plot(p["x"], p["scm"], color=c_scm, ls="--", lw=2, alpha=0.9, zorder=6,
                label="MESM-calibrated SCM")
        ax.text(0.03, 0.95, f"({chr(97 + i)}) {_italic_scenario(p['scenario'])}", transform=ax.transAxes,
                ha="left", va="top", fontsize=18, fontweight="bold", zorder=20,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.9))
        ax.set_xlabel(p["xlabel"], fontsize=16)
        if i % ncols == 0:
            ax.set_ylabel(r"$\Delta T$ [$^\circ$C]", fontsize=20)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_xlim(p["x"][0], p["x"][-1])
        ax.set_ylim(shared_ylim)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

        if i == 0:
            # loc="upper right" (with no bbox_to_anchor) places the legend
            # inside this panel's own top-right corner; swap loc or add
            # bbox_to_anchor=(x, y) in axes-fraction coords to move it.
            leg = ax.legend(handles=handles, loc="upper right", frameon=True, fancybox=True,
                             facecolor="white", edgecolor="gray", framealpha=0.9,
                             fontsize=17, handlelength=1.8)
            leg.set_zorder(25)

    for extra in axes[n:]:
        extra.axis("off")

    if save is not None:
        fig.savefig(FIGURES_DIR / f"{save}.pdf", bbox_inches="tight", transparent=True)
        fig.savefig(FIGURES_DIR / f"{save}.png", bbox_inches="tight", dpi=200, transparent=True)
    return fig, axes


# ==================================================================
# SI: representative-seed emissions comparison (CO2-only + multi-forcing)
# ==================================================================

def plot_seed_emissions_comparison(
    co2_emissions: dict,
    multi_emissions: dict,
    co2_seed_info: dict | None = None,
    multi_seed_info: dict | None = None,
    agent_units: dict | None = None,
    agent_cmaps: dict | None = None,
    shade_fracs: dict | None = None,
    title_suffix: str = '',
    panel_width: float = 14.0,
    panel_height: float = 2.0,
    figsize: tuple | None = None,
    save: bool = False,
    figname: str = 'SI_seed_emissions_comparison',
) -> None:
    """
    6-panel figure, stacked vertically on a shared x-axis: converged
    (final-iteration) emissions trajectories for the seed closest to the
    25th percentile, the median, and the 75th percentile of skill (from
    utils_inverse.load_SI_seed_emissions_comparison_data), one line each per
    panel.
    """
    if agent_units is None:
        agent_units = {"CO2": "Gt/yr", "CH4": "Mt/yr", "N2O": "Mt/yr", "Sulfur": "Mt/yr", "BC": "Mt/yr"}
    if agent_cmaps is None:
        agent_cmaps = {
            "CO2_single": cm.batlow, "CO2": cm.davos, "CH4": cm.lajolla,
            "N2O": cm.acton, "Sulfur": cm.bamako, "BC": cm.oslo,
        }
    if shade_fracs is None:
        shade_fracs = {"q25": 0.85, "median": 0.55, "q75": 0.25}

    label_map = {"q25": "25th percentile seed", "median": "Median seed", "q75": "75th percentile seed"}

    if figsize is None:
        figsize = (panel_width, panel_height * 6)

    multi_agents = ('CO2', 'CH4', 'N2O', 'Sulfur', 'BC')
    agent_math = {"CO2": "CO$_2$", "CH4": "CH$_4$", "N2O": "N$_2$O", "Sulfur": "Sulfur", "BC": "BC"}
    panels = [(f"CO$_2$ (single-forcing{title_suffix})", "CO2", "CO2_single", co2_emissions, co2_seed_info)]
    for a in multi_agents:
        panels.append((f"{agent_math[a]} (multi-forcing{title_suffix})", a, a, multi_emissions[a], multi_seed_info))

    fig, axes = plt.subplots(6, 1, figsize=figsize, sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for i, (ax, (title, agent, cmap_key, emissions, seed_info)) in enumerate(zip(axes, panels)):
        cmap = agent_cmaps[cmap_key]
        for key in ("q25", "median", "q75"):  # legend/draw order: 25th, median, 75th
            series = np.asarray(emissions[key])
            label = label_map[key]
            if seed_info is not None:
                seed, nrmse = seed_info[key]
                label = f"{label} ({seed}, NRMSE={nrmse:.3f})"
            ax.plot(np.arange(series.shape[0]), series, lw=2, color=cmap(shade_fracs[key]), label=label)

        ax.text(0.015, 0.92, f"({chr(97 + i)}) {title}", transform=ax.transAxes,
                ha="left", va="top", fontsize=12, fontweight="bold", zorder=20,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

        unit = agent_units.get(agent, "units/yr")
        ax.set_ylabel(f"Emissions\n({unit})", fontsize=11)
        ax.set_xlim(0, next(iter(emissions.values())).shape[0])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right", ncol=3)

    axes[-1].set_xlabel("Year")

    if save:
        fig.savefig(FIGURES_DIR / f"{figname}.pdf", bbox_inches="tight")
    return fig, axes


def plot_SI_baseline_convergence(curves: np.ndarray, save: bool = False,
                                 figname: str = "SI_baseline_convergence"):
    """
    Supplement: baseline emulator training loss against training step, as a
    median line with an interquartile band over seeds.

    Args:
        curves: (n_seeds, K) training-loss curves.
        save: write the figure to Figures/ as PNG.
        figname: stem of the output file.
    """
    steps = np.arange(curves.shape[1]) + 1
    median_curve = np.median(curves, axis=0)
    # Clipped so the lower band edge stays renderable on the log axis.
    iqr_lo = np.clip(np.percentile(curves, 25, axis=0), a_min=1e-12, a_max=None)
    iqr_hi = np.percentile(curves, 75, axis=0)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.semilogx(steps, median_curve, lw=2, color=cm.batlowS(0),
                label="Baseline training loss (median)")
    ax.fill_between(steps, iqr_lo, iqr_hi, color=cm.batlowS(0), alpha=0.2, linewidth=0)

    ax.set_xlabel("Training step", fontsize=14)
    ax.set_ylabel("Baseline emulator training loss (MSE)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.3, which="both", ls="-")
    ax.margins(x=0, y=0)
    ax.set_xlim(left=1.1)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color=cm.batlowS(0), alpha=0.2, label="Interquartile range"))
    labels.append("Interquartile range")
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=12)

    if save:
        fig.savefig(FIGURES_DIR / f"{figname}.png", dpi=150, bbox_inches="tight")
    return fig, ax
