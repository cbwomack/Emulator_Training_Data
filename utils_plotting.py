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
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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

# Seed-aggregation convention shared by Figures 3, 4 and 5.
#
# 'median' (the manuscript default as of 2026-08-25) reports the median with an
# interquartile band. 'mean' reproduces each figure's own prior behaviour, which
# differed per figure - min-max for Fig 3, +/- std for Fig 4 - so the two are
# not interchangeable and `mean_band` records which one to reproduce.
#
# Why median + IQR: these seed distributions are right-skewed (mean > median at
# essentially every iteration), so a symmetric +/- std band misrepresents both
# tails, and on Fig 3's log axis its lower edge can go non-positive and fail to
# render at all - the reason Fig 3 used min-max instead. Median pairs naturally
# with the IQR, both bounds stay strictly positive, and the band is ~3x tighter:
# across Figure 4's 50 cells the number whose band straddles zero drops from 18
# to 8, while only ONE cell changes sign, so legibility improves without any
# conclusion moving. The IQR covers 50% of seeds, which captions must state.
def _preds_stride(results: dict, n_preds: int) -> int:
    """Outer iterations between consecutive preds_traj entries.

    preds_traj holds one pre-loop entry plus one every `preds_every` steps,
    while errors holds one per step plus that same pre-loop entry, so the
    stride divides exactly and can be recovered rather than assumed. Deriving
    it is what lets a legend report the true iteration number at any run
    length instead of a value hardcoded for one.
    """
    n_err = len(results.get("errors", []) or [])
    if n_preds > 1 and n_err > 1 and (n_err - 1) % (n_preds - 1) == 0:
        return (n_err - 1) // (n_preds - 1)
    meta = results.get("meta") or {}
    return int(meta.get("preds_every", 50))


def _highlight_indices(sel) -> tuple:
    """First / middle / last of a subsampled index list.

    The hardcoded alternative (`i in [0, 2, 20]`, `i in [0, 100, 1000]`) is
    correct only at one run length. These index lists come from
    np.linspace(0, N-1, num=max_lines): with max_lines=11 and a 1000-iteration
    run (N=21) the selection is [0,2,4,...,20], which is exactly why 0/2/20
    worked. Extend to 2000 (N=41) and it becomes [0,4,8,...,40] - index 2 is
    never selected, so that curve and its legend entry disappear from the
    figure with no error raised, and the old "final" index 20 is now the
    midpoint. Deriving the highlights from the selection keeps three of them,
    correctly placed, at any run length.
    """
    sel = np.asarray(sel)
    if sel.size == 0:
        return ()
    return (int(sel[0]), int(sel[sel.size // 2]), int(sel[-1]))


AGGREGATION_DEFAULT = "median"


def _aggregate_seeds(stacked, aggregation: str = AGGREGATION_DEFAULT,
                     mean_band: str = "std"):
    """Collapse a (n_seeds, ...) stack to (centre, low, high) for plotting.

    `mean_band` selects which band the 'mean' convention pairs with, so each
    figure can reproduce its own prior output exactly: 'std' for Figure 4,
    'minmax' for Figure 3.
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


def plot_rmse_comparison_single(
    results_list: list[dict],       # List of dictionaries
    baseline_error_list: list[float],     # Single float value
    agents: list[str],
    save: bool = False,
    seed_errors_list: list[list[dict] | None] = None,
    seed_baseline_error_list: list[list[float] | None] = None,
    aggregation: str = AGGREGATION_DEFAULT,
) -> None:
    """5-panel NRMSE-vs-update-step comparison, one panel per single-forcing agent experiment.

    `seed_errors_list`/`seed_baseline_error_list` (both optional, default None):
    one entry per panel (same order as `agents`), each either None (that panel
    plots the single-run line from `results_list`/`baseline_error_list`, exact
    prior behavior) or a list of per-seed values - `seed_errors_list[i]` a list
    of per-seed result dicts (each shaped like `results_list[i]`, i.e. with an
    "errors" key), `seed_baseline_error_list[i]` a list of per-seed baseline
    floats. When given, that panel plots the mean trajectory across seeds with
    a shaded min-max band (min/max rather than mean +/- std, since NRMSE is
    non-negative and a log y-axis can't render a negative lower bound - matches
    the mean +/- min-max spread convention already used for this pipeline's
    other seed-uncertainty plots, e.g. 0a_bilevel_scaled_plots.ipynb) instead
    of a single line. Leaving both at None (the default) is fully
    backward-compatible - reproduces the exact prior output.

    On the y-axis: `results_list`/`seed_errors_list` are expected to carry the
    penalty-corrected NRMSE that utils_inverse's load_fig3_* loaders now
    return, NOT a checkpoint's raw 'errors' (which is the full objective,
    NRMSE + smoothness_weight * sum(dU)^2). Passing raw 'errors' overstates
    N2O and BC by ~11% and CO2 by ~1.4%.

    Note for the caption: this curve is the bilevel objective evaluated on the
    scenarios being optimized against, so it is in-sample - a training curve,
    not held-out skill. Figure 4's retrain-and-evaluate path is the
    out-of-sample counterpart.
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
            # Ragged input means a half-migrated family: some seeds have been
            # extended to a longer run and others have not, or tasks are still
            # in flight and their checkpoints sit at intermediate multiples of
            # checkpoint_every. np.stack's own message ("all input arrays must
            # have the same shape") does not say which agent or which lengths,
            # which makes a transient mid-migration state look like a code bug.
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

def plot_rmse_comparison_multi(results: dict, baseline_error: float, save: bool = False,
                               seed_errors: list[dict] = None,
                               seed_baseline_errors: list[float] = None,
                               aggregation: str = AGGREGATION_DEFAULT) -> None:
    """3-panel figure: NRMSE-vs-step (left) plus optimal WMGHG and aerosol emissions trajectories (right, via _plot_agents).

    `seed_errors`/`seed_baseline_errors` (both optional, default None): the
    multi-agent analogue of plot_rmse_comparison_single's seed mode. When given,
    the left panel plots the aggregate trajectory across seeds with a band
    instead of a single line, using the same convention as Figures 3 and 4.
    Leaving both None reproduces the prior single-seed output exactly.

    The right-hand emissions panels stay single-trajectory regardless: they show
    one realized profile (`results['U_traj'][-1]`), and overlaying 50 of them
    would obscure rather than inform.

    `results["errors"]` is expected to be the penalty-corrected NRMSE that
    load_fig5_multi_forcing_data now returns, not a checkpoint's raw 'errors'
    (the full objective). For the multi-agent family the two coincide, since
    its tuned smoothness_weight is 0 - but that is a property of the current
    checkpoints, not of the figure.

    Note for the caption: as in Figure 3, the left panel is the bilevel
    objective on the optimization target, so it is in-sample. The right panels
    show U_traj[-1], whose high-frequency content is discussed in the Figure 5
    spectral analysis (scripts/6h_emissions_spectra.py).
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

    # --- FIX 2 & 3: Match Style and Fix Layering ---
    # We add the legend to 'last_ax' (the top layer) so it sits above all lines.
    # We use framealpha, edgecolor, and facecolor to match your text boxes.
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

            # Plot every time series in this experiment's list.
            # last_i rather than a hardcoded 1000: the final iterate must stay
            # drawn and labelled whatever the run length is (see
            # _highlight_indices), and it is included explicitly in case the
            # length is not a multiple of the 100-step sampling stride.
            last_i = len(entries_list) - 1
            for i, series in enumerate(entries_list):
                if i % 100 == 0 or i == last_i:
                  y_data = np.asarray(series).reshape(-1)
                  alpha = 0.2 + 0.6 * (i / max(1, len(entries_list) - 1))
                  if col_idx < 1:
                      x_data = np.arange(0, len(y_data)) + 2024
                  else:
                      x_data = np.arange(0, len(y_data)) + 1750
                  if i in (0, last_i):
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

        # Which of the plotted curves get a legend entry, and how to name them.
        # Both are derived from the data rather than hardcoded so the figure
        # stays correct when the run length changes - see _highlight_indices.
        hl_pred = _highlight_indices(sel_pred)
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
              # Label with the true outer iteration, derived from the stride,
              # so this reads correctly at 1000, 2000 or any other length.
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

        # Which of the plotted curves get a legend entry, and how to name them.
        # Both are derived from the data rather than hardcoded so the figure
        # stays correct when the run length changes - see _highlight_indices.
        hl_pred = _highlight_indices(sel_pred)
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
              # Label with the true outer iteration, derived from the stride,
              # so this reads correctly at 1000, 2000 or any other length.
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
                                  seed_optimized_results: list[dict] = None,
                                  aggregation: str = AGGREGATION_DEFAULT) -> None:
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

    pct_err = None
    if seed_mode:
        n_seeds = len(seed_baseline_results)
        stacked = np.stack([
            _pct_improvement_for(seed_baseline_results[s], seed_optimized_results[s])
            for s in range(n_seeds)
        ], axis=0)
        plot_data, lo, hi = _aggregate_seeds(stacked, aggregation, mean_band="std")
        # ax.bar's yerr wants distances from the bar top, not absolute bounds,
        # and the IQR is asymmetric about the median, so both halves are passed
        # explicitly. Clipped at 0 because a centre outside its own band (which
        # +/- std can produce on a skewed distribution) would otherwise give
        # matplotlib a negative error length.
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
    # Same dark-gray shade used for in-objective rows in Figure 6
    # (plot_ood_r2_forest's SHADE_COLORS["dark"]). A column is leaked on a
    # given row when that column's training scenario is (or, for "Opt. All",
    # includes) the row's evaluation scenario - e.g. "Opt. Priority 1"
    # evaluated on "Priority 1", or "Opt. All" evaluated on anything (Avg.
    # included, since every component going into that average was itself
    # in-objective).
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

    # Light-gray shading (Figure 6's SHADE_COLORS["light"]) on the Avg. bar
    # for every non-"Opt. All" column: that average is a weighted mix of
    # in- and out-of-objective evaluations, so it is partially - not fully -
    # leaked. "Opt. All" is left untouched above (its Avg. bar is already
    # fully dark-shaded, since every scenario feeding it is in-objective).
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
                               seed_optimized_results_list: list[list[dict] | None] = None,
                               aggregation: str = AGGREGATION_DEFAULT) -> None:
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
                                  figname: str = 'scenario_differences',
                                  seed_baseline_results: list[dict] | None = None,
                                  seed_optimized_results_list: list[list[dict] | None] | None = None,
                                  seed_agg: str = "mean_std") -> None:
    """
    Per-scenario bar chart of global NRMSE across baseline + multiple optimized
    variants, with optional group separators/labels. Used by 5a_paper_plots.ipynb.

    `seed_baseline_results`/`seed_optimized_results_list` (both optional,
    default None): `seed_baseline_results` is a list of per-seed baseline
    result dicts (each shaped like `baseline_results`); `seed_optimized_
    results_list` has one entry per series in `optimized_results_list`, each
    either None (that series plots the single-run point estimate, exact
    prior behavior) or a list of per-seed result dicts (that series plots
    a seed-spread error bar, matching the bar-chart seed-spread convention
    plot_grouped_improvement_bars already uses for Figure 4, rather than
    Figure 3's shaded-band line-plot convention, since this is a bar chart
    too). Leaving both at None (the default) is fully backward-compatible -
    reproduces the exact prior output.

    `seed_agg`: "mean_std" (default, preserves prior behavior exactly) plots
    mean +/- std as a symmetric xerr; "median_iqr" plots the median with an
    asymmetric xerr spanning [Q1, Q3] instead - matching the median+IQR
    convention already used for Figs 3/4/5's seed-spread panels. Only takes
    effect where seed_optimized_results_list/seed_baseline_results are given.
    """
    if seed_agg not in ("mean_std", "median_iqr"):
        raise ValueError(f"seed_agg must be 'mean_std' or 'median_iqr', got {seed_agg!r}")

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
    bar_xlim = (-70, 90)  # shared with the hatch check below and ax_bar.set_xlim,
                          # so a bar clipped past the visible axis is always the
                          # one that gets hatched (previously a separate, out of
                          # sync -80 threshold left clipped bars un-hatched)

    y_positions = np.arange(n_total)
    ax_bar.invert_yaxis()

    # 2. Plotting Loop for Bar Chart
    # ------------------------------
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
    for opt_idx, opt_dict in enumerate(optimized_results_list):
        seed_opt = seed_optimized_results_list[opt_idx] if seed_optimized_results_list else None

        if seed_opt is not None and seed_baseline_results is not None:
            any_seed_err = True
            per_seed = np.array([_pct_change_row(b, o) for b, o in zip(seed_baseline_results, seed_opt)])
            if seed_agg == "median_iqr":
                diff_values = np.median(per_seed, axis=0)
                q1 = np.percentile(per_seed, 25, axis=0)
                q3 = np.percentile(per_seed, 75, axis=0)
                diff_err = np.vstack([diff_values - q1, q3 - diff_values])
            else:
                diff_values = per_seed.mean(axis=0)
                diff_err = per_seed.std(axis=0)
        else:
            diff_values = _pct_change_row(baseline_results, opt_dict)
            diff_err = None

        offset = (opt_idx - n_opts / 2) * bar_width + (bar_width / 2)

        bars = ax_bar.barh(y_positions + offset,
                   diff_values,
                   xerr=diff_err,
                   error_kw=dict(capsize=2, elinewidth=1) if diff_err is not None else None,
                   label=legend_labels[opt_idx],
                   height=bar_width,
                   color=colors[opt_idx],
                   edgecolor='black',
                   linewidth=0.7,
                   zorder=3)

        # Hatch bars clipped past the visible x-axis range.
        for bar, val in zip(bars, diff_values):
            if val < bar_xlim[0] or val > bar_xlim[1]:
                bar.set_hatch('//')

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

    ax_bar.set_xlim(bar_xlim)
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

    # Explicit solid-color legend handles for the IC series - built from proxy
    # Patches rather than the auto-generated bar-container handles, since the
    # latter pick up whichever individual bar's hatch (set above for bars
    # clipped past the x-axis limits) happens to land on the legend swatch,
    # making e.g. "Const." render hatched in the legend even though the
    # hatch is a per-bar clipping indicator, not a property of that series.
    legend_handles = [
        mpatches.Patch(facecolor=colors[i], edgecolor='black', linewidth=0.7, label=legend_labels[i])
        for i in range(n_opts)
    ]
    if seed_agg == "median_iqr" and any_seed_err:
        legend_handles.append(
            Line2D([0, 1], [0, 0], color='black', linewidth=1, marker='|', markersize=10,
                   markeredgewidth=1, label='Interquartile range')
        )

    ax_bar.legend(handles=legend_handles, title='Emulator IC',
                    loc='lower right',
                    numpoints=2,  # else Line2D's default numpoints=1 draws the IQR
                                  # marker at the swatch midpoint instead of both ends
                    title_fontsize=16,
                    frameon=True,
                    fancybox=True,
                    framealpha=0.8,
                    facecolor='white',
                    edgecolor='#cccccc',
                    fontsize=14)

    ax_bar.set_ylabel('Scenario', fontsize=18)
    x_label = (r'Median performance change from baseline emulator [\%]'
               if seed_agg == "median_iqr" and any_seed_err
               else r'Performance change from baseline emulator [\%]')
    fig.supxlabel(x_label, fontsize=18)

    if save:
        plt.savefig(FIGURES_DIR / f'{figname}.pdf', bbox_inches='tight')


# ------------------------------------------------------------------
# Figure 7 v2 (Stage 6o): the same optimized-emissions/temperature panels as
# plot_scenario_difference_bars2's (a)/(b), but one per initial condition and
# showing percentile trajectories across a 50-seed sweep instead of a single
# run. Kept as separate functions rather than options on
# plot_scenario_difference_bars2 - that function's panels are driven by real
# MESM ensemble output for two fixed trajectories, which is a different data
# source and a different claim.
# ------------------------------------------------------------------

# Line weight/alpha per percentile: the median reads as the main line, the
# quartiles as context, without needing three separate colours per axis.
_FIG7V2_QUANTILE_STYLE = {
    25: {"lw": 1.0, "alpha": 0.55, "ls": "-"},
    50: {"lw": 2.0, "alpha": 1.00, "ls": "-"},
    75: {"lw": 1.0, "alpha": 0.55, "ls": "-"},
}


def plot_fig7_v2_ic_panels(panels: list[dict],
                           years_emis: np.ndarray,
                           years_temp: np.ndarray,
                           save: bool = False,
                           figname: str = 'fig07_v2_ic_panels',
                           placeholder_bar_panel: bool = True) -> None:
    r"""
    Figure 7 v2: one panel per initial condition, each showing the 25th/50th/
    75th-percentile optimized CO2 emissions trajectory (left axis) and the
    temperature response it produces (right axis, dashed).

    `panels` is one dict per panel, in plotting order, with keys:
        'title'      panel title text, without the "(a) " prefix (added here)
        'emissions'  {25: (T,), 50: (T,), 75: (T,)}
        'delT'       {25: (T',), 50: (T',), 75: (T',)}
    plotted against `years_emis` (T,) and `years_temp` (T',) respectively. The
    two lengths are allowed to differ - the SCM response is defined on every
    emissions year (T' == T), but plot_scenario_difference_bars2's real-MESM
    series is one year shorter - so neither is assumed from the other.
        'seeds'      {25: int, 50: int, 75: int}  - which seed each one is
        'scores'     {25: float, 50: float, 75: float} - its weighted NRMSE
    Percentiles are over each seed's average skill across all scenarios, so the
    "50th" line is the median-performing seed's actual trajectory, not a
    pointwise median of 50 trajectories - the latter would be a curve no run
    ever produced.

    IMPORTANT for the caption: unlike plot_scenario_difference_bars2's (a)/(b),
    `delT` here is the MESM-calibrated SCM's own temperature response, not
    output from a real MESM ensemble run. No MESM runs exist for these
    newly-optimized trajectories.

    `placeholder_bar_panel` (default True) reserves the bottom axis the
    per-scenario NRMSE bar chart will occupy, drawn empty and labelled, so the
    figure's final proportions are visible before that panel's data source is
    settled. Set False for a clean three-panel figure.
    """
    n = len(panels)
    layout = [[f"P{i}"] for i in range(n)]
    height_ratios = [1] * n
    if placeholder_bar_panel:
        layout.append(["Bar"])
        # 5 units for the bar chart against 1 per time-series panel, and ~2.9 in
        # of figure per unit - the same proportions plot_scenario_difference_
        # bars2 uses (height_ratios [1, 1, 5] at figsize height 19.5), so the
        # two versions of Figure 7 are directly comparable side by side.
        height_ratios.append(5)

    fig, axd = plt.subplot_mosaic(
        layout, figsize=(8.7, 2.9 * sum(height_ratios)),
        constrained_layout=True, height_ratios=height_ratios,
    )

    axes = [axd[f"P{i}"] for i in range(n)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
        ax.sharey(axes[0])
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    # Shared temperature limits across panels, so the three ICs are visually
    # comparable rather than each autoscaled to its own range.
    all_t = np.concatenate([np.asarray(p['delT'][q]).reshape(-1)
                            for p in panels for q in (25, 50, 75)])
    t_min, t_max = float(all_t.min()), float(all_t.max())

    c_emis = cm.actonS(2)
    c_temp = cm.actonS(4)
    letters = 'abcdefgh'

    for i, (ax, panel) in enumerate(zip(axes, panels)):
        ax_temp = ax.twinx()

        for q in (25, 50, 75):
            style = _FIG7V2_QUANTILE_STYLE[q]
            ax.plot(years_emis, np.asarray(panel['emissions'][q]).reshape(-1),
                    c=c_emis, zorder=3, **style)
            ax_temp.plot(years_temp, np.asarray(panel['delT'][q]).reshape(-1),
                         c=c_temp, zorder=1,
                         lw=style["lw"], alpha=style["alpha"], ls='--')

        ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0, c=c_emis)
        ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
        ax.tick_params(axis='y', labelcolor=c_emis)

        ax_temp.grid(linestyle='--', alpha=0.3, zorder=0, c=c_temp)
        ax_temp.set_ylim(t_min - 0.75, t_max + 0.75)
        ax_temp.tick_params(axis='y', labelcolor=c_temp)

        ax.set_ylabel(r'Emissions [GtCO$_2$/yr]', fontsize=16, c=c_emis)
        ax_temp.set_ylabel(r"$\overline{\Delta T}(t)$ [$^\circ$C]", fontsize=16,
                           rotation=270, labelpad=15, c=c_temp)

        ax.text(
            0.02, 0.94, rf"({letters[i]}) {panel['title']}", transform=ax.transAxes,
            ha="left", va="top", fontsize=16, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
        )

        if i == 0:
            handles = [
                Line2D([0], [0], c=c_emis, **_FIG7V2_QUANTILE_STYLE[50]),
                Line2D([0], [0], c=c_emis, **_FIG7V2_QUANTILE_STYLE[25]),
                Line2D([0], [0], c=c_temp, ls='--', lw=2.0),
            ]
            labels = ['Emissions (median seed)',
                      'Emissions (25th/75th pct. seed)',
                      r'SCM $\overline{\Delta T}(t)$']
            ax_temp.legend(handles, labels, frameon=True, loc='lower left',
                           fancybox=True, framealpha=0.8, facecolor='white',
                           edgecolor='#cccccc', fontsize=13)

    axes[0].set_xlim([float(years_emis[0]), float(years_emis[-1])])
    axes[-1].set_xlabel('Year', fontsize=16)

    if placeholder_bar_panel:
        ax_bar = axd["Bar"]
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        for spine in ax_bar.spines.values():
            spine.set_linestyle((0, (6, 6)))
            spine.set_edgecolor('gray')
        ax_bar.text(
            0.5, 0.5,
            f"({letters[n]}) per-scenario NRMSE bar chart\n"
            "[placeholder - data source not yet settled]",
            transform=ax_bar.transAxes, ha="center", va="center",
            fontsize=15, color='gray',
        )

    if save:
        plt.savefig(FIGURES_DIR / f'{figname}.pdf', bbox_inches='tight')


def plot_ic_convergence_seed_spread(seed_errors_list: list[list],
                                    seed_baseline_error_list: list[list],
                                    labels: list[str],
                                    aggregation: str = AGGREGATION_DEFAULT,
                                    save: bool = False,
                                    figname: str = 'fig07_v2_convergence') -> None:
    """
    NRMSE-vs-update-step, one panel per initial condition, with a seed-spread
    band and the baseline emulator's own spread as a horizontal reference.

    Same conventions as plot_rmse_comparison_single (which is hardwired to a
    5-panel single-forcing-agent mosaic, hence a separate function): log-log
    axes, median + IQR by default via _aggregate_seeds.

    `seed_errors_list[i]` is a list of per-seed (n_updates,) NRMSE arrays for
    panel i; `seed_baseline_error_list[i]` a list of per-seed baseline floats.

    On the y-axis: these must be penalty-corrected NRMSE
    (utils_inverse.recover_nrmse_trajectory), NOT a checkpoint's raw 'errors',
    which is the full objective NRMSE + smoothness_weight * sum(dU)^2.

    Note for the caption: this curve is the bilevel objective evaluated on the
    scenarios being optimized against, so it is in-sample - a training curve,
    not held-out skill. The weighted-NRMSE comparison built from
    evaluate_optimal_emulator is the out-of-sample counterpart.
    """
    n = len(seed_errors_list)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=True,
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    cmap = cm.batlowS

    for i, ax in enumerate(axes):
        traj = [np.asarray(e).reshape(-1) for e in seed_errors_list[i]]
        lens = sorted({t.shape[0] for t in traj})
        if len(lens) > 1:
            raise ValueError(
                f"panel {i} ({labels[i]}) has seeds at different trajectory lengths "
                f"{lens}: some runs timed out mid-optimization or were resumed to a "
                f"different num_updates. Finish the family before plotting - mixing "
                f"lengths would average different run stages together."
            )
        stacked = np.stack(traj, axis=0)
        centre, lo, hi = _aggregate_seeds(stacked, aggregation, mean_band="minmax")
        x = np.arange(centre.shape[0])
        ax.loglog(x, centre, lw=2, color=cmap(0), label="Optimized emulator")
        ax.fill_between(x, lo, hi, color=cmap(0), alpha=0.2, linewidth=0)

        base = np.asarray(seed_baseline_error_list[i], dtype=float)
        b_centre, b_lo, b_hi = _aggregate_seeds(base, aggregation, mean_band="minmax")
        ax.axhline(float(b_centre), ls="--", c=cm.lipariS(5), lw=1.5,
                   label="Baseline emulator")
        if float(b_hi) > float(b_lo):
            ax.axhspan(float(b_lo), float(b_hi), color=cm.lipariS(5), alpha=0.15, linewidth=0)

        ax.margins(x=0, y=0)
        ax.grid(True, alpha=0.3, which="both", ls="-")
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.text(
            0.05, 0.95, labels[i], transform=ax.transAxes,
            ha="left", va="top", fontsize=15, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.5)
        )

        if i == 0:
            ax.set_ylabel("Emulator error (NRMSE)", fontsize=17)
            handles, leg_labels = ax.get_legend_handles_labels()
            n_seeds = stacked.shape[0]
            band = mpatches.Patch(color='gray', alpha=0.2,
                                  label=_aggregation_label(aggregation, n_seeds))
            handles.append(band)
            leg_labels.append(_aggregation_label(aggregation, n_seeds))
            ax.legend(handles, leg_labels, frameon=True, fancybox=True, framealpha=0.8,
                      facecolor='white', edgecolor='#cccccc', fontsize=11, loc='lower left')

    fig.supxlabel("Outer update step", fontsize=17)

    if save:
        plt.savefig(FIGURES_DIR / f'{figname}.pdf', bbox_inches='tight')


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

  Recreates the Revision Response Ledger's dot-plus-IQR-line chart (Major 2),
  generalized from 2 series (baseline vs. one optimized config) to 5 (baseline
  plus every entry of color_map). One row per scenario, in `scenarios` order
  top-to-bottom, with one dodged marker+IQR line per emulator. The x-axis is
  clipped to [dmin, dmax] with off-scale points pinned to the axis and labeled,
  since baseline R^2 can be far below -1 (e.g. M_AER baseline ~ -11).

  `scenario_shade` optionally maps a scenario name to 'dark' or 'light' to
  draw a shaded background band behind that row - e.g. grouping rows by which
  scenario set they were drawn from, generalizing the ledger's own single
  shaded-reference-row convention to 2 shade levels. Scenarios absent from the
  dict get no shading.

  `text_ax`/`group_labels` optionally label each scenario-set group in a
  separate axis to the side, Figure 7's own group-label convention (rotated
  text centered on the group's row span) but in a dedicated column rather
  than past this axis's own right edge. `group_labels` is `(name, n_rows)`
  tuples in the same top-to-bottom order as `scenarios`, each spanning that
  many consecutive rows starting where the previous group left off.

  `legend_valign_group` optionally vertically centers the legend on one named
  group from `group_labels` (e.g. an unshaded group, so the legend sits
  against a plain white background instead of over a shaded band) instead of
  the default title-anchored position.

  Row spacing is not uniform: adjacent rows drawn from the same
  `group_labels` entry sit `within_group_gap` apart, adjacent rows from two
  different entries sit the wider `between_group_gap` apart - giving the
  divider lines drawn between rows below a real gap to occupy, and matching
  `text_ax`'s group labels to the rows they actually span.

  `r2_table` is the long-format list of dict rows from
  scripts/6j_fig6_ood_evaluate.py's --mode merge (scenario, emulator,
  r2_median, r2_p25, r2_p75, n_seeds; values may be strings, as from csv.DictReader).
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

  # Row y-positions: within_group_gap between two rows drawn from the same
  # group_labels entry, the wider between_group_gap at a group boundary - so
  # rows in the same scenario group sit tighter together, and the divider
  # lines/group text below have a real, larger gap at each group boundary to
  # sit in. Falls back to a uniform 1-unit spacing (the old behavior) when no
  # group_labels are given, since there's then no group structure to key off.
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
          # Extend only as far as within_group_gap/2 towards a same-group
          # neighbor (so two shaded rows meet with no seam), but all the way
          # to between_group_gap/2 towards a different group or the plot's
          # own edge - i.e. to the divider line itself - rather than stopping
          # at the tighter within_group_gap/2 there and leaving an unshaded
          # sliver before the divider.
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

  # Divider lines between rows, drawn instead of a uniform horizontal grid: a
  # darker line where two different scenario_groups meet (sitting in the
  # wider between_group_gap), a lighter one between two rows drawn from the
  # same group (sitting in the tighter within_group_gap).
  if group_labels:
      for i in range(1, len(scenarios)):
          y_div = (row_y_vals[i - 1] + row_y_vals[i]) / 2
          same_group = group_of_idx[i] == group_of_idx[i - 1]
          if same_group:
              ax.axhline(y_div, color="0.82", lw=0.7, alpha=0.8, zorder=0.5)
          else:
              ax.axhline(y_div, color="0.35", lw=1.2, alpha=0.9, zorder=0.5)

  # fontstyle="italic" is a no-op under rcParams["text.usetex"]=True (the
  # renderer ignores the Text property and just typesets the raw string) -
  # italics have to be requested in the LaTeX source itself. M_GHG/M_AER are
  # internal dict keys, not display names - DAMIP's own convention (matching
  # panel labels (a)/(b) above) is the hyphenated "M-GHG"/"M-aer", not an
  # underscore. The RAMIP-analog rungs (H-ext-Maer/VLaer/Laer) are themselves
  # the internal scenario keys (used for the r2_table lookup above), so they
  # get the same hyphenated-display treatment rather than a key rename. Any
  # other underscored tag falls back to an escaped underscore, since usetex
  # text mode treats a bare "_" as a LaTeX error.
  scenario_display = {"M_GHG": "M-GHG", "M_AER": "M-aer",
                       "H-ext-Maer": "H-ext-M-aer", "H-ext-VLaer": "H-ext-VL-aer",
                       "H-ext-Laer": "H-ext-L-aer"}
  ax.set_yticks(row_y_vals)
  ax.set_yticklabels([r"\textit{" + scenario_display.get(s, s).replace("_", r"\_") + "}"
                       for s in scenarios])
  # Flush with each row's own +/-row_half span (rather than the +/-0.6
  # padding `_clip` off-scale labels get room for) so a shaded top/bottom
  # group's background reaches the axis edge with no unshaded sliver above/
  # below it.
  ax.set_ylim(row_y_vals[-1] - row_half, row_y_vals[0] + row_half)
  ax.set_xlim(dmin - 0.05, dmax + 0.05)
  ax.set_xticks([dmin, dmin / 2, 0, dmax / 2, dmax])
  ax.set_xticklabels([rf"$\leq${dmin:g}"] + [f"{t:g}" for t in [dmin / 2, 0, dmax / 2, dmax]])
  ax.axvline(0, color="0.6", lw=1, ls="--", zorder=1)
  ax.set_xlabel(r"Median $R^2$")
  ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)

  if text_ax is not None and group_labels:
      # Same y-data-coordinate system as `ax`, so a group's row span here maps
      # directly onto text_ax's y axis via get_yaxis_transform() (Figure 7's
      # own technique, just written into a dedicated column's axis instead of
      # past this axis's own right edge).
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
  # The dodged horizontal lines themselves are colored per-emulator, not
  # gray - this entry is a generic explainer of what that line style means,
  # so it uses a neutral gray swatch rather than any one emulator's color.
  handles.append(Line2D([0], [0], color="0.5", lw=2.5, alpha=0.55,
                        solid_capstyle="round", label="Interquartile range"))

  legend_x, legend_y_below_title = 0.03, 0.90
  if title is not None:
      title_artist = ax.text(
              0.03, 0.985, title, transform=ax.transAxes,
              ha="left", va="top", fontsize=14, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
            )
      # Measure the title box's actual rendered extent so the legend below it
      # can be placed genuinely flush-left with it, rather than a guessed
      # axes-fraction offset that drifts if the title text or font changes.
      ax.figure.canvas.draw()
      renderer = ax.figure.canvas.get_renderer()
      bbox_disp = title_artist.get_bbox_patch().get_window_extent(renderer=renderer)
      bbox_axes = bbox_disp.transformed(ax.transAxes.inverted())
      legend_x, legend_y_below_title = bbox_axes.x0, bbox_axes.y0 - 0.015

  legend_loc, legend_anchor = "upper left", (legend_x, legend_y_below_title)
  if group_labels and legend_valign_group is not None:
      # Center the legend on one named group's row span (converted from data
      # y-coordinates to axes-fraction via the ylim set above), so it lands
      # against that group's background rather than the title-anchored spot.
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


def plot_individual_effects_summary(
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
  """Plot the multi-agent "individual effects" summary figure, in the
  pre-revision 2-column shape: a single stacked column on the left, a
  roughly-square panel on the right.

  Left column (6 rows, a-f): SCM-projected vs. baseline- and all-four-
  optimized-configs' predicted temperature trajectories. Rows (a)-(c) are the
  single-driver in-objective scenarios (M-GHG, M-aer, G6sulfur); rows (d)-(f)
  are the 3 downselected out-of-objective scenarios
  (utils_inverse.FIG6_OOD_SCENARIOS - H-ext-VLaer, ssp534-over,
  esm-bell-2000PgC). Every row draws all four entries of
  train_scenarios_ind_effects (Opt. Tier 1 / DAMIP / GeoMIP / All) against the
  baseline - rows (d)-(f) need `ood_scenarios`/`ood_seed_traj` from
  utils_inverse.load_fig6_ood_data and are left blank without them.

  Right column ('Right', replaces the old scatter/regression panel): a forest
  plot of median R^2 with IQR across seeds, one row per scenario (the full
  out-of-objective roster minus esm-pi-CO2pulse, plus the 3 in-objective
  references, shaded) and one dodged marker per emulator config - a Python
  recreation of the Revision Response Ledger's chart (Major 2), generalized
  from 2 series to 5. See plot_ood_r2_forest. Needs `r2_table` from
  utils_inverse.load_fig6_ood_data.

  `seed_cache` optionally supplies {seed: {y_true, y_hat_baseline, y_hat}} from
  utils_inverse.regenerate_fig6_individual_effects_cache_seed_sweep. When given:

    - rows (a)-(c) draw median lines with IQR bands across seeds, matching
      Figures 3/4/5.

  When `seed_cache` is None, rows (a)-(c) behave exactly as before (single-seed
  lines). When `ood_scenarios`/`ood_seed_traj`/`r2_table` are None, the
  corresponding rows/right column are left blank (with a printed note) rather
  than raising, so a partial data refresh still produces a figure.
  """
  i_ppt = 0

  # Back to the pre-revision 2-column shape (unchanged figsize below) - the
  # left column now stacks all 6 scenarios (3 in-objective + 3
  # out-of-objective) instead of 3, and the right column's forest plot spans
  # the same full figure height it always has, keeping its original
  # roughly-square shape. A 3rd, narrow 'Text' column (carved out of 5% of
  # Left's former width, not added on top of the figure width) sits to the
  # right of 'Right' for the per-scenario-group labels (see group_labels
  # below) - Figure 7's own group-label convention places that text just
  # outside its own axes via axes-fraction x>1, but a dedicated column gives
  # explicit, guaranteed room instead of relying on constrained_layout to pad
  # around out-of-axes text.
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

  # Baseline (lipariS(5), coral ~8 deg hue) and Opt. All (osloS(2), blue ~216
  # deg) stay fixed. The other 3 configs used to all come from actonS (a
  # narrow pink-purple range), which made them hard to tell apart - drawing
  # each from a different Crameri categorical map instead spreads them to
  # gold (~47 deg), green (~139 deg), and magenta (~324 deg), roughly evenly
  # spaced around the hue wheel from the two fixed anchors.
  color_map = {'Opt. Tier 1':cm.bamakoS(15), 'Opt. DAMIP':cm.hawaiiS(3), 'Opt. GeoMIP': cm.budaS(3), 'Opt. All':cm.osloS(2)}
  baseline_color = cm.lipariS(5)
  display_label = {'Opt. Tier 1': 'Opt. Prio. 1', 'Baseline Em.': 'Baseline Em.',
                    'Opt. DAMIP': 'Opt. DAMIP', 'Opt. GeoMIP': 'Opt. GeoMIP', 'Opt. All': 'Opt. All'}

  # One unified left-column stack: in-objective references first (a-c), then
  # the 3 downselected out-of-objective scenarios (d-f).
  panel_kind = {"Left1": "in_obj", "Left2": "in_obj", "Left3": "in_obj",
                "Left4": "ood", "Left5": "ood", "Left6": "ood"}
  panel_scenario = {"Left1": "M_GHG", "Left2": "M_AER", "Left3": "G6sulfur",
                     "Left4": "H-ext-VLaer", "Left5": "ssp534-over", "Left6": "esm-bell-2000PgC"}

  # 3 harmonized x-axis sub-groups: (a)-(c) share real calendar 1850-2150
  # (unchanged); (d)-(e) share real calendar 2024-2100 (H-ext-VLaer's native
  # data runs to 2150 and gets cropped - deliberate, per user request); (f) is
  # the odd one out - esm-bell-2000PgC is a piControl-branched idealized
  # experiment, not real historical time (see load_fig6_ood_data), so instead
  # of a calendar axis it gets a relative "years since branch" axis, 0-200.
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
          # 'YLabel' is a narrow spacer column carved out to the left of
          # 'Left', holding the manually-drawn y-axis label (see below) -
          # w_pad/h_pad are zeroed for the whole figure (next line), so
          # fig.supylabel's automatic placement sat flush against 'Left'
          # with no real gap; a dedicated column, drawn into the same way as
          # 'Text' on the right, gives that gap explicit, guaranteed room.
          "width_ratios": [0.35, 4.75, 3, 0.25],
          # The 2 '.' spacer rows (breaks between harmonized x-axis groups)
          # get a small fraction of a data row's height - visual separation
          # between groups is concentrated there, not in gridspec's own
          # wspace/hspace (see set_constrained_layout_pads below). Reduced
          # 0.15 -> 0.08 -> 0.03 across successive tightening passes; the
          # freed height_ratios budget also goes straight to the 6 data
          # rows' own share of a fixed figsize, so panels grow slightly too.
          "height_ratios": [1, 1, 1, 0.03, 1, 1, 0.03, 1],
      }
  )
  # constrained_layout ignores gridspec_kw's own wspace/hspace here - it
  # reserves a fixed minimum gap for each axes' tick-label space regardless
  # (even where tick_params(labelbottom=False) hides the labels: hiding
  # doesn't free the space constrained_layout reserved for them). The actual
  # knob is the layout engine's own pads, set directly - zeroing these is what
  # lets same-harmonized-group panels (a-b, b-c, d-e) sit close together,
  # while the '.' spacer rows above still give real separation between groups.
  fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.0, wspace=0.0, hspace=0.0)
  ax_dict["Text"].axis("off")
  ax_dict["YLabel"].axis("off")

  has_ood = bool(ood_scenarios and ood_seed_traj)
  if not has_ood:
      print("plot_individual_effects_summary: no OOD data supplied - Left4-6 left blank. "
            "Pass ood_scenarios/ood_seed_traj from utils_inverse.load_fig6_ood_data.")

  # Tracks the min/max of every array actually drawn (band edges included, not
  # just medians) across all 6 panels, so every row can share one y-axis range
  # covering the full data extent regardless of scenario.
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

          # Ground truth is the SCM, not an emulator, so it carries no seed
          # spread and stays a single line. Everything below it gets a band
          # when the seed cache is supplied.
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
              # bottom=False (not just labelbottom=False): constrained_layout
              # reserves padding for the tick marks themselves, not only
              # their labels - with h_pad/hspace already at 0, this residual
              # reservation was the remaining within-group gap.
              ax_plot.tick_params(axis="x", bottom=False, labelbottom=False)

      else:  # kind == "ood"
          tag = scen_plot
          scen = ood_scenarios.get(tag)
          if scen is None:
              continue
          years = scen["years"]
          x_plot = (years - years.min()) if tag in ood_relative_x else years
          xlim = ood_xlim[tag]
          # Only track y-bounds over the portion actually left visible after
          # cropping to the harmonized xlim, so a cropped-off tail (e.g.
          # H-ext-VLaer's 2100-2150) doesn't waste shared y-axis range.
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

          # Unlike the top 3 rows, every optimized config is drawn here - the
          # point of these rows is comparing all four configs' genuinely
          # out-of-distribution skill, not just 'Opt. All's.
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

  # Apply one shared y-axis range (data min/max across every panel, with a
  # small pad) to all 6 rows, in place of the old fixed [-4.2, 5.2] used only
  # for the in-objective rows.
  if np.isfinite(y_min) and np.isfinite(y_max):
      data_range = y_max - y_min
      bottom_pad = 0.04 * data_range
      # The (a)-(f) title box sits near the top of each panel (y=0.915 axes
      # fraction, below); a bigger top-only pad gives it clear whitespace
      # above the plotted curves instead of the tight 4% pad crowding it.
      top_pad = 0.16 * data_range
      shared_ylim = [y_min - bottom_pad, y_max + top_pad]
      for ax_label in panel_scenario:
          ax_dict[ax_label].set_ylim(shared_ylim)

  # Each of the 3 harmonized x-axis sub-groups gets its own axis label at the
  # bottom of that group, since (f)'s relative axis isn't calendar "Year".
  ax_dict["Left3"].set_xlabel("Year")
  ax_dict["Left5"].set_xlabel("Year")
  ax_dict["Left6"].set_xlabel("Years since piControl branch")

  # One shared legend for the whole left column - every row uses the same
  # color/style mapping (SCM-projected, Baseline Em., and all 4 optimized
  # configs), so a single legend inside panel (b) replaces the two separate
  # in-panel legends that used to fight the compressed row heights for space.
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

  # Drawn into the dedicated 'YLabel' column rather than fig.supylabel, whose
  # automatic placement sat flush against 'Left' once w_pad was zeroed above
  # - x=0.3 (left-of-center within the column) leaves a visible gap between
  # the label and 'Left's own left edge instead.
  ax_dict["YLabel"].text(0.3, 0.5, r'Temperature anomaly [$^\circ$C]',
                          transform=ax_dict["YLabel"].transAxes,
                          rotation=90, ha='center', va='center', fontsize=16)

  # -- Right column: median R^2 + IQR forest plot across all scenarios/configs --
  if r2_table:
      # The full out-of-objective roster minus esm-pi-CO2pulse (Ledger Major
      # 2's own critique: 0.21 K of total signal, R^2 dominated by a
      # single-year spike), grouped by which scenario set each was drawn from
      # rather than by skill - within each group, order matches the ledger's
      # own descending-'Opt. All'-median-R^2 convention. CMIP5's only entries
      # here are the RCP scenarios (rcp45/rcp85); CMIP6 covers 3 different
      # MIPs native to that generation (ScenarioMIP's ssp534-over, RAMIP's own
      # ssp370-126aer, AerChemMIP's ssp370-lowNTCF) - as opposed to the CMIP7
      # RAMIP-analogue rungs (H-ext-*), which get their own group.
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
      print("plot_individual_effects_summary: no r2_table supplied - right column left blank. "
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
            # i indexes preds_traj, which is sampled every preds_every outer
            # steps - so it must be scaled to be reported as a step number.
            # Labelling it directly understated every step by that factor
            # (at preds_every=50, "Step 4" was really step 200).
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
    by supplementary_notebooks/SI_plots.ipynb's sensitivity-sweep figures
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
            plot_rmse_comparison_single convention) instead of a single
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
# Stage 6i: Figure 6 OOD extension
# ==================================================================

# One colour per scenario GROUP, linestyle separating members within a group -
# identity is never carried by colour alone. Colours are cmcrameri batlowS
# entries, matching this repo's existing figures (Figure 6 uses cm.lipariS /
# cm.osloS). Validated all-pairs: worst CVD deltaE 10.0, worst normal-vision
# deltaE 16.5, every contrast-vs-white >= 2.48 - so the set clears the CVD (>=8)
# and normal-vision (>=15) separation targets.
#
# NOTE on lightness: batlowS deliberately spans a WIDE lightness range, which is
# what keeps its categories distinguishable when a journal prints in greyscale.
# That is an intentional deviation from the narrow-lightness-band convention used
# for on-screen categorical palettes, and it is the right trade for a manuscript
# figure. Do not "fix" it by re-picking within one lightness band: doing so
# collapses the available batlowS colours to a single olive/orange family whose
# pairwise separation then fails outright.
#
# Colours are keyed by group, NOT by position, so downselecting the scenario
# roster later cannot repaint the survivors.
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
    within that group. Shared by every Stage 6i figure so a scenario keeps one
    identity across them."""
    style, per_group = {}, {}
    for tag, d in scen.items():
        g = d["group"]
        i = per_group.get(g, 0)
        per_group[g] = i + 1
        style[tag] = dict(color=OOD_GROUP_COLORS[g], ls=OOD_LINESTYLES[i % len(OOD_LINESTYLES)])
    return style


def plot_ood_scenario_overview(cache, xlim=(1850, 2160), save=None, figsize=(18, 9)):
    """Six panels - one per forcing agent plus the SCM-simulated GMST - with one
    line per OOD scenario on every panel.

    This is the Phase-A diagnostic for Stage 6i: it makes the whole candidate
    roster legible before any emulator touches it, showing which scenarios sit
    genuinely far from the training distribution, which are near-duplicates, and
    whether any splice or unit conversion went wrong.

    `cache` is the dict written by scripts/6i_fig6_ood_extension.py, with keys
    'scenarios', 'references', 'historical', 'agents'. In-sample CMIP7 references
    are drawn as thin grey lines for context and are not part of the roster.

    CAVEAT on the x-axis: the CO2-only idealized scenarios are piControl-branched
    and carry NO calendar meaning - RCMIP simply labels their branch year 1850, so
    esm-bell-* peaks at "1899" and esm-pi-CO2pulse at "1860". They are drawn on the
    same calendar axis for compactness, but they are not contemporaneous with the
    historical period the other scenarios are prepended with. This is the same
    convention the repo already uses for DECK (utils_inverse.py:683).
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    axes = axes.ravel()

    scen = cache["scenarios"]
    refs = cache.get("references", {})
    hist = cache.get("historical", {})

    style = _ood_scenario_styles(scen)

    hist_years = None
    if hist:
        n_hist = len(next(iter(hist.values())))
        hist_years = np.arange(2024 - n_hist, 2024)

    for ax, (key, ylabel) in zip(axes, _OOD_PANEL_SPECS):
        # Historical context (shared by every scenario that prepends it).
        if key != "GMST" and hist_years is not None:
            ax.plot(hist_years, hist[key], color="0.55", lw=1.0, zorder=1)

        # In-sample references, recessive.
        for rtag, rd in refs.items():
            yv = rd["y_scm"] if key == "GMST" else rd["emis"][key]
            ax.plot(rd["years"], yv, color="0.7", lw=1.0, zorder=1)

        for tag, d in scen.items():
            yv = d["y_scm"] if key == "GMST" else d["emis"][key]
            ax.plot(d["years"], yv, lw=1.8, zorder=3,
                    color=style[tag]["color"], ls=style[tag]["ls"])

        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    for ax in axes[3:]:
        ax.set_xlabel("Year")

    # Legend: group colour + per-scenario linestyle, so identity is colour+dash.
    handles = []
    for g, label in OOD_GROUP_LABELS.items():
        members = [t for t, d in scen.items() if d["group"] == g]
        if not members:
            continue
        handles.append(Line2D([], [], color="none", label=r"\textbf{%s}" % label))
        for t in members:
            handles.append(Line2D([], [], color=style[t]["color"], ls=style[t]["ls"],
                                  lw=1.8, label=t))
    handles.append(Line2D([], [], color="0.7", lw=1.0, label="In-sample reference"))

    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=12, handlelength=2.4)

    if save is not None:
        fig.savefig(FIGURES_DIR / f"{save}.pdf", bbox_inches="tight", transparent=True)
    return fig, axes


def plot_harmonization_diagnostic(cache, xlim=(2010, 2100), save=None, figsize=(18, 9)):
    """Before/after evidence for the Gidden et al. (2018) harmonization.

    Five emissions panels plus a summary panel. In each emissions panel the
    published RCMIP pathway is drawn faint and dashed, the harmonized pathway
    solid, and the repo's own CMIP7 historical in grey. The two vertical rules
    mark the harmonization year (2023, where the corrected pathway is pinned to
    the historical inventory) and the convergence year (2080, from which the
    correction is exactly zero and the pathway is the scenario as published).

    The sixth panel is what the exercise is for: the size of the step at the
    2023->2024 handoff, per species and scenario, before and after. Open marker =
    published, filled = harmonized, joined by a rule.

    Only scenarios with harmonize=True appear. The CMIP7 crosses are drawn from
    the same source file as the historical and the CO2-only experiments are
    piControl-branched, so neither is harmonized and neither belongs here.

    `cache` is the dict written by scripts/6i_fig6_ood_extension.py.
    """
    scen = {t: d for t, d in cache["scenarios"].items() if d.get("harmonized")}
    if not scen:
        raise ValueError("no harmonized scenarios in this cache")
    hist = cache["historical"]
    agents = cache["agents"]
    style = _ood_scenario_styles(cache["scenarios"])

    n_hist = len(next(iter(hist.values())))
    hist_years = np.arange(2024 - n_hist, 2024)

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    axes = axes.ravel()

    units = dict(_OOD_PANEL_SPECS)
    for ax, agent in zip(axes, agents):
        ax.plot(hist_years, hist[agent], color="0.55", lw=1.2, zorder=2)
        for tag, d in scen.items():
            # Raw and harmonized share the scenario's linestyle and differ only in
            # weight, so a pair reads as one scenario. Giving the raw line its own
            # dash pattern instead makes it collide with whichever scenario is
            # already dashed.
            ax.plot(d["years"], d["emis_raw"][agent], lw=0.9, alpha=0.4, zorder=3,
                    color=style[tag]["color"], ls=style[tag]["ls"])
            ax.plot(d["years"], d["emis"][agent], lw=1.8, zorder=4,
                    color=style[tag]["color"], ls=style[tag]["ls"])

        for yr in (2023, 2080):
            ax.axvline(yr, color="0.4", lw=0.8, ls=(0, (1, 3)), zorder=1)

        ax.set_ylabel(units[agent])
        ax.set_xlim(xlim)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    # -- summary: the handoff step, before and after
    ax = axes[5]
    tags = list(scen)
    span = 0.72
    for i, agent in enumerate(agents):
        h = float(hist[agent][-1])
        for j, tag in enumerate(tags):
            d = scen[tag]
            y = i + span * (j / max(len(tags) - 1, 1) - 0.5)
            raw = 100.0 * (float(d["emis_raw"][agent][0]) - h) / abs(h)
            new = 100.0 * (float(d["emis"][agent][0]) - h) / abs(h)
            c = style[tag]["color"]
            ax.plot([raw, new], [y, y], color=c, lw=1.2, alpha=0.6, zorder=2)
            ax.plot(raw, y, "o", ms=6.5, mfc="white", mec=c, mew=1.6, zorder=3)
            ax.plot(new, y, "o", ms=6.5, color=c, mec="white", mew=1.0, zorder=4)

    ax.axvline(0.0, color="0.35", lw=1.0, zorder=1)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents)
    ax.set_ylim(len(agents) - 0.5, -0.5)
    ax.set_xlabel(r"Step at the 2023$\rightarrow$2024 handoff [\%]")
    ax.grid(alpha=0.25, lw=0.6, axis="x")
    ax.set_axisbelow(True)

    for a in axes[3:5]:
        a.set_xlabel("Year")

    handles = []
    for g, label in OOD_GROUP_LABELS.items():
        members = [t for t in tags if scen[t]["group"] == g]
        if not members:
            continue
        handles.append(Line2D([], [], color="none", label=r"\textbf{%s}" % label))
        for t in members:
            handles.append(Line2D([], [], color=style[t]["color"], ls=style[t]["ls"],
                                  lw=1.8, label=t))
    handles += [
        Line2D([], [], color="none", label=r"\textbf{Harmonization}"),
        Line2D([], [], color="0.35", ls="-", lw=0.9, alpha=0.4, label="Published (RCMIP)"),
        Line2D([], [], color="0.35", ls="-", lw=1.8, label="Harmonized"),
        Line2D([], [], color="0.55", ls="-", lw=1.2, label="CMIP7 historical"),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=12, handlelength=2.4)

    if save is not None:
        fig.savefig(FIGURES_DIR / f"{save}.pdf", bbox_inches="tight", transparent=True)
    return fig, axes


def plot_ood_scenario_grid(cache, sweep, aggregation=AGGREGATION_DEFAULT,
                           save=None, figsize=(18, 16)):
    """Stage 6i Phase B: every OOD scenario's emulated trajectory against SCM truth.

    A plain panel grid plus one pooled scatter, sized from the scenario list.
    `plot_individual_effects_summary` is deliberately NOT reused or modified
    here: its mosaic, shared `xlim=[1850,2150]`/`ylim=[-4.2,5.2]` and hardcoded
    legend pixel offsets `shift=[65,2,26,0,3,6]` all break on a different panel
    count, and the CO2-only scenarios span a different set of years anyway. The
    survivors get folded into the real figure only after downselection.

    `cache`  - the Phase A scenario cache (scripts/6i_fig6_ood_extension.py)
    `sweep`  - {seed: {optimized: {tag: {...}}, baseline: {...}}} from
               scripts/6j_fig6_ood_evaluate.py --mode collect

    Colours follow the existing Figure 6 convention exactly: SCM truth black
    dashed, baseline cm.lipariS(5), optimized cm.osloS(2). Median lines with IQR
    bands across seeds, matching Figures 3/4/5/6.
    """
    scen = cache["scenarios"]
    tags = list(scen)
    seeds = sorted(sweep)

    n = len(tags) + 1  # +1 for the pooled scatter
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    c_base, c_opt = cm.lipariS(5), cm.osloS(2)

    def _band(tag, key):
        stack = np.stack([np.asarray(sweep[s][key][tag]["yhat"]).reshape(-1) for s in seeds])
        return _aggregate_seeds(stack, aggregation, mean_band="minmax")

    def _r2_summary(tag, key):
        vals = np.asarray([sweep[s][key][tag]["r2"] for s in seeds])
        return np.median(vals), np.percentile(vals, 25), np.percentile(vals, 75)

    for ax, tag in zip(axes, tags):
        d = scen[tag]
        x = np.asarray(d["years"])
        ax.plot(x, d["y_scm"], c="black", ls="--", lw=2, alpha=0.8, zorder=10,
                label="SCM-projected")

        for key, colour, lbl in (("baseline", c_base, "Baseline Em."),
                                 ("optimized", c_opt, "Opt. All")):
            centre, lo, hi = _band(tag, key)
            ax.fill_between(x, lo, hi, color=colour, alpha=0.18, lw=0, zorder=1)
            ax.plot(x, centre, color=colour, lw=2, zorder=5, label=lbl)

        r2_o = _r2_summary(tag, "optimized")
        r2_b = _r2_summary(tag, "baseline")
        ax.set_title(
            f"{tag}\n"
            rf"$R^2$ opt {r2_o[0]:.3f} [{r2_o[1]:.3f}, {r2_o[2]:.3f}]  |  "
            rf"base {r2_b[0]:.3f}",
            fontsize=11)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylabel(r"$\Delta T$ [$^\circ$C]")
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    # -- pooled scatter, at the seed whose pooled optimized R^2 is nearest median
    ax = axes[len(tags)]
    truth = np.concatenate([np.asarray(scen[t]["y_scm"]).reshape(-1) for t in tags])

    def _pooled(seed, key):
        return np.concatenate([np.asarray(sweep[seed][key][t]["yhat"]).reshape(-1)
                               for t in tags])

    def _pooled_r2(seed, key):
        yh = _pooled(seed, key)
        ss_res = float(np.sum((truth - yh) ** 2))
        ss_tot = float(np.sum((truth - truth.mean()) ** 2))
        return 1.0 - ss_res / ss_tot

    pooled_o = np.array([_pooled_r2(s, "optimized") for s in seeds])
    pooled_b = np.array([_pooled_r2(s, "baseline") for s in seeds])
    med_seed = seeds[int(np.argmin(np.abs(pooled_o - np.median(pooled_o))))]

    lims = [min(truth.min(), _pooled(med_seed, "baseline").min()),
            max(truth.max(), _pooled(med_seed, "baseline").max())]
    ax.plot(lims, lims, c="black", ls="--", lw=1.5, alpha=0.8, zorder=1)
    ax.scatter(truth, _pooled(med_seed, "baseline"), s=6, color=c_base, alpha=0.35,
               lw=0, zorder=2,
               label=rf"Baseline, $R^2$={np.median(pooled_b):.3f} "
                     rf"[{np.percentile(pooled_b, 25):.3f}, {np.percentile(pooled_b, 75):.3f}]")
    ax.scatter(truth, _pooled(med_seed, "optimized"), s=6, color=c_opt, alpha=0.55,
               lw=0, zorder=3,
               label=rf"Opt. All, $R^2$={np.median(pooled_o):.3f} "
                     rf"[{np.percentile(pooled_o, 25):.3f}, {np.percentile(pooled_o, 75):.3f}]")
    ax.set_xlabel(r"SCM $\Delta T$ [$^\circ$C]")
    ax.set_ylabel(r"Emulated $\Delta T$ [$^\circ$C]")
    ax.set_title(f"All OOD scenarios pooled\n(scatter from seed {med_seed}, "
                 f"nearest median of {len(seeds)})", fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    for extra in axes[n:]:
        extra.axis("off")

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, -0.02))

    if save is not None:
        fig.savefig(FIGURES_DIR / f"{save}.pdf", bbox_inches="tight", transparent=True)
    return fig, axes


def plot_scm_mesm_fidelity_grid(panels, ncols=3, save=None, figsize=None):
    """Additional Major Point 4 - one panel per scenario, recalibrated SCM
    (mode='MESM') global-mean-temperature output against real MESM
    ground truth (data/MESM/emis_driven/zonal_data_mean/, area-weighted to a
    global mean). Companion figure to scripts/6d_scm_mesm_fidelity.py, which
    computes the NRMSE/R^2 this function only visualizes - no metric here is
    recomputed independently of that script's `nrmse_r2`; those numbers are
    reported in the companion LaTeX table instead of in-panel.

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

    Panels carry no title - each is labelled (a), (b), ... plus its scenario
    name, matching Figure 3's boxed-annotation convention (`_add_textbox`).
    All panels share one y-axis range, set from the min/max of every
    SCM/MESM trajectory actually drawn, so warming magnitudes are directly
    comparable across scenarios.
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
    # Extra headroom at the top for the (a)/(b)/... label box and (in the
    # first panel only) the legend, both of which sit near the top of the
    # panel (see the text()/legend() calls below).
    shared_ylim = [y_min - 0.05 * data_range, y_max + 0.18 * data_range]

    handles = [Line2D([0], [0], color=c_mesm, lw=2, label="MESM global average"),
               Line2D([0], [0], color=c_scm, ls="--", lw=2, label="MESM-calibrated SCM")]

    def _italic_scenario(name):
        # Mathtext-italicize each hyphen-separated piece individually (same
        # convention as panel_labels in plot_individual_effects_summary) so
        # the hyphens in e.g. "H-ext-OS" stay literal hyphens instead of
        # being absorbed into the math block and rendered as minus signs.
        return "-".join(rf"$\it{{{part}}}$" for part in name.split("-"))

    for i, (ax, p) in enumerate(zip(axes, panels)):
        ax.plot(p["x"], p["mesm"], color=c_mesm, lw=2, alpha=0.85, zorder=5,
                label="MESM global average")
        ax.plot(p["x"], p["scm"], color=c_scm, ls="--", lw=2, alpha=0.9, zorder=6,
                label="MESM-calibrated SCM")
        # (x, y) below: axes-fraction position of the (a)/(b)/... label box -
        # adjust these two numbers to nudge it (0,0)=bottom-left,
        # (1,1)=top-right of the panel. y=0.995 currently sits flush
        # against the panel's top edge, tracking constrained_layout
        # regardless of panel size.
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

    Panel (a) is the CO2-only single-forcing experiment (Figure 3); panels
    (b)-(f) are the multi-forcing experiment's (Figure 5) 5 individual
    forcing agents - CO2, CH4, N2O, Sulfur, BC, in the same order as Figure
    3's own panels. Labels sit boxed in each panel's top-left corner,
    matching this notebook's other multi-panel figures (e.g.
    plot_comparison_results, plot_individual_effects_summary).

    Each panel draws from its own fabiocrameri (cmcrameri) colormap
    (`agent_cmaps`, default one distinct sequential map per agent), sampled
    at 3 shades (`shade_fracs`, default q25=0.85/median=0.55/q75=0.25 of the
    map) rather than one fixed 3-color scheme shared across panels.

    `panel_width`/`panel_height` set the default figsize as
    (panel_width, panel_height * 6) - i.e. one row's worth of width times 6
    stacked rows. Pass `figsize` directly to override both.

    `title_suffix` is appended to every panel's boxed title (e.g. ", Opt.
    All") - the data alone doesn't say which training group produced it, so
    the caller states it explicitly rather than the label silently going
    stale.
    """
    if agent_units is None:
        agent_units = {"CO2": "Gt/yr", "CH4": "Mt/yr", "N2O": "Mt/yr", "Sulfur": "Mt/yr", "BC": "Mt/yr"}
    if agent_cmaps is None:
        # One distinct sequential cmcrameri map per panel, so no two panels
        # (including the two CO2 panels, single- vs multi-forcing) share a
        # color family.
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
