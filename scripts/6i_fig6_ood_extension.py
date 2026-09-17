#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage 6i - Figure 6 OOD extension: scenario construction (Phase A).

Figure 6's reported R^2 is computed entirely from in-objective evaluations
(3b_inverse_all_agents.py:129 passes DAMIP=True, GeoMIP=True, so eval_sets['All']
- the 'Opt. All' emulator's own bilevel objective - contains M_GHG, M_AER and
G6sulfur). This stage builds a roster of genuinely out-of-objective scenarios so
the figure can report an honest out-of-sample number alongside the in-sample one.

PHASE A (this file's `build` command) constructs every scenario's emissions and
its SCM ground-truth GMST, and nothing else. It reads no checkpoint and no
hyperparameter file, so it is unblocked by the in-flight multi-agent
hyperparameter regeneration. PHASE B (emulator evaluation, R^2 table) is gated on
that regeneration finishing and being approved, and lives in a later commit.

See SCENARIOS.md for the full provenance and citation of every scenario below.

Usage:
    python scripts/6i_fig6_ood_extension.py build      # construct + cache + verify
    python scripts/6i_fig6_ood_extension.py verify     # re-run checks on the cache
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import pickle
import numpy as np
import jax.numpy as jnp

import utils_inverse

AGENTS = ["CO2", "CH4", "N2O", "Sulfur", "BC"]
MODE = "FaIR"
EMA_WINDOWS = (5.0, 30.0, 100.0)
OUT_DIR = Path("data/SI_results/fig6_ood")
CACHE_PATH = OUT_DIR / "fig6_ood_scenarios.pkl"

RCMIP_CSV = "data/RCMIP/rcmip-emissions-annual-means-v5-1-0.csv"
RCMIP_REGION = "World"

# The repo's 'CO2' agent is CO2 FFI - fossil-fuel-and-industrial only
# (run_fair.load_scenarioMIP_CMIP7 maps 'CO2' -> the CSV's 'CO2 FFI' variable).
# RCMIP's bare 'Emissions|CO2' is the TOTAL including AFOLU; for ssp370 in 2020
# that is 44,808 vs. 40,933 Mt CO2/yr, a ~9.5% systematic overstatement. Use the
# Fossil-and-Industrial decomposition to match the repo's own convention.
RCMIP_VARIABLES = {
    "CO2": "Emissions|CO2|MAGICC Fossil and Industrial",
    "CH4": "Emissions|CH4",
    "N2O": "Emissions|N2O",
    "Sulfur": "Emissions|Sulfur",
    "BC": "Emissions|BC",
}
# Divisors converting RCMIP's native units to this repo's. N2O is a unit-family
# conversion only (kt -> Mt), NOT a molar-mass N2O->N conversion: the repo's own
# ScenarioMIP CSV stores N2O as Mt N2O/yr with no molar conversion applied
# anywhere in run_fair.load_scenarioMIP_CMIP7.
RCMIP_EXPECTED_UNIT = {
    "CO2": "Mt CO2/yr", "CH4": "Mt CH4/yr", "N2O": "kt N2O/yr",
    "Sulfur": "Mt SO2/yr", "BC": "Mt BC/yr",
}
RCMIP_UNIT_DIVISOR = {"CO2": 1000.0, "CH4": 1.0, "N2O": 1000.0, "Sulfur": 1.0, "BC": 1.0}

GHG_AGENTS = ("CO2", "CH4", "N2O")
AER_AGENTS = ("Sulfur", "BC")

# -- Harmonization (Gidden et al. 2018) ----------------------------------------
# RCMIP's scenarios were harmonized to their own historical inventory, which is
# not this repo's. Splicing them straight onto the repo's CMIP7 historical opens
# a 17-37% step at the 2023->2024 handoff. See SCENARIOS.md.
HARMONIZE_YEAR = 2023               # last year of the repo's CMIP7 historical
HARMONIZE_CONVERGENCE_YEAR = 2080   # aneris default; the year the correction vanishes

# ------------------------------------------------------------------
# Scenario registry
# ------------------------------------------------------------------
# `source`:
#   "cmip7_cross" - splice two of this repo's own CMIP7 scenarios (GHGs from one,
#                   aerosols from the other). Emissions come from
#                   data/FaIR/extensions_1750-2500.csv via
#                   run_fair.load_scenarioMIP_CMIP7; no external data.
#   "rcmip"       - read from data/RCMIP/rcmip-emissions-annual-means-v5-1-0.csv.
#   "rcmip_cross" - as "rcmip", but GHGs and aerosols from two different RCMIP
#                   scenarios.
# `prepend_historical` - False only for piControl-branched idealized experiments,
#   which have no historical period at all (matching how DECK is already handled,
#   utils_inverse.py:683).
# `harmonize` - True for every scenario read from RCMIP and spliced onto the
#   repo's own historical, since RCMIP harmonized to a different inventory.
#   False for "cmip7_cross" (both parents come from the same
#   extensions_1750-2500.csv as the historical, so they are consistent by
#   construction) and for the piControl-branched experiments (no historical to
#   be consistent with).

SCENARIOS = [
    # -- Group 1: CMIP7 RAMIP analogue (high-emission GHGs x lower-emission aerosols).
    # The CMIP7 set contains exactly FOUR distinct aerosol pathways, not seven:
    # {high-extension, high-overshoot}, {medium-extension, medium-overshoot},
    # {low}, {verylow, verylow-overshoot} are identical in Sulfur, BC and CH4
    # within each group (verified against extensions_1750-2500.csv; verylow and
    # verylow-overshoot differ ONLY in CO2 FFI). So the ladder uses one donor per
    # distinct pathway - M-ext / L-ext / VLLO-ext - and CMIP7_AEROSOL_DONORS below
    # guards against a future edit reintroducing a duplicate rung.
    # Truncated at 2150 to match the existing Figure 6 panels' x-range; beyond
    # ~2250 every low pathway has converged and the rungs become indistinguishable.
    dict(tag="H-ext-Maer", group="cmip7_ramip", source="cmip7_cross",
         mip="RAMIP analogue (CMIP7)", ghg_from="H-ext", aer_from="M-ext",
         year_end=2150, prepend_historical=True, harmonize=False),
    dict(tag="H-ext-Laer", group="cmip7_ramip", source="cmip7_cross",
         mip="RAMIP analogue (CMIP7)", ghg_from="H-ext", aer_from="L-ext",
         year_end=2150, prepend_historical=True, harmonize=False),
    dict(tag="H-ext-VLaer", group="cmip7_ramip", source="cmip7_cross",
         mip="RAMIP analogue (CMIP7)", ghg_from="H-ext", aer_from="VLLO-ext",
         year_end=2150, prepend_historical=True, harmonize=False),

    # -- Group 2: CO2-only idealized, piControl-branched (no historical)
    dict(tag="esm-bell-1000PgC", group="co2_only", source="rcmip",
         mip="ZECMIP / CDRMIP", scenario="esm-bell-1000PgC", model="idealised",
         year_range=(1850, 2049), prepend_historical=False, harmonize=False),
    dict(tag="esm-bell-2000PgC", group="co2_only", source="rcmip",
         mip="ZECMIP / CDRMIP", scenario="esm-bell-2000PgC", model="idealised",
         year_range=(1850, 2049), prepend_historical=False, harmonize=False),
    dict(tag="esm-pi-CO2pulse", group="co2_only", source="rcmip",
         mip="CDRMIP", scenario="esm-pi-CO2pulse", model="idealised",
         year_range=(1850, 2100), prepend_historical=False, harmonize=False),

    # -- Group 3: a different scenario generation
    dict(tag="rcp45", group="other_generation", source="rcmip",
         mip="CMIP5 RCP", scenario="rcp45", model="MiniCAM",
         year_range=(2024, 2100), prepend_historical=True, harmonize=True),
    dict(tag="rcp85", group="other_generation", source="rcmip",
         mip="CMIP5 RCP", scenario="rcp85", model="MESSAGE",
         year_range=(2024, 2100), prepend_historical=True, harmonize=True),
    dict(tag="ssp534-over", group="other_generation", source="rcmip",
         mip="ScenarioMIP (CMIP6)", scenario="ssp534-over", model="REMIND-MAGPIE",
         year_range=(2024, 2100), prepend_historical=True, harmonize=True),

    # -- Group 4: RAMIP's actual global Tier-1 experiment, CMIP6 native
    dict(tag="ssp370-126aer", group="cmip6_aerosol", source="rcmip_cross",
         mip="RAMIP (CMIP6)",
         ghg_scenario="ssp370", ghg_model="AIM/CGE",
         aer_scenario="ssp126", aer_model="IMAGE",
         year_range=(2024, 2100), prepend_historical=True, harmonize=True),

    # -- Group 5: the existing Stage 6c(b) scenario, with the CO2 FFI correction.
    # Shares the 'cmip6_aerosol' plotting group with ssp370-126aer: both are
    # CMIP6 SSP-based aerosol/NTCF perturbation experiments.
    dict(tag="ssp370-lowNTCF", group="cmip6_aerosol", source="rcmip",
         mip="AerChemMIP (CMIP6)", scenario="ssp370-lowNTCF-aerchemmip",
         model="AIM/CGE", year_range=(2024, 2100), prepend_historical=True, harmonize=True),
]

# In-sample references, plotted as context in the overview figure. Not evaluated.
REFERENCE_SCENARIOS = ["H-ext", "M", "L"]

# The Group 1 rungs, in order of increasing aerosol reduction at 2030.
CMIP7_AEROSOL_DONORS = ["M-ext", "L-ext", "VLLO-ext"]


# ------------------------------------------------------------------
# RCMIP access
# ------------------------------------------------------------------
def load_rcmip_species(scenario, model, agent):
    """One species' {year: value} from the RCMIP CSV, in this repo's units.

    Asserts the CSV's own 'Unit' column against RCMIP_EXPECTED_UNIT before
    applying a divisor, rather than assuming the unit family is uniform across
    scenarios - the divisors were originally verified for ssp370-lowNTCF only.
    """
    variable = RCMIP_VARIABLES[agent]
    with open(RCMIP_CSV, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        year_cols = [(i, int(h)) for i, h in enumerate(header) if h.isdigit()]
        unit_col = header.index("Unit")
        for row in r:
            if (row[0] == model and row[1] == scenario
                    and row[2] == RCMIP_REGION and row[3] == variable):
                unit = row[unit_col].strip()
                expected = RCMIP_EXPECTED_UNIT[agent]
                if unit != expected:
                    raise ValueError(
                        f"Unit mismatch for {scenario}/{model}/{variable}: CSV says "
                        f"'{unit}', expected '{expected}'. Refusing to apply a divisor "
                        f"calibrated for a different unit family."
                    )
                vals = {yr: float(row[i]) for (i, yr) in year_cols if row[i].strip() != ""}
                if not vals:
                    raise ValueError(f"No non-empty year values for {scenario}/{variable}")
                return vals
    raise ValueError(f"No RCMIP row for {model}/{scenario}/{RCMIP_REGION}/{variable}")


def interp_rcmip(scenario, model, agent, years):
    """Linearly interpolate one species onto an annual calendar-year grid.

    RCMIP reports at irregular intervals (annual to 2014, then 5- and 10-yearly);
    linear interpolation is the protocol's own expected treatment
    (Nicholls et al. 2020, Section 2).
    """
    raw = load_rcmip_species(scenario, model, agent)
    yrs = np.array(sorted(raw.keys()), dtype=float)
    vals = np.array([raw[y] for y in sorted(raw.keys())]) / RCMIP_UNIT_DIVISOR[agent]
    return np.interp(np.asarray(years, dtype=float), yrs, vals).astype(np.float32)


# ------------------------------------------------------------------
# Harmonization
# ------------------------------------------------------------------
# Gidden, M.J., Fujimori, S., van den Berg, M., Klein, D., Smith, S.J.,
# van Vuuren, D.P. & Riahi, K. (2018). "A methodology and implementation of
# automated emissions harmonization for use in Integrated Assessment Models."
# Environmental Modelling & Software 105, 187-200.
# https://doi.org/10.1016/j.envsoft.2018.04.002
#
# This is the methodology paper behind `aneris`, the tool that produced the
# harmonized CMIP6 SSP emissions and that was used in the IPCC AR6 WGIII
# workflow - i.e. the same procedure the source dataset's own maintainers apply.
# (RCMIP's `ssp370-lowNTCF-gidden` variant is named for this work.)
#
# THE PROBLEM. Every scenario in Group 3/4/5 is read from RCMIP, which harmonized
# each pathway to ITS OWN historical inventory. This repo's historical comes from
# data/FaIR/extensions_1750-2500.csv (CMIP7). The two inventories disagree at
# 2023 by -12% to +39% depending on the species, so splicing an RCMIP future
# directly onto the repo's historical opens a step change at the 2023->2024
# handoff that the SCM reads as a real emissions shock.
#
# THE METHOD ("reduce" convergence, Gidden et al. 2018 Section 2.2). Force the
# model pathway to equal the historical inventory at the harmonization year, then
# decay that correction linearly to zero at a convergence year, after which the
# pathway is exactly the scenario as published:
#
#     f(t)  = 1 - (t - t_harm) / (t_conv - t_harm),  clipped to [0, 1]
#     ratio  method:  m'(t) = m(t) * [1 + (r - 1) * f(t)],  r = h(t_harm)/m(t_harm)
#     offset method:  m'(t) = m(t) + o * f(t),              o = h(t_harm)-m(t_harm)
#
# This is the paper's central design choice: fix the near-term inconsistency
# without overwriting the scenario's long-term signal. Both the near-term level
# and the end-of-century target are preserved.
#
# METHOD SELECTION follows the paper's default decision tree. The tree's full
# form handles cases that arise for sectoral/regional IAM output and cannot arise
# here (zero historical, all-zero model, near-zero base year, the land-use CO2
# coefficient-of-variation branch); `choose_harmonization_method` raises rather
# than guessing if the data ever reaches one of them. For this roster it reduces
# to two branches, and the reduction was verified against the data, not assumed:
#
#   - ratio, converging 2080 (aneris `reduce_ratio_2080`, the package default) -
#     24 of the 25 (scenario, species) pairs. All ratios fall in [0.72, 1.14].
#   - offset, converging 2080 (aneris `reduce_offset_2080`) - ssp534-over CO2
#     alone, which goes net-negative from 2066. A ratio is unstable across a sign
#     change: it scales a negative value further negative, and is undefined at
#     the crossing. The paper switches to an additive offset for exactly this.
#
# CONVERGENCE YEAR. 2080 is the aneris default and the value used for AR6 WGIII
# (SR1.5 used 2050). It sits inside every scenario's 2024-2100 span, so each
# pathway rejoins its published trajectory with two decades to spare.

def _convergence_factors(years, harmonize_year, convergence_year):
    """f(t): 1 at the harmonization year, 0 from the convergence year onward."""
    span = float(convergence_year - harmonize_year)
    f = 1.0 - (np.asarray(years, dtype=float) - harmonize_year) / span
    return np.clip(f, 0.0, 1.0)


def harmonize_ratio(model, years, ratio,
                    harmonize_year=HARMONIZE_YEAR,
                    convergence_year=HARMONIZE_CONVERGENCE_YEAR):
    """aneris `reduce_ratio`: multiplicative correction decaying to 1."""
    f = _convergence_factors(years, harmonize_year, convergence_year)
    return (np.asarray(model, dtype=float) * (1.0 + (ratio - 1.0) * f)).astype(np.float32)


def harmonize_offset(model, years, offset,
                     harmonize_year=HARMONIZE_YEAR,
                     convergence_year=HARMONIZE_CONVERGENCE_YEAR):
    """aneris `reduce_offset`: additive correction decaying to 0."""
    f = _convergence_factors(years, harmonize_year, convergence_year)
    return (np.asarray(model, dtype=float) + offset * f).astype(np.float32)


def choose_harmonization_method(hist_value, model_value, model_future,
                                convergence_year=HARMONIZE_CONVERGENCE_YEAR,
                                years=None):
    """The paper's default decision tree, restricted to the branches this data reaches.

    Raises on any branch that would require a judgement call not verified against
    this roster, rather than silently picking a method.
    """
    if hist_value == 0:
        raise ValueError(
            "history is zero at the harmonization year (aneris `hist_zero`). Not "
            "reachable for any species here; no verified handling."
        )
    if np.allclose(model_future, 0.0):
        raise ValueError(
            "model pathway is identically zero (aneris `model_zero`). Not reachable "
            "for any species here; no verified handling."
        )
    if abs(model_value) < 1e-9 or abs(model_value) < 1e-4 * abs(hist_value):
        raise ValueError(
            f"model is ~zero at the harmonization year ({model_value:.4g} vs history "
            f"{hist_value:.4g}); aneris switches to an offset method here, but no "
            f"species in this roster reaches it, so the choice is unverified."
        )

    # A ratio is unstable across a sign change and undefined at the crossing;
    # the paper's tree switches to an additive offset. Only the span up to the
    # convergence year matters - past it the correction is already zero.
    future = np.asarray(model_future, dtype=float)
    if years is not None:
        future = future[np.asarray(years) <= convergence_year]
    if future.size and float(np.min(future)) <= 0.0:
        return "reduce_offset"
    return "reduce_ratio"


def _model_value_at_harmonize_year(entry, eval_sets):
    """The scenario's own value at HARMONIZE_YEAR, via the same source dispatch."""
    probe = {**entry, "year_range": (HARMONIZE_YEAR, HARMONIZE_YEAR)}
    _yrs, emis = build_emissions(probe, eval_sets)
    return {a: float(emis[a][0]) for a in AGENTS}


def harmonize_emissions(entry, years, emis, historical, eval_sets):
    """-> ({agent: harmonized (T,)}, {agent: provenance dict}).

    A no-op returning ({}, {}) as the metadata for entries with harmonize=False.
    """
    if not entry.get("harmonize", False):
        return emis, {}

    if int(historical_year_end(historical)) != HARMONIZE_YEAR:
        raise ValueError(
            f"historical ends at {historical_year_end(historical)}, not "
            f"{HARMONIZE_YEAR}; the harmonization year must be a year both the "
            f"historical inventory and the scenario report."
        )

    model_at_harm = _model_value_at_harmonize_year(entry, eval_sets)
    out, meta = {}, {}
    for a in AGENTS:
        h = float(historical[a][-1])
        m = model_at_harm[a]
        method = choose_harmonization_method(h, m, emis[a], years=years)
        if method == "reduce_ratio":
            corr = h / m
            harmonized = harmonize_ratio(emis[a], years, corr)
        else:
            corr = h - m
            harmonized = harmonize_offset(emis[a], years, corr)
        out[a] = harmonized
        meta[a] = dict(
            method=method, harmonize_year=HARMONIZE_YEAR,
            convergence_year=HARMONIZE_CONVERGENCE_YEAR,
            history_value=h, model_value=m,
            ratio=(h / m) if method == "reduce_ratio" else float("nan"),
            offset=(h - m) if method == "reduce_offset" else float("nan"),
            raw_first=float(emis[a][0]), harmonized_first=float(harmonized[0]),
        )
    return out, meta


def historical_year_end(historical):
    """The repo's historical runs 1750..N; return its final calendar year."""
    return 1750 + len(historical[AGENTS[0]]) - 1


# ------------------------------------------------------------------
# Scenario construction
# ------------------------------------------------------------------
def _cmip7_agent_arrays(eval_sets, tag):
    """{agent: (T,) float32} for one of this repo's own CMIP7 scenarios."""
    for key in ("Tier 1", "Tier 2"):
        if tag in eval_sets[key]:
            _years, emis = utils_inverse.extract_years_and_emis(
                eval_sets[key][tag], agents=AGENTS
            )
            return {a: np.asarray(emis[a], dtype=np.float32) for a in AGENTS}
    raise KeyError(f"CMIP7 scenario '{tag}' not found in Tier 1 or Tier 2")


def build_emissions(entry, eval_sets):
    """-> (years_calendar (T,), {agent: (T,) float32}).

    Calendar years are returned for plotting and for RCMIP interpolation only.
    They are NOT what simulate_targets_gmst gets - see build_features.
    """
    src = entry["source"]

    if src == "cmip7_cross":
        ghg = _cmip7_agent_arrays(eval_sets, entry["ghg_from"])
        aer = _cmip7_agent_arrays(eval_sets, entry["aer_from"])
        n_ghg = len(ghg["CO2"])
        n_aer = len(aer["Sulfur"])
        if n_ghg != n_aer:
            raise ValueError(
                f"{entry['tag']}: cannot splice '{entry['ghg_from']}' (T={n_ghg}) with "
                f"'{entry['aer_from']}' (T={n_aer}) - lengths differ."
            )
        emis = {a: (ghg[a] if a in GHG_AGENTS else aer[a]).copy() for a in AGENTS}
        # CMIP7 future segments start the year after historical ends (2024).
        years = np.arange(2024, 2024 + n_ghg)
        year_end = entry.get("year_end")
        if year_end is not None:
            keep = years <= year_end
            years = years[keep]
            emis = {a: v[keep] for a, v in emis.items()}
        return years, emis

    y0, y1 = entry["year_range"]
    years = np.arange(y0, y1 + 1)

    if src == "rcmip":
        emis = {a: interp_rcmip(entry["scenario"], entry["model"], a, years) for a in AGENTS}
        return years, emis

    if src == "rcmip_cross":
        emis = {}
        for a in AGENTS:
            scen = entry["ghg_scenario"] if a in GHG_AGENTS else entry["aer_scenario"]
            model = entry["ghg_model"] if a in GHG_AGENTS else entry["aer_model"]
            emis[a] = interp_rcmip(scen, model, a, years)
        return years, emis

    raise ValueError(f"Unknown source '{src}' for {entry['tag']}")


def build_features(entry, eval_sets):
    """-> dict with years / emissions / features / SCM ground truth.

    Two traps this deliberately avoids, both previously found the hard way
    (REVISIONS.md session log, 2026-08-12):

    1. Every scenario in this codebase carries a 0-BASED RELATIVE year index,
       not calendar years. simulate_targets_gmst's _make_contiguous_years relies
       on that to rebase the future segment onto the end of historical. Handing
       it calendar years (2024, ...) makes it see 2024 > 273 (historical's own
       0-based length), conclude the segments are already contiguous, and splice
       in a ~1750-year phantom gap.
    2. build_dataset_from_runfair_dict decides whether to prepend historical via
       a hardcoded scenario-name allowlist (utils_inverse.py:321) with no entry
       for any scenario here - routing through it silently drops historical
       context entirely. Historical is prepended explicitly instead.
    """
    years_cal, emis_curr = build_emissions(entry, eval_sets)
    emis_raw = {a: np.asarray(v).copy() for a, v in emis_curr.items()}

    if entry["prepend_historical"]:
        years_hist, emis_hist = utils_inverse.extract_years_and_emis(
            eval_sets["Tier 1"]["historical"], agents=AGENTS
        )
        # Reconcile the scenario's inventory with this repo's before splicing
        # (Gidden et al. 2018). No-op unless entry['harmonize'] is True.
        emis_curr, harm_meta = harmonize_emissions(
            entry, years_cal, emis_curr,
            {a: np.asarray(emis_hist[a]) for a in AGENTS}, eval_sets,
        )
    else:
        harm_meta = {}
        # piControl-branched idealized experiment: no historical period at all,
        # matching how DECK is handled (utils_inverse.py:683).
        years_hist, emis_hist = None, None

    years_rel = jnp.arange(len(years_cal), dtype=jnp.float32)

    X = utils_inverse.make_features_emissions_generic(
        emis_curr_dict=emis_curr, emis_hist_dict=emis_hist, agents=AGENTS,
        ema_windows_years=EMA_WINDOWS, dt_years=1.0, zero_fill_missing=True,
    )
    y = utils_inverse.simulate_targets_gmst(
        years_curr=years_rel, emis_curr_dict=emis_curr,
        years_hist=years_hist, emis_hist_dict=emis_hist, agents=AGENTS, mode=MODE,
    )
    n = min(int(X.shape[0]), int(y.shape[0]))

    return dict(
        tag=entry["tag"], group=entry["group"], mip=entry["mip"],
        source=entry["source"], prepend_historical=entry["prepend_historical"],
        harmonized=bool(entry.get("harmonize", False)),
        harmonization=harm_meta,
        years=np.asarray(years_cal[:n]),
        emis={a: np.asarray(v[:n]) for a, v in emis_curr.items()},
        emis_raw={a: np.asarray(v[:n]) for a, v in emis_raw.items()},
        X=np.asarray(X[:n]), y_scm=np.asarray(y[:n]),
    )


def build_reference(tag, eval_sets):
    """An in-sample CMIP7 scenario, built the same way, for plot context."""
    entry = dict(tag=tag, group="reference", source="cmip7_cross",
                 mip="ScenarioMIP (CMIP7)", ghg_from=tag, aer_from=tag,
                 prepend_historical=True)
    return build_features(entry, eval_sets)


# ------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------
def verify(cache):
    """Plan verification items 1 and 4-7. Returns a list of failure strings."""
    failures = []
    scen = cache["scenarios"]

    for tag, d in scen.items():
        y = d["y_scm"]

        # Item 6 - physical sanity
        if not (-1.5 <= float(np.min(y)) and float(np.max(y)) <= 8.0):
            failures.append(f"{tag}: GMST out of plausible range "
                            f"[{float(np.min(y)):.2f}, {float(np.max(y)):.2f}] K")
        if not np.all(np.isfinite(y)):
            failures.append(f"{tag}: non-finite GMST")

        # Item 5 - no-historical scenarios start from a clean slate
        if not d["prepend_historical"]:
            cum_col = d["X"][0, AGENTS.index("CO2") * 5 + 4]
            if abs(float(cum_col)) > 1e-6:
                failures.append(f"{tag}: prepend_historical=False but cumulative-CO2 "
                                f"feature starts at {float(cum_col):.4g}, not 0")

        # Item 4 - continuity at the 2023->2024 handoff.
        # A harmonized scenario is pinned to the historical inventory at 2023, so
        # the only step left at 2024 is one year of genuine change plus one year
        # of convergence decay (1/57 of the original mismatch): a few percent, not
        # the 17-37% the raw RCMIP pathways showed.
        #
        # The unharmonized CMIP7 crosses get a looser bound because their step is
        # INHERITED, not introduced: VLLO-ext's own first future year is already
        # -12.9% in Sulfur and -11.3% in CO2 against the repo's historical, an
        # aggressive near-term mitigation built into the scenario as this repo
        # ships it and present in the in-sample set. The splice adds nothing to it
        # (test_ood_scenarios asserts the cross equals its parents value-for-value),
        # so harmonizing here would be overwriting the repo's own source data.
        if d["prepend_historical"] and int(d["years"][0]) == 2024:
            hist = cache["historical"]
            tol = 0.05 if d.get("harmonized") else 0.15
            for a in AGENTS:
                last_hist, first_fut = float(hist[a][-1]), float(d["emis"][a][0])
                denom = max(abs(last_hist), 1e-3)
                if abs(first_fut - last_hist) / denom > tol:
                    failures.append(
                        f"{tag}/{a}: discontinuity at 2023->2024 "
                        f"({last_hist:.3g} -> {first_fut:.3g}, "
                        f"{100 * (first_fut - last_hist) / denom:+.1f}%, tol {100 * tol:.0f}%)"
                    )

        # Harmonization did something, and undid itself on schedule.
        if d.get("harmonized"):
            if not d["harmonization"]:
                failures.append(f"{tag}: harmonize=True but no harmonization recorded")
            for a in AGENTS:
                if np.allclose(d["emis"][a], d["emis_raw"][a]):
                    failures.append(f"{tag}/{a}: harmonized series is identical to the raw one")
                after = d["years"] >= HARMONIZE_CONVERGENCE_YEAR
                if after.any() and not np.allclose(
                    d["emis"][a][after], d["emis_raw"][a][after], rtol=1e-5, atol=1e-6
                ):
                    failures.append(
                        f"{tag}/{a}: harmonized series has not rejoined the published "
                        f"pathway by {HARMONIZE_CONVERGENCE_YEAR}"
                    )

    # Item 6 - the bell scenarios must peak and then decline
    for tag in ("esm-bell-1000PgC", "esm-bell-2000PgC"):
        if tag in scen:
            co2 = scen[tag]["emis"]["CO2"]
            k = int(np.argmax(co2))
            if not (0 < k < len(co2) - 1 and co2[-1] < 0.1 * co2[k]):
                failures.append(f"{tag}: CO2 does not peak and return toward zero "
                                f"(argmax={k}/{len(co2)}, end={co2[-1]:.3g}, peak={co2[k]:.3g})")

    # Item 7 - the CMIP7 cross must behave physically against its GHG parent.
    #
    # The invariant is POINTWISE, not a fixed ordering of the rungs: the CMIP7
    # 'low' and 'verylow' aerosol pathways cross over around 2060 (low starts
    # higher, ends lower), so the rungs genuinely reorder mid-century. What must
    # hold at every year is that less aerosol => more warming, at identical GHGs.
    rungs = [e["tag"] for e in SCENARIOS if e["group"] == "cmip7_ramip"]
    if all(t in scen for t in rungs) and "H-ext" in cache["references"]:
        ref = cache["references"]["H-ext"]
        n = len(scen[rungs[0]]["years"])

        # (a) every rung strictly warmer than H-ext, since every rung has
        #     strictly less aerosol than H-ext throughout.
        for t in rungs:
            d_su = scen[t]["emis"]["Sulfur"] - ref["emis"]["Sulfur"][:n]
            d_T = scen[t]["y_scm"] - ref["y_scm"][:n]
            if np.any(d_su > 1e-6):
                failures.append(f"{t}: aerosol exceeds H-ext somewhere - not a reduction")
            elif np.any(d_T[10:] <= 0):
                failures.append(f"{t}: not warmer than H-ext at every year past 2034 "
                                f"(min delta {float(np.min(d_T[10:])):+.4f} K)")

        # (b) at each year, rank by aerosol and check temperature ranks inversely.
        su = np.stack([scen[t]["emis"]["Sulfur"] for t in rungs])
        tt = np.stack([scen[t]["y_scm"] for t in rungs])
        # Compare only where the pathways are actually separated, and skip the
        # first year: the SCM has not yet responded to that year's emissions, so
        # all rungs are bit-identical there and a strict comparison sees a tie.
        sep = (su.max(axis=0) - su.min(axis=0)) > 1.0
        sep[0] = False
        if sep.sum() > 0:
            lo, hi = np.argmin(su[:, sep], axis=0), np.argmax(su[:, sep], axis=0)
            cols = np.arange(sep.sum())
            bad = tt[:, sep][lo, cols] <= tt[:, sep][hi, cols]
            if bad.any():
                failures.append(
                    f"CMIP7 cross: lowest-aerosol rung is not warmest in "
                    f"{int(bad.sum())}/{int(sep.sum())} separated years"
                )

        # (c) guard against a future edit reintroducing a duplicate rung.
        for i in range(len(rungs)):
            for j in range(i + 1, len(rungs)):
                if np.allclose(scen[rungs[i]]["emis"]["Sulfur"],
                               scen[rungs[j]]["emis"]["Sulfur"]):
                    failures.append(
                        f"CMIP7 cross: '{rungs[i]}' and '{rungs[j]}' have identical "
                        f"Sulfur - the CMIP7 set has only 4 distinct aerosol pathways "
                        f"(see CMIP7_AEROSOL_DONORS), so this rung is redundant."
                    )

    return failures


# ------------------------------------------------------------------
def build():
    print("Loading CMIP7 scenario emissions (no checkpoints, no hyperparameters)...")
    eval_sets, *_ = utils_inverse.generate_eval_data(
        tuple(AGENTS), DECK=True, CS3=True, DAMIP=True, GeoMIP=True
    )

    _yh, emis_hist = utils_inverse.extract_years_and_emis(
        eval_sets["Tier 1"]["historical"], agents=AGENTS
    )
    historical = {a: np.asarray(emis_hist[a]) for a in AGENTS}

    scenarios = {}
    for entry in SCENARIOS:
        print(f"  building {entry['tag']:20s} ({entry['mip']})")
        scenarios[entry["tag"]] = build_features(entry, eval_sets)

    references = {}
    for tag in REFERENCE_SCENARIOS:
        print(f"  reference {tag}")
        references[tag] = build_reference(tag, eval_sets)

    cache = dict(agents=AGENTS, historical=historical,
                 scenarios=scenarios, references=references)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"wrote {CACHE_PATH}")

    write_harmonization_table(cache)
    return cache


def write_harmonization_table(cache):
    """Per-species harmonization provenance, for the SI."""
    rows = []
    for tag, d in cache["scenarios"].items():
        for a in AGENTS:
            m = d.get("harmonization", {}).get(a)
            if m is None:
                continue
            raw, harm = m["raw_first"], m["harmonized_first"]
            rows.append(dict(
                scenario=tag, agent=a, method=m["method"],
                harmonize_year=m["harmonize_year"],
                convergence_year=m["convergence_year"],
                history_2023=round(m["history_value"], 6),
                scenario_2023=round(m["model_value"], 6),
                ratio=round(m["ratio"], 6), offset=round(m["offset"], 6),
                raw_2024=round(raw, 6), harmonized_2024=round(harm, 6),
                pct_change_2024=round(100.0 * (harm - raw) / raw, 3) if raw else float("nan"),
            ))
    path = OUT_DIR / "fig6_ood_harmonization.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} species-scenario pairs)")
    return rows


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        cache = build()
    elif cmd == "verify":
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
    else:
        raise SystemExit(f"unknown command '{cmd}' (expected 'build' or 'verify')")

    failures = verify(cache)
    print()
    for tag, d in cache["scenarios"].items():
        print(f"  {tag:20s} n={len(d['years']):4d}  "
              f"{int(d['years'][0])}-{int(d['years'][-1])}  "
              f"GMST {float(np.min(d['y_scm'])):+.2f} to {float(np.max(d['y_scm'])):+.2f} K")
    print()
    if failures:
        print(f"VERIFICATION FAILED ({len(failures)}):")
        for f_ in failures:
            print(f"  - {f_}")
        raise SystemExit(1)
    print("all verification checks passed")


if __name__ == "__main__":
    main()
