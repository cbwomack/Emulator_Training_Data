# Imports

## Standard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

## FaIR
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise

## Misc.
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional
from scipy.fft import rfft, rfftfreq

# ----------------------------
# Constants and Configurations
# ----------------------------

DEFAULT_SCENARIO = 'ssp245'
EBM_CONFIG = 'data/FaIR/4xCO2_cummins_ebm3.csv'
VOLCANIC_FORCING = 'data/FaIR/volcanic_ERF_monthly_175001-201912.csv'

SPECIES = ['CO2','CH4','N2O','Sulfur','BC','Aerosol-radiation interactions','Aerosol-cloud interactions']
_DF_EBM = pd.read_csv(EBM_CONFIG)
_PROPERTIES = read_properties()[1]
DEFAULT_ESMs = _DF_EBM['model'].unique()

def _ebm_config_names(models: Iterable[str]) -> list:
    """Return list like ['ModelA_r1','ModelA_r2', ...] from the EBM table."""
    names = []
    for model in models:
        sub = _DF_EBM[_DF_EBM["model"] == model]
        for run in sub["run"]:
            names.append(f"{model}_{run}")
    return names

def _apply_ebm_configs(f: FAIR, configs: Iterable[str], stochastic: bool=False) -> None:
    """Fill FaIR climate_configs from the EBM CSV for all configs."""
    seed = 1355763
    for cfg in configs:
        model, run = cfg.split("_")
        cond = (_DF_EBM["model"] == model) & (_DF_EBM["run"] == run)

        # ocean heat capacity & transfer (expects arrays of length 3)
        fill(f.climate_configs["ocean_heat_capacity"], _DF_EBM.loc[cond, "C1":"C3"].values.squeeze(), config=cfg)
        fill(f.climate_configs["ocean_heat_transfer"], _DF_EBM.loc[cond, "kappa1":"kappa3"].values.squeeze(), config=cfg)

        # scalars
        fill(f.climate_configs["deep_ocean_efficacy"],     _DF_EBM.loc[cond, "epsilon"].values[0], config=cfg)
        fill(f.climate_configs["gamma_autocorrelation"],   _DF_EBM.loc[cond, "gamma"].values[0],   config=cfg)
        fill(f.climate_configs["sigma_eta"],               _DF_EBM.loc[cond, "sigma_eta"].values[0], config=cfg)
        fill(f.climate_configs["sigma_xi"],                _DF_EBM.loc[cond, "sigma_xi"].values[0],  config=cfg)

        # stochastic controls
        fill(f.climate_configs["stochastic_run"], stochastic, config=cfg)
        fill(f.climate_configs["use_seed"], stochastic, config=cfg)
        fill(f.climate_configs["seed"], seed, config=cfg)
        seed += 399

def _species_properties(input_mode: str) -> Dict[str, dict]:
    """Copy baseline species properties and set CO2 input_mode."""
    props = {s: dict(_PROPERTIES[s]) for s in SPECIES}
    props["CO2"]["input_mode"] = input_mode
    props["CH4"]["input_mode"] = input_mode
    props["N2O"]["input_mode"] = input_mode
    props["Aerosol-radiation interactions"]["input_mode"] = "calculated"
    props["Aerosol-cloud interactions"]["input_mode"]    = "calculated"

    return props

# ---------------------
# Functions to run FaIR
# ---------------------

def build_fair(
    start: int,
    stop: int,
    input_mode: str = "emissions",
    default_scenario: str = DEFAULT_SCENARIO,
    esms: Iterable[str] = DEFAULT_ESMs,
    scenario_name: str = "custom",
    ) -> FAIR:
    """
    Create a FaIR model with EBM configs loaded and species ready for custom emissions.
    """
    f = FAIR(ghg_method="meinshausen2020", ch4_method="thornhill2021")

    # define time and configs
    f.define_time(start, stop + 1, 1)  # keep consistent everywhere
    configs = _ebm_config_names(esms)
    f.define_configs(configs)

    # define species
    props = _species_properties(input_mode)
    f.define_species(SPECIES, props)

    # pull baseline info from RCMIP using any valid scenario (for shapes/initials)
    f.define_scenarios([default_scenario])
    f.allocate()
    f.fill_from_rcmip()
    f.fill_species_configs()

    # initialise state
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    # rename scenario coordinates to 'custom'
    f.define_scenarios([scenario_name])
    f.emissions            = f.emissions.assign_coords(scenario=f.scenarios)
    f.cumulative_emissions = f.cumulative_emissions.assign_coords(scenario=f.scenarios)
    f.concentration        = f.concentration.assign_coords(scenario=f.scenarios)
    f.forcing              = f.forcing.assign_coords(scenario=f.scenarios)
    f.temperature          = f.temperature.assign_coords(scenario=f.scenarios)
    f.species_configs      = f.species_configs.assign_coords(scenario=f.scenarios)

    f.emissions[:] = 0

    # apply EBM configs
    _apply_ebm_configs(f, configs, stochastic=False)

    return f

def reset_state(f: FAIR) -> None:
    """Zero mutable state so you can re-run without re-instantiating."""
    f.emissions[:] = 0
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    for da in (f.forcing, f.temperature, f.cumulative_emissions, f.airborne_emissions):
        da.loc[...] = 0

def set_emissions(
    f: FAIR,
    years: Iterable[int],
    emis_by_agent: Dict[str, np.ndarray],
    ) -> None:
    """
    Provide per-agent emissions (length == len(years)). Missing agents default to zeros.
    Keys may be 'SO2' or 'Sulfur'; both map to FaIR's 'Sulfur'.
    """
    yrs = list(years)
    n = len(yrs)
    ncfg = len(f.configs)

    # validate lengths & build a dense dict over SPECIES
    dense = {}
    for a in SPECIES:
        dense[a] = np.zeros(n)

    for k, v in emis_by_agent.items():
        v = np.asarray(v)
        if v.shape[0] != n:
            raise ValueError(f"Emissions for {k} has length {v.shape[0]} but years has length {n}.")
        dense[k] = v

    # write into f.emissions (timepoints × scenario × config)
    # broadcast across configs once to avoid repeated assignments
    tstart, tstop = min(yrs), max(yrs)
    for specie, vec in dense.items():
        arr = np.tile(vec, (ncfg, 1)).T[:, None, :]  # (time, 1, config)
        f.emissions.sel(specie=specie, timepoints=slice(tstart, tstop + 1))[:] = arr

def run_fair(f: FAIR, years: Iterable[int], layer: int = 0) -> Tuple[list, np.ndarray, np.ndarray, float]:
    """
    Run FaIR and return (t, T_ens, T_mean, ECS).
    """
    tmin, tmax = min(years), max(years)
    f.run(progress=False)

    ECS = float(np.round(f.ebms.ecs.mean().item(), 2))
    t = list(range(tmin, tmax + 1))

    # ensemble over configs at surface layer
    T = f.temperature.sel(timebounds=slice(tmin, tmax)).loc[dict(layer=layer)].values[:,0,:]
    T_mean = T.mean(axis=1)
    return t, T, T_mean, ECS

# ----------------------
# Misc. Helper Functions
# ----------------------

def impulse_profile(n_steps: int, dt: float, magnitude: float = 1.0) -> np.ndarray:
    v = np.zeros(n_steps)
    v[0] = magnitude / dt   # integrates to 'magnitude'
    return v

def white_noise_profile(n_steps: int, sigma: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=n_steps)

def ensure_all_agents(d, n_steps):
    for a in SPECIES:
        d.setdefault(a, np.zeros(n_steps))
    return d

# -------------
# CMIP7 Helpers
# -------------

def load_scenarioMIP_CMIP7(agents: list) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns dict[scenario][agent] -> emissions array
    """

    # Define CMIP7 scenarios and extensions
    scenarios_tier1  = ['high-extension','medium-extension','medium-overshoot',
                        'low','verylow','verylow-overshoot']
    scenarios_tier2  = ['high-overshoot','medium-extension','medium-overshoot',
                        'low','verylow-overshoot']
    scenarios_all    = scenarios_tier1 + scenarios_tier2

    tags_tier1 = ['H-ext','M','ML','L','VLLO-ext','VLHO']
    tags_tier2 = ['H-ext-OS','M-ext','ML-ext','L-ext','VLHO-ext']
    tags_all   = tags_tier1 + tags_tier2
    data_path  = "data/FaIR/extensions_1750-2500.csv"
    emis_df    = pd.read_csv(data_path)

    # Indices for emissions dataset
    n_col_skip = 5
    n_years_base, n_years_ext = 127, 477
    emis_dict_tier1, emis_dict_tier2 = {}, {}

    # Historical emissions
    n_years_hist = 274
    emis_dict_tier1['historical'], emis_dict_tier2['historical'] = {}, {}
    for a in agents:
        if a == 'CO2':
            a_full = 'CO2 FFI'
        else:
            a_full = a

        historical_emis = emis_df[(emis_df["scenario"] == scenarios_all[0]) & (emis_df["variable"] == a_full)].iloc[:, n_col_skip:n_col_skip + n_years_hist].to_numpy().reshape(-1)
        emis_dict_tier1['historical'][a] = historical_emis
        emis_dict_tier2['historical'][a] = historical_emis

    # All other scenarios
    for scen, tag in zip(scenarios_all, tags_all):
        if tag in tags_tier1:
            emis_dict_tier1[tag] = {}
        else:
            emis_dict_tier2[tag] = {}

        if 'ext' in tag:
            n_years = n_years_ext
        else:
            n_years = n_years_base

        for a in agents:
            if a == 'CO2':
                a_full = 'CO2 FFI'
            else:
                a_full = a

            vals = emis_df[(emis_df["scenario"] == scen) & (emis_df["variable"] == a_full)].iloc[:, n_col_skip + n_years_hist:n_col_skip + n_years_hist + n_years].to_numpy().reshape(-1)

            #if 'ext' not in tag:
                #print(n_col_skip + n_years_hist + n_years)

            if tag in tags_tier1:
                emis_dict_tier1[tag][a] = vals
            else:
                emis_dict_tier2[tag][a] = vals

    return emis_dict_tier1, emis_dict_tier2

def get_cmip7_delT(emis_dict, scenarios, agents, DECK=False):
    delT_dict = {}
    for scen in scenarios:
        if not DECK:
            if scen == 'historical':
                start, stop = 1750, 2024
                ind1, ind2 = start - start, stop - 1750
            elif 'ext' not in scen:
                start, stop = 1750, 2151
                ind1, ind2 = 2024 - start, 2151 - start
            else:
                start, stop = 1750, 2501
                ind1, ind2 = 2024 - start, stop - start
        else:
            if 'abrupt' in scen:
                start, stop = 1750, 2051
            elif '1pct' in scen:
                start, stop = 1750, 1901
            ind1, ind2 = start - start, stop - start

        years = np.arange(start, stop)
        n_steps = len(years)
        nan_force = np.zeros(n_steps)

        f = build_fair(start, stop, input_mode="emissions")
        emis = {a: nan_force for a in agents}

        for a in agents:
            if scen != 'historical' and not DECK:
                emis[a] = np.concatenate((emis_dict['historical'][a], emis_dict[scen][a]), axis=None)
            else:
                emis[a] = emis_dict[scen][a]

        reset_state(f)
        set_emissions(f, years, emis)
        _, T, T_mean, _ = run_fair(f, years)
        delT_dict[scen] = T_mean[ind1:ind2]

    return delT_dict

def plot_emissions(emis_dict, agent, experiment_id, DECK=False):
    fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)
    for tag in emis_dict.keys():
        if not DECK:
            if tag == 'historical':
                years = np.arange(1750, 2024)
                ls = '-'
            elif 'ext' not in tag:
                years = np.arange(2024, 2151)
                ls = '-'
            else:
                years = np.arange(2024, 2501)
                ls = '--'

        elif DECK:
            if 'abrupt' in tag:
                years = np.arange(1750, 2051)
                ls = '--'
            elif '1pct' in tag:
                years = np.arange(1750, 1901)
                ls = '-'
        ax.plot(years, emis_dict[tag][agent], label=tag, ls=ls, lw=2, c=colors[tag])

    ax.legend(loc='upper right')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{agent} emissions')
    ax.set_title(f'CMIP7 {experiment_id} scenarios')

    return

def plot_cmip7_delT(delT_dict, scen_to_plot, experiment_id, DECK=False):
    fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)
    for tag in scen_to_plot:
        if not DECK:
            if tag == 'historical':
                years = np.arange(1750, 2024)
                ls = '-'
            elif 'ext' not in tag:
                years = np.arange(2024, 2151)
                ls = '-'
            else:
                years = np.arange(2024, 2501)
                ls = '--'
        else:
            if 'abrupt' in tag:
                years = np.arange(1750, 2051)
                ls = '--'
            elif '1pct' in tag:
                years = np.arange(1750, 1901)
                ls = '-'
        ax.plot(years, delT_dict[tag], label=tag, ls=ls, lw=2, c=colors[tag])

    ax.legend()#loc='upper left')
    ax.set_xlabel('Year')
    ax.set_ylabel(r'$\overline{\Delta T}(t)$')
    ax.set_title(f'CMIP7 {experiment_id} scenarios')

    return

# ----------------------
# DECK for CMIP7 Helpers
# ----------------------
# - Dunne et al. (2025)

# Includes abrupt-4xGHG and 1pctGHG;
# historical is loaded with ScenarioMIP

def generate_DECK_profile(agent, type):
    # If we select Sulfur or BC, calculate the CO2 profile
    # and scale bby the estimated historical ratio of CO2
    # emissions to the emissions of that agent

    c0 = {'CO2': 278.0, # ppm
          'CH4': 722.0, # ppb
          'N2O': 270.0} # ppb

    # Approx. ratio of X emissions to CO2 emissions in 2020
    convt = {'Sulfur': 4,
             'BC': 0.4}

    if agent == 'Sulfur' or agent == 'BC':
        agent_calc = 'CO2'
    else:
        agent_calc = agent

    if type == '1pct':
        start, stop = 1750, 1902
        years = np.arange(start, stop)
        n_steps = len(years)
        time_from_start = np.arange(n_steps)
        target_conc = c0[agent_calc] * (1.01 ** time_from_start)

    elif type == 'abrupt':
        start, stop = 1750, 2052
        years = np.arange(start, stop)
        n_steps = len(years)
        target_conc = np.full(n_steps, c0[agent_calc] * 4)

    f_conc = build_fair(start, stop, input_mode="concentration")

    target_conc_broadcast = target_conc[:, np.newaxis, np.newaxis]
    f_conc.concentration.loc[dict(specie=agent_calc, timebounds=slice(start,stop-1))] = target_conc_broadcast

    f_conc.run(progress=False)

    emis = f_conc.emissions.sel(specie=agent_calc, timepoints=slice(start,stop)).mean(dim='config').values.squeeze()

    if agent == 'Sulfur' or agent == 'BC':
        return emis[:-1] * convt[agent]
    return emis[:-1]

def load_DECK_CMIP7(agents):

    # Define DECK scenarios
    scenarios_DECK  = ['abrupt-4xCO2','abrupt-4xCH4',
                       'abrupt-4xN2O','abrupt-4xSulfur',
                       'abrupt-4xBC','1pctCO2','1pctCH4',
                       '1pctN2O','1pctSulfur','1pctBC']

    emis_dict_DECK = {}

    # All other scenarios
    for scen in scenarios_DECK:
        emis_dict_DECK[scen] = {}

        for a in agents:
            if a in scen and 'abrupt' in scen:
                vals = generate_DECK_profile(a, 'abrupt')
            elif a in scen and '1pct' in scen:
                vals = generate_DECK_profile(a, '1pct')

            if a in scen:
                emis_dict_DECK[scen][a] = vals
            elif a not in scen and '1pct' in scen:
                n_steps = 1901 - 1750
                emis_dict_DECK[scen][a] = np.zeros(n_steps)
            elif a not in scen and 'abrupt' in scen:
                n_steps = 2051 - 1750
                emis_dict_DECK[scen][a] = np.zeros(n_steps)

    return emis_dict_DECK


# -----------------------
# DAMIP for CMIP7 Helpers
# -----------------------
# - Gillett et al. (2025)

# ------------------------
# GeoMIP for CMIP6 Helpers
# ------------------------

# ----------------------------
# CS3 Outlook Scenario Helpers
# ----------------------------

def load_outlook_ghgs(historical_CMIP7):
    scenarios_outlook = ['historical','AA','CT']
    agents = ['CO2','CH4','N2O']

    path_AA_ghgs = 'data/CS3_outlook25/ghg_AA_OUTLOOK25_rcp26_2006-2150_c100405.nc'
    path_CT_ghgs = 'data/CS3_outlook25/ghg_CT_scenario_outlook25_2006-2150_c100901.nc'

    emis_df_AA_ghgs = xr.open_dataset(path_AA_ghgs)
    emis_df_CT_ghgs = xr.open_dataset(path_CT_ghgs)

    return #emis_dict_outlook



snames_short = ['historical','H-ext','H-ext-OS',
                'M','M-ext','ML','ML-ext','L',
                'L-ext','VLLO-ext','VLHO','VLHO-ext',
                'abrupt-4xCO2','abrupt-4xCH4','abrupt-4xN2O',
                'abrupt-4xSulfur','abrupt-4xBC','1pctCO2',
                '1pctCH4','1pctN2O','1pctSulfur','1pctBC']

colors = {
    snames_short[0]: '#808080', # historical
    snames_short[1]: '#800000', # H-ext
    snames_short[2]: '#ff0000', # H-ext-OS
    snames_short[3]: '#fc7b03', # M
    snames_short[4]: '#fc7b03', # M-ext
    snames_short[5]: '#d3a640', # ML
    snames_short[6]: '#d3a640', # ML-ext
    snames_short[7]: '#098740', # L
    snames_short[8]: '#098740', # L-ext
    snames_short[9]: '#0080d0', # VLLO-ext
    snames_short[10]: '#100060', # VLHO
    snames_short[11]: '#100060', # VLHO-ext
    snames_short[12]: '#800000', # abrupt-4xCO2
    snames_short[13]: '#ff0000', # abrupt-4xCH4
    snames_short[14]: '#fc7b03', # abrupt-4xN2O
    snames_short[15]: '#d3a640', # abrupt-4xSulfur
    snames_short[16]: '#098740', # abrupt-4xBC
    snames_short[17]: '#800000', # 1pctCO2
    snames_short[18]: '#ff0000', # 1pctCH4
    snames_short[19]: '#fc7b03', # 1pctN2O
    snames_short[20]: '#d3a640', # 1pctSulfur
    snames_short[21]: '#098740', # 1pctBC
}