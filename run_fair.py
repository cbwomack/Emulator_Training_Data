# Imports

## Standard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os
import pickle


## FaIR
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise

## Misc.
from typing import Dict, Iterable, Tuple, Optional

## Local
from paths import DATA_DIR

# ----------------------------
# Constants and Configurations
# ----------------------------

DEFAULT_SCENARIO = 'ssp245'
FAIR_PARAMS_CSV   = str(DATA_DIR / 'FaIR' / 'calibrated_constrained_parameters_median.csv')
FAIR_SPECIES_CSV  = str(DATA_DIR / 'FaIR' / 'species_configs_properties_1.4.1.csv')
VOLCANIC_FORCING = str(DATA_DIR / 'FaIR' / 'volcanic_ERF_monthly_175001-201912.csv')
SPECIES = ['CO2','CH4','N2O','Sulfur','BC','Aerosol-radiation interactions','Aerosol-cloud interactions']
_PROPERTIES = None  # lazily loaded on first use, see _species_properties()

def _species_properties(input_mode: str) -> Dict[str, dict]:
    """Copy baseline species properties and set CO2 input_mode."""
    global _PROPERTIES
    if _PROPERTIES is None:
        _PROPERTIES = read_properties(filename=FAIR_SPECIES_CSV)[1]
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
    #esms: Iterable[str] = DEFAULT_ESMs,
    scenario_name: str = "custom",
    ) -> FAIR:
    """
    Create a FaIR model with EBM configs loaded and species ready for custom emissions.
    """
    f = FAIR(ghg_method="meinshausen2020", ch4_method="thornhill2021")

    # define time and configs
    f.define_time(start, stop + 1, 1)  # keep consistent everywhere
    #configs = _ebm_config_names(esms)
    #f.define_configs(configs)

    #configs = _calib_config_labels(FAIR_PARAMS_CSV)
    df_configs = pd.read_csv(FAIR_PARAMS_CSV, index_col=0)
    configs = df_configs.index
    f.define_configs(configs)

    # define species
    props = _species_properties(input_mode)
    f.define_species(SPECIES, props)

    # pull baseline info from RCMIP using any valid scenario (for shapes/initials)
    f.define_scenarios([default_scenario])
    f.allocate()
    f.fill_from_rcmip()
    #f.fill_species_configs()

    f.fill_species_configs(FAIR_SPECIES_CSV)
    f.override_defaults(FAIR_PARAMS_CSV)

    # initialise state
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    initialise(f.ocean_heat_content_change, 0)

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
    #_apply_ebm_configs(f, configs, stochastic=False)

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

    # Check if this scenario is already cached
    tag_post = ''
    for a in agents:
        tag_post += '_' + a
    filepath_tier1 = str(DATA_DIR / 'FaIR_IO' / 'emissions' / f'ScenarioMIP_tier1{tag_post}.pkl')
    filepath_tier2 = str(DATA_DIR / 'FaIR_IO' / 'emissions' / f'ScenarioMIP_tier2{tag_post}.pkl')

    if os.path.exists(filepath_tier1) and os.path.exists(filepath_tier2):
        with open(filepath_tier1, 'rb') as f:
            emis_dict_tier1 = pickle.load(f)
        with open(filepath_tier2, 'rb') as f:
            emis_dict_tier2 = pickle.load(f)

    else:
        # Define CMIP7 scenarios and extensions
        scenarios_tier1  = ['high-extension','medium-extension','medium-overshoot',
                            'low','verylow','verylow-overshoot']
        scenarios_tier2  = ['high-overshoot','medium-extension','medium-overshoot',
                            'low','verylow-overshoot']
        scenarios_all    = scenarios_tier1 + scenarios_tier2

        tags_tier1 = ['H-ext','M','ML','L','VLLO-ext','VLHO']
        tags_tier2 = ['H-ext-OS','M-ext','ML-ext','L-ext','VLHO-ext']
        tags_all   = tags_tier1 + tags_tier2
        data_path  = str(DATA_DIR / 'FaIR' / 'extensions_1750-2500.csv')
        emis_df    = pd.read_csv(data_path)

        # Indices for emissions dataset
        n_col_skip = 5
        n_years_base, n_years_ext = 127, 477
        emis_dict_tier1, emis_dict_tier2 = {}, {}

        # Historical emissions
        n_years_hist = 274 # Up to 2023
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

                if tag in tags_tier1:
                    emis_dict_tier1[tag][a] = vals
                else:
                    emis_dict_tier2[tag][a] = vals

        with open(filepath_tier1, 'wb') as f:
            pickle.dump(emis_dict_tier1, f)
        with open(filepath_tier2, 'wb') as f:
            pickle.dump(emis_dict_tier2, f)

    return emis_dict_tier1, emis_dict_tier2

def get_delT(emis_dict, scenarios, agents, MIP='ScenarioMIP_tier1'):

    # Check if this scenario is already cached
    tag_post = ''
    for a in agents:
        tag_post += '_' + a
    filepath = str(DATA_DIR / 'FaIR_IO' / 'delT' / f'{MIP}{tag_post}.pkl')

    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            delT_dict = pickle.load(f)

    # Otherwise, generate it
    else:
        delT_dict = {}
        for scen in scenarios:
            if MIP in ['ScenarioMIP_tier1','ScenarioMIP_tier2','GeoMIP']:
                if scen == 'historical':
                    start, stop = 1750, 2024
                    ind1, ind2 = start - start, stop - 1750
                elif 'ext' not in scen:
                    start, stop = 1750, 2151
                    ind1, ind2 = 2024 - start, stop - start
                else:
                    start, stop = 1750, 2501
                    ind1, ind2 = 2024 - start, stop - start
            elif MIP == 'DECK':
                if 'abrupt' in scen:
                    start, stop = 1750, 2051
                elif '1pct' in scen:
                    start, stop = 1750, 1901
                ind1, ind2 = start - start, stop - start
            elif MIP == 'CS3':
                if scen == 'historical':
                    start, stop = 1750, 2006
                    ind1, ind2 = start - start, stop - 1750
                else:
                    start, stop = 1750, 2151
                    ind1, ind2 = 2006 - start, stop - start
            elif MIP == 'Optimal':
                start, stop = 1750, 2501
                ind1, ind2 = start - start, stop - start
            else:
                raise ValueError(f'Error: type {MIP} not recognized.')

            years = np.arange(start, stop)
            n_steps = len(years)
            nan_force = np.zeros(n_steps)

            f = build_fair(start, stop, input_mode="emissions")
            emis = {a: nan_force for a in agents}

            for a in agents:
                if scen != 'historical' and MIP in ['ScenarioMIP_tier1','ScenarioMIP_tier2','GeoMIP','CS3']:
                    emis[a] = np.concatenate((emis_dict['historical'][a][:ind1], emis_dict[scen][a]), axis=None)
                else:
                    emis[a] = emis_dict[scen][a]

            reset_state(f)
            set_emissions(f, years, emis)
            _, T, T_mean, _ = run_fair(f, years)
            delT_dict[scen] = T_mean[ind1:ind2]

        with open(filepath, 'wb') as f:
            pickle.dump(delT_dict, f)

    return delT_dict

def plot_emissions(emis_dict, agent, experiment_id, MIP='ScenarioMIP_tier1'):
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
            ax.semilogy(years, emis_dict[tag][agent], label=tag, ls=ls, lw=2, c=colors[tag])
        else:
            ax.plot(years, emis_dict[tag][agent], label=tag, ls=ls, lw=2, c=colors[tag])

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

def plot_delT(delT_dict, scen_to_plot, experiment_id, MIP='ScenarioMIP'):
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

        ax.plot(years, delT_dict[tag], label=tag, ls=ls, lw=2, c=colors[tag])

    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel(r'$\overline{\Delta T}(t)$')
    ax.set_title(f'{experiment_id} scenarios')

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


# ------------------------
# GeoMIP for CMIP6 Helpers
# ------------------------
# - Visioni et al. (2023)

def find_sulfur_for_target_temp(
    modify_year,
    target_year,
    hist_emissions,
    future_baseline_emissions,
    solved_sulfur_so_far,
    target_temp,
    initial_guess=50.0
):
    """
    Solves for the sulfur emission in `modify_year` to hit a target temperature in `target_year`.
    Uses an intelligent first guess and a temperature-based tolerance for faster convergence.
    """
    start_year = 1750
    run_stop_year = target_year + 1
    run_years = np.arange(start_year, run_stop_year)

    modify_idx = modify_year - 2024
    target_idx = target_year - 2024

    # Initialize previous temperature with a value far from the target
    prev_temp_result = -999.0

    # Set initial search bounds. A wide range is still needed for stability.
    min_S_guess = 100
    max_S_guess = 1000
    # Use the intelligent guess as the first try
    current_S_guess = initial_guess

    for i in range(20):
        # Construct the full emissions scenario for this trial run
        run_emissions = {}
        for agent in future_baseline_emissions.keys():
            hist_part = hist_emissions[agent]
            future_part = future_baseline_emissions[agent][:target_idx + 1].copy()
            if agent == 'Sulfur':
                future_part[:len(solved_sulfur_so_far)] = solved_sulfur_so_far
                future_part[modify_idx] = current_S_guess
            run_emissions[agent] = np.concatenate([hist_part, future_part])

        # Run a new FaIR instance for this trial
        f_temp = build_fair(start_year, run_stop_year)
        set_emissions(f_temp, run_years, run_emissions)
        _, _, T_mean, _ = run_fair(f_temp, run_years)

        temp_result = T_mean[-1]

        # New stopping condition: check if temperature has converged
        if abs(temp_result - prev_temp_result) < 1e-4:
            break
        prev_temp_result = temp_result

        # Adjust search bounds
        if temp_result > target_temp:
            min_S_guess = current_S_guess
        else:
            max_S_guess = current_S_guess

        # For the next iteration, the guess is the midpoint of the new bounds
        current_S_guess = (min_S_guess + max_S_guess) / 2

    return current_S_guess

def solve_G3(emis_dict_tier1, verbose=False):

    print("--- Establishing Target Temperature ---")
    hist_start, hist_stop = 1750, 2024
    hist_years = np.arange(hist_start, hist_stop)
    f_hist = build_fair(hist_start, hist_stop)
    set_emissions(f_hist, hist_years, emis_dict_tier1['historical'])
    _, _, T_mean_hist, _ = run_fair(f_hist, hist_years)
    target_temp = T_mean_hist[-1]
    print(f"Target temperature from 2024 locked in: {target_temp:.4f}°C\n")

    print("--- Solving for Future Sulfur Emissions ---")
    baseline_scenario_name = 'ML'
    hist_emissions = emis_dict_tier1['historical']
    future_baseline_emissions = emis_dict_tier1[baseline_scenario_name]

    future_start, future_stop = 2024, 2150
    future_years = np.arange(future_start, future_stop)

    calculated_S_emissions = np.zeros_like(future_baseline_emissions['Sulfur'])
    last_solved_sulfur = hist_emissions['Sulfur'][-1]

    for i, year in enumerate(future_years):
        modify_year = year
        target_year = year + 1

        solved_sulfur_so_far = calculated_S_emissions[:i]

        required_sulfur = find_sulfur_for_target_temp(
            modify_year,
            target_year,
            hist_emissions,
            future_baseline_emissions,
            solved_sulfur_so_far,
            target_temp,
            initial_guess=last_solved_sulfur # Pass the previous solution as the first guess
        )

        calculated_S_emissions[i] = required_sulfur
        last_solved_sulfur = required_sulfur # Update for the next iteration

        # --- Yearly Reporting ---
        final_stop_year = target_year + 1
        final_run_years = np.arange(hist_start, final_stop_year)
        final_run_emissions = {}
        for agent in future_baseline_emissions.keys():
            hist_part = hist_emissions[agent]
            future_part_for_run = future_baseline_emissions[agent][:i+2].copy()
            if agent == 'Sulfur':
                future_part_for_run[:i+1] = calculated_S_emissions[:i+1]
            final_run_emissions[agent] = np.concatenate([hist_part, future_part_for_run])

        f_report = build_fair(hist_start, final_stop_year)
        set_emissions(f_report, final_run_years, final_run_emissions)
        _, _, T_mean_report, _ = run_fair(f_report, final_run_years)

        if target_year % 25 == 0 and verbose:
            print(f"Controlling Temp({target_year}) with Sulfur({modify_year}): Required S = {required_sulfur:8.4f} Tg/yr, Resulting Temp = {T_mean_report[-1]:.4f}°C")

    print(f"\nFinished.")

    return calculated_S_emissions

def load_geoMIP_CMIP6(agents=None, emis_dict_tier1=None, verbose=False):

    geoMIP_filepath = str(DATA_DIR / 'saved_emissions' / 'emis_dict_geoMIP.pkl')

    if os.path.exists(geoMIP_filepath):
        with open(geoMIP_filepath, 'rb') as f:
            emis_dict_geoMIP = pickle.load(f)

    else:
        emis_dict_geoMIP = {'ML-G3':{}, # Hold temperature at 2024 levels under ML
                            'ML-G4':{}, # Inject 5 extra Tg of SO2 per year under ML until 2070
                            'historical':{}}
        for a in agents:
            emis_dict_geoMIP['ML-G3'][a] = emis_dict_tier1['ML'][a].copy()
            emis_dict_geoMIP['ML-G4'][a] = emis_dict_tier1['ML'][a].copy()
            emis_dict_geoMIP['historical'][a] = emis_dict_tier1['historical'][a].copy()

        emis_dict_geoMIP['ML-G3']['Sulfur'] = solve_G3(emis_dict_tier1, verbose=verbose)
        emis_dict_geoMIP['ML-G4']['Sulfur'][:46] += 5

        with open(geoMIP_filepath, 'wb') as f:
            pickle.dump(emis_dict_geoMIP, f)

    return emis_dict_geoMIP

# ----------------------------
# CS3 Outlook Scenario Helpers
# ----------------------------

def load_CS3(agents=None, emis_dict_tier1=None):

    CS3_filepath = str(DATA_DIR / 'saved_emissions' / 'emis_dict_CS3.pkl')

    if os.path.exists(CS3_filepath):
        with open(CS3_filepath, 'rb') as f:
            emis_dict_CS3 = pickle.load(f)

    else:
        filepath_AA = str(DATA_DIR / 'CS3_outlook25' / 'edaily.e5_AA_20250717.fordata')
        filepath_CT = str(DATA_DIR / 'CS3_outlook25' / 'edaily.e5_CT_20250712.fordata')
        emis_df_AA = pd.read_csv(filepath_AA, sep=r'\s+', usecols=[1,4,7,8,12])
        emis_df_CT = pd.read_csv(filepath_CT, sep=r'\s+', usecols=[1,4,7,8,12])

        fair_to_cs3 = {'CO2':'CO2:Pg',
                       'CH4':'CH4:Pg',
                       'N2O':'N2O:Tg',
                       'Sulfur':'SO2:Tg',
                       'BC':'BC:Tg'}

        emis_dict_CS3 = {'AA':{}, # Accelerated Actions
                         'CT':{}, # Current Trends
                         'historical':{}}

        for a in agents:
            if a == 'CH4':
                factor = 1000 # CS3 stores CH4 in Pg instead of Tg that FaIR expects
            else:
                factor = 1
            emis_dict_CS3['AA'][a] = emis_df_AA[fair_to_cs3[a]].values * factor
            emis_dict_CS3['CT'][a] = emis_df_CT[fair_to_cs3[a]].values * factor
            emis_dict_CS3['historical'][a] = emis_dict_tier1['historical'][a].copy()[0:256]

        with open(CS3_filepath, 'wb') as f:
            pickle.dump(emis_dict_CS3, f)

    return emis_dict_CS3



snames_short = ['historical','H-ext','H-ext-OS',
                'M','M-ext','ML','ML-ext','L',
                'L-ext','VLLO-ext','VLHO','VLHO-ext',
                'abrupt-4xCO2','abrupt-4xCH4','abrupt-4xN2O',
                'abrupt-4xSulfur','abrupt-4xBC','1pctCO2',
                '1pctCH4','1pctN2O','1pctSulfur','1pctBC',
                'AA','CT','ML-G3','ML-G4',
                'opt_constant_tier1','opt_constant_tier2','opt_constant_DECK','opt_constant_all',
                'opt_ramp_tier1','opt_ramp_tier2','opt_ramp_DECK','opt_ramp_all',
                'opt_gaussian_tier1','opt_gaussian_tier2','opt_gaussian_DECK','opt_gaussian_all']

colors = {
    snames_short[0]:  '#808080', # historical
    snames_short[1]:  '#800000', # H-ext
    snames_short[2]:  '#ff0000', # H-ext-OS
    snames_short[3]:  '#fc7b03', # M
    snames_short[4]:  '#fc7b03', # M-ext
    snames_short[5]:  '#d3a640', # ML
    snames_short[6]:  '#d3a640', # ML-ext
    snames_short[7]:  '#098740', # L
    snames_short[8]:  '#098740', # L-ext
    snames_short[9]:  '#0080d0', # VLLO-ext
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
    snames_short[22]: '#098740', # AA
    snames_short[23]: '#800000', # CT
    snames_short[24]: '#d3a640', # ML-G3
    snames_short[25]: '#fc7b03', # ML-G4

    snames_short[26]: '#800000', # opt_const_tier1
    snames_short[27]: '#ff0000',
    snames_short[28]: '#fc7b03',
    snames_short[29]: '#d3a640',
    snames_short[30]: '#800000', # opt_ramp_tier1
    snames_short[31]: '#ff0000',
    snames_short[32]: '#fc7b03',
    snames_short[33]: '#d3a640',
    snames_short[34]: '#800000', # opt_gaussian_tier1
    snames_short[35]: '#ff0000',
    snames_short[36]: '#fc7b03',
    snames_short[37]: '#d3a640',
}