# Imports

## Standard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    emis_dict_tier1['historical'] = {}
    for a in agents:
        if a == 'CO2':
            a_full = 'CO2 FFI'
        else:
            a_full = a

        historical_emis = emis_df[(emis_df["scenario"] == scenarios_all[0]) & (emis_df["variable"] == a_full)].iloc[:, n_col_skip:n_col_skip + n_years_hist].to_numpy().reshape(-1)
        emis_dict_tier1['historical'][a] = historical_emis

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

def run_cmip7_sweeps(
    start: int, stop: int, years: Iterable[int], agents: Iterable[str], co2_only: bool=True
    ) -> Dict[str, Dict[str, np.ndarray]]:
    """
    For each CMIP7 scenario, and for each agent in `agents`,
    run FaIR with only that agent's emissions active (others zero).
    Returns dict[scenario_tag][agent] = T_mean (GMST mean across configs).
    """
    emis_all = load_cmip7_extensions(n_steps=len(list(years)), co2_only=co2_only)
    tags = list(emis_all.keys())
    out: Dict[str, Dict[str, np.ndarray]] = {}

    # Build once per scenario, reuse via reset_state
    for tag in tags:
        out[tag] = {}
        f = build_fair(start, stop)  # reuse within scenario
        for agent in agents:
            reset_state(f)
            emis = {a: np.zeros(len(list(years))) for a in SPECIES}
            key = agent if agent != "SO2" else "Sulfur"
            emis[key] = emis_all[tag][key]
            set_emissions(f, years, emis)
            _, _, T_mean, _ = run_fair(f, years)
            out[tag][agent] = T_mean
    return out

snames_short = ['historical','H-ext','H-ext-OS',
                'M','M-ext','ML','ML-ext','L',
                'L-ext','VLLO-ext','VLHO','VLHO-ext']

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
}

def plot_cmip7(emis_dict, agent, tier):
    fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)
    for tag in emis_dict.keys():
        if tag == 'historical':
            years = np.arange(1750, 2024)
            ls = '-'
        elif 'ext' not in tag:
            years = np.arange(2024, 2151)
            ls = '-'
        else:
            years = np.arange(2024, 2501)
            ls = '--'
        ax.plot(years, emis_dict[tag][agent], label=tag, ls=ls, lw=2, c=colors[tag])

    ax.legend(loc='upper right')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{agent} emissions')
    ax.set_title(f'CMIP7 tier {tier} scenarios')

    return

"""
def get_ebm_configs(esms):
    ebm_configs = []
    for model in esms:
        for run in df.loc[df['model']==model, 'run']:
            ebm_configs.append(f"{model}_{run}")
    return ebm_configs

def get_scenario_agent(scenario, agent, esms=DEFAULT_ESMs):
    start, stop = 1750, 2100

    f = FAIR(ghg_method="meinshausen2020", ch4_method='thornhill2021')
    f.define_time(start, stop, 1)
    f.define_scenarios([scenario])

    configs = get_ebm_configs([x.split()[0] for x in esms])
    f.define_configs(configs)

    species = ['CO2', 'CH4', 'N2O', 'Sulfur', 'BC']#, 'Volcanic']
    properties = {s: read_properties()[1][s] for s in species}
    properties['CO2']['input_mode'] = 'emissions'
    f.define_species(species, properties)

    f.allocate()
    f.fill_from_rcmip()
    f.fill_species_configs()

    initialise(f.concentration, f.species_configs['baseline_concentration'])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    # Load each set of energy balance model configs
    seed = 1355763
    for config in configs:
        model, run = config.split('_')
        condition = (df['model']==model) & (df['run']==run)
        fill(f.climate_configs['ocean_heat_capacity'], df.loc[condition, 'C1':'C3'].values.squeeze(), config=config)
        fill(f.climate_configs['ocean_heat_transfer'], df.loc[condition, 'kappa1':'kappa3'].values.squeeze(), config=config)
        fill(f.climate_configs['deep_ocean_efficacy'], df.loc[condition, 'epsilon'].values[0], config=config)
        fill(f.climate_configs['gamma_autocorrelation'], df.loc[condition, 'gamma'].values[0], config=config)
        fill(f.climate_configs['sigma_eta'], df.loc[condition, 'sigma_eta'].values[0], config=config)
        fill(f.climate_configs['sigma_xi'], df.loc[condition, 'sigma_xi'].values[0], config=config)
        fill(f.climate_configs['stochastic_run'], False, config=config)
        fill(f.climate_configs['use_seed'], False, config=config)
        fill(f.climate_configs['seed'], seed, config=config)
        seed = seed + 399

    emis_ens = f.emissions.sel(timepoints=slice(start, stop)).loc[dict(scenario=scenario)].sel(specie=agent).values
    emis_agent = np.mean(emis_ens, axis=1)

    return emis_agent


def initialise_fair(start, stop, mode='emissions', default_scenario=DEFAULT_SCENARIO, esms=DEFAULT_ESMs):
    # Instantiate FaIR model
    f = FAIR(ghg_method="meinshausen2020", ch4_method='thornhill2021')
    f.define_time(start, stop + 1, 1)

    # Load energy balance model configs tuned against 66 different ESMs
    configs = get_ebm_configs([x.split()[0] for x in esms])
    f.define_configs(configs)

    # Define species we work with (meinshausen2020 method requires CH4 and N2O)
    species = ['CO2', 'CH4', 'N2O', 'Sulfur', 'BC']#, 'Volcanic']
    properties = {s: read_properties()[1][s] for s in species}
    properties['CO2']['input_mode'] = mode
    f.define_species(species, properties)

    # Load emission data 1750-2000 by querying an arbitrary scenario
    f.define_scenarios([default_scenario])
    f.allocate()
    f.fill_from_rcmip()

    # Load prescribed volcanic forcing for 1750-2000
    #df_volcanic = pd.read_csv(VOLCANIC_FORCING, index_col='year')
    #volcanic_forcing = np.zeros(352)
    #volcanic_forcing[:271] = df_volcanic.loc[1749:].groupby(np.ceil(df_volcanic.loc[1749:].index) // 1).mean().squeeze().values
    #fill(f.forcing, volcanic_forcing[:, None, None], specie="Volcanic")

    # Load species config for gas cycle and radiative forcing models
    f.fill_species_configs()

    # Initialise variable at starting point
    initialise(f.concentration, f.species_configs['baseline_concentration'])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    # Rename all the scenario fields to 'custom' in the xarray attributes
    f.define_scenarios(["custom"])
    f.emissions = f.emissions.assign_coords(scenario=f.scenarios)
    f.cumulative_emissions = f.cumulative_emissions.assign_coords(scenario=f.scenarios)
    f.concentration = f.concentration.assign_coords(scenario=f.scenarios)
    f.forcing = f.forcing.assign_coords(scenario=f.scenarios)
    f.temperature = f.temperature.assign_coords(scenario=f.scenarios)
    f.species_configs = f.species_configs.assign_coords(scenario=f.scenarios)

    # Load each set of energy balance model configs
    seed = 1355763
    for config in configs:
        model, run = config.split('_')
        condition = (df['model']==model) & (df['run']==run)
        fill(f.climate_configs['ocean_heat_capacity'], df.loc[condition, 'C1':'C3'].values.squeeze(), config=config)
        fill(f.climate_configs['ocean_heat_transfer'], df.loc[condition, 'kappa1':'kappa3'].values.squeeze(), config=config)
        fill(f.climate_configs['deep_ocean_efficacy'], df.loc[condition, 'epsilon'].values[0], config=config)
        fill(f.climate_configs['gamma_autocorrelation'], df.loc[condition, 'gamma'].values[0], config=config)
        fill(f.climate_configs['sigma_eta'], df.loc[condition, 'sigma_eta'].values[0], config=config)
        fill(f.climate_configs['sigma_xi'], df.loc[condition, 'sigma_xi'].values[0], config=config)
        fill(f.climate_configs['stochastic_run'], False, config=config)
        fill(f.climate_configs['use_seed'], False, config=config)
        fill(f.climate_configs['seed'], seed, config=config)
        seed = seed + 399
    return f

def run(f, years, co2, ch4, n2o, so2, BC):
    # Replace CO2 emission with custom emissions
    co2Xconfigs = np.tile(co2, (len(f.configs), 1)).T[:, None, :]
    ch4Xconfigs = np.tile(ch4, (len(f.configs), 1)).T[:, None, :]
    n2oXconfigs = np.tile(n2o, (len(f.configs), 1)).T[:, None, :]
    so2Xconfigs = np.tile(so2, (len(f.configs), 1)).T[:, None, :]
    BCXconfigs = np.tile(BC, (len(f.configs), 1)).T[:, None, :]
    f.emissions.sel(specie='CO2', timepoints=slice(min(years), max(years) + 1))[:] = co2Xconfigs
    f.emissions.sel(specie='CH4', timepoints=slice(min(years), max(years) + 1))[:] = ch4Xconfigs
    f.emissions.sel(specie='N2O', timepoints=slice(min(years), max(years) + 1))[:] = n2oXconfigs
    f.emissions.sel(specie='Sulfur', timepoints=slice(min(years), max(years) + 1))[:] = so2Xconfigs
    f.emissions.sel(specie='BC', timepoints=slice(min(years), max(years) + 1))[:] = BCXconfigs

    # Run fair for each set of energy balance model configs
    f.run()

    # Compute ECS
    ECS = round(f.ebms.ecs.mean().item(), 2)

    # Return GMST anomaly
    t = list(range(min(years), max(years) + 1))
    T = f.temperature.sel(timebounds=slice(min(years), max(years))).loc[dict(scenario='custom', layer=0)].values
    Tbar = np.mean(T, axis=1)
    return t, T, Tbar, ECS

def run_custom(fair, n_steps, years, agent, dt, pulse=True):
    if pulse:
        emissions = np.zeros(n_steps)
        emissions[0] = 1.0 / dt
    else:
        emissions = np.random.randn(n_steps)

    nan_force = np.linspace(0, 0, n_steps)

    co2, ch4, n2o, so2, BC = nan_force, nan_force, nan_force, nan_force, nan_force

    if agent == 'CO2':
        co2 = emissions
    elif agent == 'CH4':
        ch4 = emissions
    elif agent == 'N2O':
        n2o = emissions
    elif agent == 'SO2':
        so2 = emissions
    elif agent == 'BC':
        BC = emissions
    else:
        raise ValueError(f'Error agent {agent} not recognized.')

    t, h_ens, h_mean, ECS = run(fair, years, co2, ch4, n2o, so2, BC)

    if pulse:
        freq = rfftfreq(n_steps, d=dt)
        H_mean = rfft(h_mean) * dt

        return t, h_ens, h_mean, ECS, freq, H_mean

    return t, h_ens, h_mean, ECS

def run_multi_agent(fair, n_steps, years, agents, emission_dict):
    emissions = np.zeros((5, n_steps))

    for i, agent in enumerate(agents):
        emissions[i] = emission_dict[agent]

    t, T_ens, T_mean, ECS = run(fair, years, emissions[0],
                                             emissions[1],
                                             emissions[2],
                                             emissions[3],
                                             emissions[4])

    return t, T_ens, T_mean, ECS

def get_CMIP7(n_steps, co2_only=True):
    scenarios_CMIP7 = ['high-overshoot','high-extension','verylow','verylow-overshoot','low','medium-extension','medium-overshoot']
    scenarios_CMIP7_short = ['HO','HE','VL','VLO','L','ME','MO']
    if co2_only:
        forcing_names = ['CO2 FFI']
        forcing_short = ['CO2']
    else:
        forcing_names = ['CO2 FFI','CH4','N2O','Sulfur','BC']
        forcing_short = ['CO2','CH4','N2O','Sulfur','BC']
    data_path = 'data/FaIR/extensions_1750-2500.csv'

    emis_df = pd.read_csv(data_path)
    emis_dict = {}

    for i, scen in enumerate(scenarios_CMIP7):
        scen_short = scenarios_CMIP7_short[i]
        emis_dict[scen_short] = {}
        for j, forcing in enumerate(forcing_names):
            emis_dict[scen_short][forcing_short[j]] = emis_df[(emis_df['scenario'] == scen) & (emis_df['variable'] == forcing)].iloc[:, 5:5 + n_steps].to_numpy().reshape(-1)

    return emis_dict

def run_CMIP7(start, stop, years, n_steps, agents):
    emis_dict = get_CMIP7(n_steps)
    scenarios = [scen for scen in emis_dict.keys()]

    nan_force = np.zeros(n_steps)
    delT_dict = {}

    for scen in scenarios:
        delT_dict[scen] = {}

        for agent in agents:
            fair_train = initialise_fair(start, stop)
            emis_dict_temp = {
            'CO2':emis_dict[scen][agent] if agent == 'CO2' else nan_force,
            'CH4':emis_dict[scen][agent] if agent == 'CH4' else nan_force,
            'N2O':emis_dict[scen][agent] if agent == 'N2O' else nan_force,
            'Sulfur':emis_dict[scen][agent] if agent == 'Sulfur' else nan_force,
            'BC':emis_dict[scen][agent] if agent == 'BC' else nan_force,
            }
            t, T_ens, T_train, ECS = run_multi_agent(fair_train, n_steps, years, agents, emis_dict_temp)
            delT_dict[scen][agent] = T_train

    return delT_dict






"""

"""
def generate_responses(start, stop, n_steps, years, dt, freq_s, nperseg):

    agents = ['CO2','CH4','N2O']
    response_dict = {}

    for agent in agents:
        response_dict[agent] = {}

        fair_pulse = initialise_fair(start, stop)
        fair_noise = initialise_fair(start, stop)

        t, h_true_ens, response_dict[agent]['h_true_ET'], ECS_true, freq_true, response_dict[agent]['H_true_ET'] = run_custom(fair_pulse, n_steps, years, agent, dt)
        t, T_ens, response_dict[agent]['temp'], ECS_CO2_mean = run_custom(fair_noise, n_steps, years, agent, dt, pulse=False)

        response_dict[agent]['pulse'], response_dict[agent]['noise'] = fair_pulse, fair_noise

        h_true_EC_ens = fair_pulse.concentration.sel(timebounds=slice(min(years), max(years))).loc[dict(scenario='custom')].sel(specie=agent).values
        response_dict[agent]['h_true_EC'] = np.mean(h_true_EC_ens, axis=1)
        response_dict[agent]['H_true_EC'] = rfft(response_dict[agent]['h_true_EC']) * dt

        h_true_EF_ens = fair_pulse.forcing.sel(timebounds=slice(min(years), max(years))).loc[dict(scenario='custom')].sel(specie=agent).values
        response_dict[agent]['h_true_EF'] = np.mean(h_true_EF_ens, axis=1)
        response_dict[agent]['H_true_EF'] = rfft(response_dict[agent]['h_true_EF']) * dt

        emis_ens = fair_noise.emissions.sel(timepoints=slice(min(years), max(years)+1)).loc[dict(scenario='custom')].sel(specie=agent).values
        response_dict[agent]['emis'] = np.mean(emis_ens, axis=1)

        conc_ens = fair_noise.concentration.sel(timebounds=slice(min(years), max(years))).loc[dict(scenario='custom')].sel(specie=agent).values
        response_dict[agent]['conc'] = np.mean(conc_ens, axis=1)

        forc_ens = fair_noise.forcing.sel(timebounds=slice(min(years), max(years))).loc[dict(scenario='custom')].sel(specie=agent).values
        response_dict[agent]['forc'] = np.mean(forc_ens, axis=1)

        # Responses in terms of emissions -> target
        response_dict[agent]['H_est_EC'], response_dict[agent]['h_est_EC'], response_dict[agent]['freq_EC'] = utils_TF.identify_system(response_dict[agent]['emis'],
                                                                                                               response_dict[agent]['conc'],
                                                                                                               freq_s, nperseg)
        response_dict[agent]['H_est_EF'], response_dict[agent]['h_est_EF'], response_dict[agent]['freq_EF']  = utils_TF.identify_system(response_dict[agent]['emis'],
                                                                                                               response_dict[agent]['forc'],
                                                                                                               freq_s, nperseg)
        response_dict[agent]['H_est_ET'], response_dict[agent]['h_est_ET'], response_dict[agent]['freq_ET']  = utils_TF.identify_system(response_dict[agent]['emis'],
                                                                                                               response_dict[agent]['temp'],
                                                                                                               freq_s, nperseg)

        # Responses in terms of concentration -> target
        response_dict[agent]['H_est_CF'], response_dict[agent]['h_est_CF'], response_dict[agent]['freq_CF']  = utils_TF.identify_system(response_dict[agent]['conc'],
                                                                                                               response_dict[agent]['forc'],
                                                                                                               freq_s, nperseg)
        response_dict[agent]['H_est_CT'], response_dict[agent]['h_est_CT'], response_dict[agent]['freq_CT']  = utils_TF.identify_system(response_dict[agent]['conc'],
                                                                                                               response_dict[agent]['temp'],
                                                                                                               freq_s, nperseg)

        # Responses in terms of forcing -> target
        response_dict[agent]['H_est_FT'], response_dict[agent]['h_est_FT'], response_dict[agent]['freq_FT']  = utils_TF.identify_system(response_dict[agent]['forc'],
                                                                                                               response_dict[agent]['temp'],
                                                                                                               freq_s, nperseg)

    return response_dict
"""