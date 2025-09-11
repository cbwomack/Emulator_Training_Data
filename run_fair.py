import numpy as np
import pandas as pd
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise
from scipy.fft import ifft, rfft, rfftfreq
import utils_TF

DEFAULT_SCENARIO = 'ssp245'
EBM_CONFIG = 'data/FaIR/4xCO2_cummins_ebm3.csv'
VOLCANIC_FORCING = 'data/FaIR/volcanic_ERF_monthly_175001-201912.csv'
df = pd.read_csv(EBM_CONFIG)
DEFAULT_ESMs = df['model'].unique()


def get_ebm_configs(esms):
    ebm_configs = []
    for model in esms:
        for run in df.loc[df['model']==model, 'run']:
            ebm_configs.append(f"{model}_{run}")
    return ebm_configs

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
