import jax
import jax.numpy as jnp
from jax import lax
import optax
import pickle
import matplotlib.pyplot as plt

import run_fair
from jax import config
config.update("jax_debug_nans", True)

params_FaIR = {
    # Energy Balance Model 4.12957105, 17.21094065, 85.39779719
    'ocean_heat_capacity': jnp.array([3.7, 17.21094065, 85.39779719]),
    'ocean_heat_transfer': jnp.array([1.298404664, 2.447033095, 1.155547811]), #1.298404664, 2.447033095, 1.155547811
    'deep_ocean_efficacy': 1.285510371, #1.285510371
    'gamma_autocorrelation': 5.910175419,
    'timestep': 0.5,

    # Indices & Species Count
    'n_ghgs': 3, # [CO2, CH4, N2O]
    'n_gas_boxes': 4, # Maximum boxes (CO2 needs 4)
    'co2_idx': 0, 'ch4_idx': 1, 'n2o_idx': 2, 'so2_idx': 3, 'bc_idx': 4,

    # Carbon Cycle (iIRF) for CO2
    'iirf_0': 31.6137431,
    'iirf_airborne': 0.003177357, #0.003177357
    'iirf_uptake': 0.002,#0.001226978,
    'iirf_temperature': 2.018002016, # 2.018002016
    'iirf_max': 97.0,
    'g0': 0.010178288, #0.010178288,
    'g1': 11.41262243, #11.41262243

    # CH4 Lifetime Chemistry (OH feedback)
    'ch4_lifetime_chemical_sensitivity': jnp.array([0.0, 0.000254, 0.0, 0.0, 0.0]),
    'ch4_lifetime_temperature_sensitivity': -0.0408,
    'emissions_indices': jnp.array([False, True, False, False, False]), # CH4 emissions index
    'concentration_indices': jnp.array([False, True, False, False, False]), # CH4 conc index

    # Greenhouse Gas Properties
    'lifetime': jnp.array([
        [1.0e+09, 394.4, 36.54, 4.304], # CO2 [1.0e+09, 394.4, 36.54, 4.304]
        [8.25, 8.25, 8.25, 8.25],       # CH4
        [109.0, 109.0, 109.0, 109.0]    # N2O
    ]),
    'partition_fraction': jnp.array([
        [0.2173, 0.224, 0.2824, 0.2763],  # CO2
        [1.0, 0.0, 0.0, 0.0],             # CH4
        [1.0, 0.0, 0.0, 0.0]              # N2O
    ]),
    'baseline_concentration': jnp.array([278.0387652, 729.2, 270.1, 0.0, 0.0]),
    'baseline_emissions': jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    'concentration_per_emission': jnp.array([0.46897, 0.35173, 0.201, 0.0, 0.0]),
    'forcing_scaling': jnp.array([0.989765309, 1.016970059, 1.007316444]),
    'radiative_efficiency': jnp.array([0.0000133, 0.000389, 0.003196]),

    # Meinshausen 2020 Species Mapping
    'co2_indices': jnp.array([True, False, False]),
    'ch4_indices': jnp.array([False, True, False]),
    'n2o_indices': jnp.array([False, False, True]),
    'minor_ghg_indices': jnp.array([False, False, False]),

    # ARI Parameters
    'erfari_radiative_efficiency': jnp.array([0.0, -0.000003, -0.000037, -0.003, 0.027]), # SO2 is negative, BC is positive
    'ari_em_indices': jnp.array([False, False, False, True, True]), # Sulfur, BC
    'ari_conc_indices': jnp.array([False, True, True, False, False]), # CH4, N2O also have ARI

    # ACI Parameters (Stevens 2015 / FaIR style)
    'aci_scale': -1.5066,      # Beta (scale factor)
    'aci_sensitivity': jnp.array([0.0, 0.0, 0.0, 0.0052, 0.0]), # Often just SO2
    'aci_em_indices': jnp.array([False, False, False, True, False]),
    'aci_conc_indices':jnp.array([False, False, False, False, False]),
    'forcing_scaling_aci': 1.0,
    'forcing_scaling_ari': 1.0,

    # Meinshausen 2020 CO2 parameters
    'm_a1': -2.4785e-07, 'm_b1': 0.00075906, 'm_c1': -0.0021492, 'm_d1': 5.2488, # CO2
    #'m_a1': -2.4785e-07, 'm_b1': 0.00075906, 'm_c1': -0.0021492, 'm_d1': 5.2488, # CO2
}

SCALES = {
    'ocean': 1,#0.1,         # Log-space scale
    'iirf': 1,#2.0,          # Physical space scale
    'forcing_val': 1,#0.05,  # Scaling factors (1.0 +/- 0.05)
    'mein_large': 1,#0.1,    # For d1, d2, d3 (magnitude ~5.0)
    'mein_small': 1,#1e-4,   # For b1, b2, b3 (magnitude ~1e-3)
    'mein_tiny': 1,#1e-7     # For a1, a2, a3 (magnitude ~1e-7)
}

def make_theta0(mode='FaIR'):
    """
    Consolidates all climate, carbon-cycle, and chemical parameters into theta.
    """
    if mode == 'FaIR':
        params = params_FaIR
    else:
        raise ValueError(f'Error, mode {mode} not recognized.')

    #return jnp.zeros(24)
    return jnp.concatenate([
        # 0-6: Ocean Physics (Log-space for positivity)
        jnp.log(params['ocean_heat_capacity']),
        jnp.log(params['ocean_heat_transfer']),
        jnp.log(jnp.array([params['deep_ocean_efficacy']])),
        # 7-12: CO2 iIRF and Scaling
        jnp.array([params['iirf_0'], params['iirf_uptake'], params['iirf_temperature'],
                   params['iirf_airborne'], params['g0'], params['g1']]),
        # 13-14: CH4 Feedback
        jnp.array([params['ch4_lifetime_chemical_sensitivity'][1],
                   params['ch4_lifetime_temperature_sensitivity']]),
        # 15-17: Radiative Forcing Scalings (CO2, CH4, N2O)
        params['forcing_scaling'],
        # 18-20: Radiative Efficiencies
        jnp.log(params['radiative_efficiency']),
        # 21-24: Meinshausen parameters (CO2)
        jnp.array([params['m_a1'], params['m_b1'], params['m_c1'], params['m_d1']])
    ])
"""
def params_from_theta(theta, base_params):
    p = dict(base_params)

    # --- 1. Ocean Heat (Log-Space Update) ---
    # Physical = Base * exp(theta * scale)
    # This keeps values positive and handles the O(100) magnitude difference.
    p['ocean_heat_capacity'] = base_params['ocean_heat_capacity'] * jnp.exp(theta[0:3] * SCALES['ocean'])
    p['ocean_heat_transfer'] = base_params['ocean_heat_transfer'] * jnp.exp(theta[3:6] * SCALES['ocean'])
    p['deep_ocean_efficacy'] = base_params['deep_ocean_efficacy'] * jnp.exp(theta[6] * SCALES['ocean'])

    # --- 2. Carbon Cycle (Linear Update with constraints) ---
    # Physical = Base + (theta * scale)
    # We use softplus/abs on critical terms to prevent negative values/division by zero

    # iIRF Parameters (Index 7-10)
    p['iirf_0'] = base_params['iirf_0'] + theta[7] * SCALES['iirf']
    p['iirf_uptake'] = base_params['iirf_uptake'] + theta[8] * (SCALES['iirf'] * 1e-3)
    p['iirf_temperature'] = base_params['iirf_temperature'] + theta[9] * SCALES['iirf']
    p['iirf_airborne'] = base_params['iirf_airborne'] + theta[10] * (SCALES['iirf'] * 1e-3)

    # g0, g1 (Index 11, 12) - Critical for stability
    # Use additive update, but enforce positivity on the result
    raw_g0 = base_params['g0'] + theta[11] * 1e-3
    raw_g1 = base_params['g1'] + theta[12] * 1.0
    p['g0'] = jax.nn.softplus(raw_g0) # Softplus is smooth ReLU (never 0 or negative)
    p['g1'] = jax.nn.softplus(raw_g1)

    # --- 3. CH4 Feedbacks (Index 13-14) ---
    p['ch4_lifetime_chemical_sensitivity'] = p['ch4_lifetime_chemical_sensitivity'].at[1].set(
        base_params['ch4_lifetime_chemical_sensitivity'][1] + theta[13] * 1e-4
    )
    p['ch4_lifetime_temperature_sensitivity'] = base_params['ch4_lifetime_temperature_sensitivity'] + theta[14] * 1e-2

    # --- 4. Forcing Scaling (Index 15-17) ---
    p['forcing_scaling'] = base_params['forcing_scaling'] + theta[15:18] * SCALES['forcing_val']

    # --- 5. Radiative Efficiency (Index 18-20) ---
    # Log-space update ensures positivity
    p['radiative_efficiency'] = base_params['radiative_efficiency'] * jnp.exp(theta[18:21] * SCALES['ocean'])

    # --- 6. Meinshausen Coeffs (Index 21-24) ---
    p['m_a1'] = base_params['m_a1'] + theta[21] * SCALES['mein_tiny']
    p['m_b1'] = base_params['m_b1'] + theta[22] * SCALES['mein_small']
    p['m_c1'] = base_params['m_c1'] + theta[23] * SCALES['mein_small']
    p['m_d1'] = base_params['m_d1'] + theta[24] * SCALES['mein_large']

    return p
"""
def params_from_theta(theta, base_params):
    p = dict(base_params)
    p['ocean_heat_capacity'] = jnp.exp(theta[0:3])
    p['ocean_heat_transfer'] = jnp.exp(theta[3:6])
    p['deep_ocean_efficacy'] = jnp.exp(theta[6])
    p['iirf_0'], p['iirf_uptake'], p['iirf_temperature'] = theta[7], theta[8], theta[9]
    p['iirf_airborne'], p['g0'], p['g1'] = theta[10], theta[11], theta[12]

    # CH4 Feedback
    p['ch4_lifetime_chemical_sensitivity'] = p['ch4_lifetime_chemical_sensitivity'].at[1].set(theta[13])
    p['ch4_lifetime_temperature_sensitivity'] = theta[14]

    # Radiative
    p['forcing_scaling'] = theta[15:18]
    p['radiative_efficiency'] = jnp.exp(theta[18:21])

    # Meinshausen 2020
    p['m_a1'], p['m_b1'], p['m_c1'], p['m_d1'] = theta[21], theta[22], theta[23], theta[24]
    return p

def make_theta_mask(theta, target='CO2'):
    """
    Returns a mask for the 21-parameter theta vector.
    Targets: 'CO2', 'CH4', 'N2O', 'Climate', or 'All'.
    """
    mask = jnp.zeros_like(theta)

    # 1. Global Climate (Ocean layers) - often tuned with CO2
    if target in ['Climate', 'All']:
        mask = mask.at[0:7].set(1.0)
        mask = mask.at[15].set(1.0)   # Forcing scaling
        #mask = mask.at[21:24].set(1.0) # Meinshausen 2020

    # 2. CO2 Specific (iIRF + Forcing)
    if target in ['CO2', 'All']:
        mask = mask.at[7:13].set(1.0) # iIRF and g-factors
        mask = mask.at[15].set(1.0)   # Forcing scaling
        mask = mask.at[18].set(1.0)   # Radiative efficiency

    # 3. CH4 Specific (Chemistry + Forcing)
    if target in ['CH4', 'All']:
        mask = mask.at[13:15].set(1.0) # Chemical/Temp sensitivities
        mask = mask.at[16].set(1.0)    # Forcing scaling
        mask = mask.at[19].set(1.0)    # Radiative efficiency

    # 4. N2O Specific (Forcing only)
    if target in ['N2O', 'All']:
        mask = mask.at[17].set(1.0)    # Forcing scaling
        mask = mask.at[20].set(1.0)    # Radiative efficiency

    return mask

def calculate_alpha(airborne_em, cum_em, g0, g1, iirf_0, iirf_airborne, iirf_temp, iirf_uptake, temp, iirf_max):
    iirf = iirf_0 + iirf_uptake * (cum_em - airborne_em) + iirf_temp * temp + iirf_airborne * airborne_em
    iirf = jnp.minimum(iirf, iirf_max)
    alpha = g0 * jnp.exp(iirf / g1)

    return jnp.where(jnp.isnan(alpha), 1.0, alpha)

def calculate_alpha_ch4(emissions, concentration, temperature, baseline_em, baseline_conc,
                        chem_sens, temp_sens, em_idx, conc_idx):
    # Convert boolean masks to floats (0.0 or 1.0) for element-wise math
    em_mask = em_idx.astype(jnp.float32)
    conc_mask = conc_idx.astype(jnp.float32)

    # Calculate differences across all indices, then zero out non-relevant ones
    em_diff = (emissions - baseline_em) * em_mask
    # Sum across the species axis to get the scalar value for CH4
    log_em = jnp.log(jnp.maximum(1e-10, 1.0 + jnp.sum(em_diff * chem_sens, axis=-1, keepdims=True)))

    conc_diff = (concentration - baseline_conc) * conc_mask
    log_conc = jnp.sum(jnp.log(jnp.maximum(1e-10, 1.0 + conc_diff * chem_sens)), axis=-1, keepdims=True)

    log_temp = jnp.log(jnp.maximum(1e-10, 1.0 + temperature * temp_sens))

    return jnp.exp(log_em + log_conc + log_temp).squeeze()

def step_concentration(emissions, gasboxes_old, airborne_old, alpha, base_conc, base_em, conc_per_em, lifetime, partition, timestep):
    # Broadcasting alpha (3,) against lifetime (3, 4)
    decay_rate = timestep / (alpha[:, None] * lifetime)
    decay_factor = jnp.exp(-decay_rate)

    # Broadcasting emissions (3,) against partition (3, 4)
    em_excess = (emissions - base_em)[:, None]

    gasboxes_new = partition * em_excess * (1/decay_rate) * (1 - decay_factor) * timestep + gasboxes_old * decay_factor
    airborne_new = jnp.sum(gasboxes_new, axis=-1)
    conc_out = base_conc + conc_per_em * airborne_new
    return conc_out, gasboxes_new, airborne_new

def meinshausen2020(conc, conc_base, scale, rad_eff, co2_idx, ch4_idx, n2o_idx, minor_idx,
                    a1=-2.4785e-07, b1=0.00075906, c1=-0.0021492, d1=5.2488,
                    a2=-0.00034197, b2=0.00025455, c2=-0.00024357, d2=0.12173,
                    a3=-8.9603e-05, b3=-0.00012462, d3=0.045194):

    co2, ch4, n2o = conc[..., co2_idx], conc[..., ch4_idx], conc[..., n2o_idx]
    co2_b, ch4_b, n2o_b = conc_base[..., co2_idx], conc_base[..., ch4_idx], conc_base[..., n2o_idx]

    # CO2 calculation
    ca_max = co2_b - b1 / (2 * a1)
    alpha_p = jnp.where(co2 <= co2_b, d1, jnp.where(co2 <= ca_max, d1 + a1*(co2-co2_b)**2 + b1*(co2-co2_b), d1 - b1**2/(4*a1)))
    erf_co2 = (alpha_p + c1*jnp.sqrt(n2o)) * jnp.log(co2/co2_b) * scale[..., co2_idx]

    # CH4 and N2O
    erf_ch4 = (a3*jnp.sqrt(ch4) + b3*jnp.sqrt(n2o) + d3) * (jnp.sqrt(ch4)-jnp.sqrt(ch4_b)) * scale[..., ch4_idx]
    erf_n2o = (a2*jnp.sqrt(co2) + b2*jnp.sqrt(n2o) + c2*jnp.sqrt(ch4) + d2) * (jnp.sqrt(n2o)-jnp.sqrt(n2o_b)) * scale[..., n2o_idx]

    erf_out = jnp.zeros_like(conc)
    return erf_out.at[..., co2_idx].set(erf_co2).at[..., ch4_idx].set(erf_ch4).at[..., n2o_idx].set(erf_n2o)

def calculate_erfari_linear(
    emissions,
    concentration,
    baseline_emissions,
    baseline_concentration,
    forcing_scaling,
    radiative_efficiency,
    emissions_indices,
    concentration_indices
):
    """
    Calculate ERFari using linear radiative efficiency.
    F = sum(RE_i * (E_i - E_base_i))
    """
    # Emissions-driven ARI (e.g., SO2, BC, OC)
    erf_em = jnp.sum(
        emissions_indices * (emissions - baseline_emissions) * radiative_efficiency,
        axis=-1
    )

    # Concentration-driven ARI (e.g., CH4, N2O)
    erf_conc = jnp.sum(
        concentration_indices * (concentration - baseline_concentration) * radiative_efficiency,
        axis=-1
    )

    return (erf_em + erf_conc) * forcing_scaling

def calculate_erfaci_logsum(
    emissions,
    concentration,
    baseline_emissions,
    baseline_concentration,
    forcing_scaling,
    scale,
    sensitivity,
    slcf_indices,
    ghg_indices
):
    """
    Calculate ERFaci using the logarithmic relationship.
    F = beta * log(1 + sum(s_i * A_i))
    """
    # Calculate effect for current state
    # We use jnp.sum with a mask (indices) instead of nansum for better JIT performance
    total_sensitivity_state = (
        jnp.sum(slcf_indices * sensitivity * emissions, axis=-1) +
        jnp.sum(ghg_indices * sensitivity * concentration, axis=-1)
    )
    radiative_effect = scale * jnp.log(jnp.maximum(1e-10, 1.0 + total_sensitivity_state))

    # Calculate effect for baseline
    total_sensitivity_base = (
        jnp.sum(slcf_indices * sensitivity * baseline_emissions, axis=-1) +
        jnp.sum(ghg_indices * sensitivity * baseline_concentration, axis=-1)
    )
    baseline_radiative_effect = scale * jnp.log(jnp.maximum(1e-10, 1.0 + total_sensitivity_base))

    # Difference is the forcing
    erf_out = (radiative_effect - baseline_radiative_effect) * forcing_scaling
    return erf_out

def get_fair_jax(mode='FaIR', params_in=None):
    """
    Returns three JAX-JIT compiled functions for different simulation modes:
    1. conc_to_temp: (Years, 3) Concentrations -> (Years,) Temperature
    2. em_to_conc: (Years, 3) Emissions & (Years,) Temp -> (Years, 3) Concentrations
    3. em_to_temp: (Years, 3) Emissions -> (Years,) Temperature
    """
    if mode == 'FaIR':
        params = params_FaIR
    elif mode == 'Calibrate':
        params = params_in
    else:
        raise ValueError(f'Error, mode {mode} not recognized.')

    # --- PRE-COMPUTE SHARED PHYSICS ---
    n = len(params['ocean_heat_capacity'])
    c, k = params['ocean_heat_capacity'], params['ocean_heat_transfer']

    def build_A():
        eps = jnp.ones(n).at[n-2].set(params['deep_ocean_efficacy'])
        A = jnp.zeros((n, n))
        A = A.at[0, 0].set(-(k[0] + eps[0]*k[1]) / c[0])
        A = A.at[0, 1].set(eps[0]*k[1] / c[0])
        A = A.at[-1, -2].set(k[-1] / c[-1])
        A = A.at[-1, -1].set(-k[-1] / c[-1])

        def body_fun(i, m):
            m = m.at[i, i-1].set(k[i] / c[i])
            m = m.at[i, i].set(-(k[i] + eps[i]*k[i+1]) / c[i])
            m = m.at[i, i+1].set(eps[i]*k[i+1] / c[i])
            return m

        return lax.fori_loop(1, n-1, body_fun, A)

    A_mat = build_A()
    eb_matrix_d = jax.scipy.linalg.expm(A_mat * params['timestep'])
    forcing_vector_d = jax.scipy.linalg.solve(
        A_mat,
        (eb_matrix_d - jnp.eye(n)) @ jnp.zeros(n).at[0].set(1.0/c[0])
    )

    # --- FUNCTION 1: CONCENTRATION TO TEMPERATURE ---
    @jax.jit
    def conc_to_temp(concentration_series):
        def body_fun(temp_layers, conc_t):
            erf = meinshausen2020(conc_t, params['baseline_concentration'], params['forcing_scaling'],
                                  params['radiative_efficiency'], params['co2_indices'], params['ch4_indices'],
                                  params['n2o_indices'], params['minor_ghg_indices'])
            temp_next = eb_matrix_d @ temp_layers + forcing_vector_d * jnp.sum(erf)
            return temp_next, temp_next[0]

        init_temp = jnp.zeros(n)
        _, t_history = lax.scan(body_fun, init_temp, concentration_series)
        return t_history

    # --- FUNCTION 2: EMISSIONS TO CONCENTRATION ---
    @jax.jit
    def em_to_conc(emissions_series, temperature_series):
        def body_fun(carry, inputs):
            (gas_boxes, airborne_em, cum_em) = carry
            em_t, t_surf = inputs

            # Feedbacks
            a_co2 = calculate_alpha(airborne_em[0], cum_em[0], params['g0'], params['g1'], params['iirf_0'],
                                    params['iirf_airborne'], params['iirf_temperature'], params['iirf_uptake'], t_surf, params['iirf_max'])

            prev_total_conc = params['baseline_concentration'] + params['concentration_per_emission'] * airborne_em

            a_ch4 = calculate_alpha_ch4(em_t, prev_total_conc, t_surf, params['baseline_emissions'], params['baseline_concentration'],
                                        params['ch4_lifetime_chemical_sensitivity'], params['ch4_lifetime_temperature_sensitivity'],
                                        params['emissions_indices'], params['concentration_indices'])

            alphas = jnp.ones(params['n_ghgs']).at[0].set(a_co2).at[1].set(a_ch4)

            conc_t, boxes_next, airborne_next = step_concentration(em_t, gas_boxes, airborne_em, alphas, params['baseline_concentration'],
                                                                   params['baseline_emissions'], params['concentration_per_emission'],
                                                                   params['lifetime'], params['partition_fraction'], params['timestep'])

            return (boxes_next, airborne_next, cum_em + em_t), conc_t

        init_gas = (jnp.zeros((params['n_ghgs'], params['n_gas_boxes'])), jnp.zeros(params['n_ghgs']), jnp.zeros(params['n_ghgs']))
        _, conc_history = lax.scan(body_fun, init_gas, (emissions_series, temperature_series))
        return conc_history

    # --- FUNCTION 3: EMISSIONS TO TEMPERATURE ---
    @jax.jit
    def em_to_temp(emissions_series):
        def body_fun(carry, em_t):
            (gas_boxes, airborne_em, cum_em, temp_layers) = carry
            t_surf = temp_layers[0]

            a_co2 = calculate_alpha(airborne_em[0], cum_em[0], params['g0'], params['g1'], params['iirf_0'],
                                    params['iirf_airborne'], params['iirf_temperature'], params['iirf_uptake'], t_surf, params['iirf_max'])

            prev_total_conc = params['baseline_concentration'] + params['concentration_per_emission'] * airborne_em

            a_ch4 = calculate_alpha_ch4(em_t, prev_total_conc, t_surf, params['baseline_emissions'], params['baseline_concentration'],
                                        params['ch4_lifetime_chemical_sensitivity'], params['ch4_lifetime_temperature_sensitivity'],
                                        params['emissions_indices'], params['concentration_indices'])

            alphas = jnp.ones(params['n_ghgs']).at[0].set(a_co2).at[1].set(a_ch4)

            conc_t, boxes_next, airborne_next = step_concentration(em_t, gas_boxes, airborne_em, alphas, params['baseline_concentration'],
                                                                   params['baseline_emissions'], params['concentration_per_emission'],
                                                                   params['lifetime'], params['partition_fraction'], params['timestep'])

            erf = meinshausen2020(conc_t, params['baseline_concentration'], params['forcing_scaling'],
                                  params['radiative_efficiency'], params['co2_indices'], params['ch4_indices'],
                                  params['n2o_indices'], params['minor_ghg_indices'])

            temp_next = eb_matrix_d @ temp_layers + forcing_vector_d * jnp.sum(erf)

            return (boxes_next, airborne_next, cum_em + em_t, temp_next), temp_next[0]

        init_state = (jnp.zeros((params['n_ghgs'], params['n_gas_boxes'])), jnp.zeros(params['n_ghgs']),
                      jnp.zeros(params['n_ghgs']), jnp.zeros(n))
        _, t_history = lax.scan(body_fun, init_state, emissions_series)
        return t_history

    return conc_to_temp, em_to_conc, em_to_temp

needs_historical = ['H-ext','H-ext-OS','M',
                    'M-ext','ML','ML-ext','L',
                    'L-ext','VLLO-ext','VLHO','VLHO-ext']

def run_scenarios(emis_dict, model_func):
    """
    Transforms a nested dictionary of emissions into matrices and runs the FaIR model.

    Parameters:
    ----------
    scenarios_dict : dict
        Format: { 'scenario_name': { 'CO2': [...], 'CH4': [...], ... } }
    model_func : function
        The JIT-compiled model function returned by get_fair_emulator.

    Returns:
    -------
    results : dict
        Format: { 'scenario_name': array([temp_series]) }
    """
    results = {}

    # Mapping of keys to columns based on the model's species indices
    # 0: CO2, 1: CH4, 2: N2O
    agent_mapping = {"CO2": 0, "CH4": 1, "N2O": 2, "Sulfur": 3, "BC": 4}
    historical_agents = emis_dict.get('historical', {})

    for scenario_name, agents in emis_dict.items():
        # 1. Determine the length of the simulation (years)
        # We look at the first available gas list to find the length
        check_agent = next(iter(agents))
        scenario_vals = jnp.array(agents[check_agent])

        if scenario_name in needs_historical:
            hist_vals = jnp.array(historical_agents.get(check_agent, []))
            total_years = len(hist_vals) + len(scenario_vals)
        else:
            total_years = len(scenario_vals)

        # 2. Create an empty emissions matrix (Years, 3)
        # We use jnp.zeros so that missing gases default to no emissions
        em_matrix = jnp.zeros((total_years, 5))

        # 3. Fill the matrix with available data
        for a, col_idx in agent_mapping.items():
            current_data = jnp.array(agents.get(a, jnp.zeros_like(scenario_vals)))

            if scenario_name in needs_historical:
                # Get historical data for this agent (default to zeros if missing)
                h_data = jnp.array(historical_agents.get(a, jnp.zeros(len(hist_vals))))
                combined_data = jnp.concatenate([h_data, current_data])
            else:
                combined_data = current_data

            em_matrix = em_matrix.at[:, col_idx].set(combined_data)

        # 4. Run the model and store the result
        # The output is a (n_years,) temperature array
        results[scenario_name] = model_func(em_matrix)

    return results



def fair_emulator_core(emissions_series, params):
    """
    Functional FaIR core. Differentiable and VMAP-compatible.
    Input: (Years, 3) emissions, (Dict) params
    """
    n = len(params['ocean_heat_capacity'])
    n_ghgs = params['n_ghgs']
    init_state = (
        jnp.zeros((n_ghgs, params['n_gas_boxes'])), # gas boxes for CO2, CH4, N2O
        jnp.zeros(n_ghgs),                          # airborne excess for CO2, CH4, N2O
        jnp.zeros(n_ghgs),                          # cumulative emissions for CO2, CH4, N2O
        jnp.zeros(n)                                # temperature layers
    )

    # 1. Discretize Physics (Inside to allow parameter gradients)
    c, k = params['ocean_heat_capacity'], params['ocean_heat_transfer']

    def build_A():
        eps = jnp.ones(n).at[n-2].set(params['deep_ocean_efficacy'])
        A = jnp.zeros((n, n))
        A = A.at[0, 0].set(-(k[0] + eps[0]*k[1]) / c[0])
        A = A.at[0, 1].set(eps[0]*k[1] / c[0])
        A = A.at[-1, -2].set(k[-1] / c[-1])
        A = A.at[-1, -1].set(-k[-1] / c[-1])
        def body_fun(i, m):
            return m.at[i, i-1].set(k[i]/c[i]).at[i, i].set(-(k[i]+eps[i]*k[i+1])/c[i]).at[i, i+1].set(eps[i]*k[i+1]/c[i])
        return lax.fori_loop(1, n-1, body_fun, A)

    A_mat = build_A()
    eb_matrix_d = jax.scipy.linalg.expm(A_mat * params['timestep'])
    B_cont = jnp.zeros(n).at[0].set(1.0 / c[0])
    forcing_vector_d = jax.scipy.linalg.solve(A_mat, (eb_matrix_d - jnp.eye(n)) @ B_cont)

    # 2. Step Function
    def fair_step(carry, em_t):
        (gas_boxes, airborne_em, cum_em, temp_layers) = carry
        t_surf = temp_layers[0]
        em_ghg = em_t[:3]

        total_conc_full = params['baseline_concentration'].at[:3].add(params['concentration_per_emission'][:3] * airborne_em)
        total_conc_ghg = total_conc_full[:3]

        # Feedback logic (Alpha)
        a_co2 = calculate_alpha(airborne_em[0], cum_em[0], params['g0'], params['g1'],
                                params['iirf_0'], params['iirf_airborne'],
                                params['iirf_temperature'], params['iirf_uptake'], t_surf, params['iirf_max'])

        a_ch4 = calculate_alpha_ch4(em_ghg,
                                    total_conc_ghg,
                                    t_surf,
                                    params['baseline_emissions'][:3],
                                    params['baseline_concentration'][:3],
                                    params['ch4_lifetime_chemical_sensitivity'][:3],
                                    params['ch4_lifetime_temperature_sensitivity'],
                                    params['emissions_indices'][:3],
                                    params['concentration_indices'][:3])

        alphas = jnp.ones(params['n_ghgs']).at[0].set(a_co2).at[1].set(a_ch4)

        # Decay, Forcing, and Temperature
        conc_t, boxes_next, airborne_next = step_concentration(em_ghg, gas_boxes, airborne_em, alphas,
                                                               params['baseline_concentration'][:3], params['baseline_emissions'][:3],
                                                               params['concentration_per_emission'][:3], params['lifetime'],
                                                               params['partition_fraction'], params['timestep'])

        erf_ghg = meinshausen2020(conc_t, params['baseline_concentration'][:3], params['forcing_scaling'],
                                  params['radiative_efficiency'], params['co2_indices'], params['ch4_indices'],
                                  params['n2o_indices'], params['minor_ghg_indices'],
                                  a1=params['m_a1'], b1=params['m_b1'], c1=params['m_c1'], d1=params['m_d1'])

        erf_ari = calculate_erfari_linear(
            em_t, total_conc_full, params['baseline_emissions'], params['baseline_concentration'],
            params['forcing_scaling_ari'], params['erfari_radiative_efficiency'],
            params['ari_em_indices'], params['ari_conc_indices']
        )

        erf_aci = calculate_erfaci_logsum(
            em_t, total_conc_full, params['baseline_emissions'], params['baseline_concentration'],
            params['forcing_scaling_aci'], params['aci_scale'], params['aci_sensitivity'],
            params['aci_em_indices'], params['aci_conc_indices']
        )

        erf_tot = jnp.sum(erf_ghg) + erf_ari + erf_aci

        temp_next = eb_matrix_d @ temp_layers + forcing_vector_d * erf_tot
        return (boxes_next, airborne_next, cum_em + em_ghg, temp_next), temp_next[0]

    init_state = (jnp.zeros((params['n_ghgs'], params['n_gas_boxes'])), jnp.zeros(params['n_ghgs']),
                  jnp.zeros(params['n_ghgs']), jnp.zeros(n))

    _, t_history = lax.scan(fair_step, init_state, emissions_series)
    return t_history


def fair_concentration_core(concentration_series, params):
    """
    Differentiable concentration-to-temperature core.
    Input: concentration_series (Years, 3) -> CO2, CH4, N2O
    """
    n = len(params['ocean_heat_capacity'])

    # 1. Discretize Physics (allowing gradients for ocean parameters)
    c, k = params['ocean_heat_capacity'], params['ocean_heat_transfer']

    def build_A():
        eps = jnp.ones(n).at[n-2].set(params['deep_ocean_efficacy'])
        A = jnp.zeros((n, n))
        A = A.at[0, 0].set(-(k[0] + eps[0]*k[1]) / c[0])
        A = A.at[0, 1].set(eps[0]*k[1] / c[0])
        A = A.at[-1, -2].set(k[-1] / c[-1])
        A = A.at[-1, -1].set(-k[-1] / c[-1])
        def body_fun(i, m):
            return m.at[i, i-1].set(k[i]/c[i]).at[i, i].set(-(k[i]+eps[i]*k[i+1])/c[i]).at[i, i+1].set(eps[i]*k[i+1]/c[i])
        return lax.fori_loop(1, n-1, body_fun, A)

    A_mat = build_A()
    eb_matrix_d = jax.scipy.linalg.expm(A_mat * params['timestep'])
    B_cont = jnp.zeros(n).at[0].set(1.0 / c[0])
    forcing_vector_d = jax.scipy.linalg.solve(A_mat, (eb_matrix_d - jnp.eye(n)) @ B_cont)

    # 2. Iterative Temperature Step
    def temp_step(temp_layers, conc_t):
        # Calculate Greenhouse Gas forcing
        erf_ghg = meinshausen2020(
            conc_t,
            params['baseline_concentration'][:3],
            params['forcing_scaling'],
            params['radiative_efficiency'],
            params['co2_indices'],
            params['ch4_indices'],
            params['n2o_indices'],
            params['minor_ghg_indices'],
            a1=params['m_a1'], b1=params['m_b1'], c1=params['m_c1'], d1=params['m_d1']
        )

        # Total forcing (using prescribed concentrations only)
        erf_tot = jnp.sum(erf_ghg)

        temp_next = eb_matrix_d @ temp_layers + forcing_vector_d * erf_tot
        return temp_next, temp_next[0]

    init_temp = jnp.zeros(n)
    _, t_history = lax.scan(temp_step, init_temp, concentration_series)
    return t_history


def run_scenarios_batched(emis_dict, params):
    """
    Refactored for 5-species: CO2, CH4, N2O, Sulfur, BC
    """
    scen_names = [k for k in emis_dict.keys() if k != 'historical']
    historical_agents = emis_dict.get('historical', {})

    # Mapping for 5 species
    agent_mapping = {"CO2": 0, "CH4": 1, "N2O": 2, "Sulfur": 3, "BC": 4}

    # 1. Determine max length across all combined (hist + future) scenarios
    max_len = 0
    h_len_global = len(next(iter(historical_agents.values()))) if historical_agents else 0
    scen_info = []
    for name in scen_names:
        any_f_agent = next(iter(emis_dict[name].values()))
        f_len = len(any_f_agent)
        h_len = h_len_global if name in needs_historical else 0
        scen_info.append(h_len + f_len)
        max_len = max(max_len, h_len + f_len)

    # 2. Build padded matrix (Scenarios, Time, 5)
    padded_emissions = jnp.zeros((len(scen_names), max_len, 5))

    for i, name in enumerate(scen_names):
        h_len = h_len_global if name in needs_historical else 0
        for gas, col in agent_mapping.items():
            f_data = jnp.array(emis_dict[name].get(gas, []))
            if name in needs_historical:
                h_data = jnp.array(historical_agents.get(gas, jnp.zeros(h_len)))
                combined = jnp.concatenate([h_data, f_data])
            else:
                combined = f_data

            # Place data into the matrix
            padded_emissions = padded_emissions.at[i, :len(combined), col].set(combined)

    # 3. Parallel Execution via vmap
    # We vmap over the scenario axis (axis 0) of the emissions matrix
    vmapped_model = jax.vmap(fair_emulator_core, in_axes=(0, None))
    all_temps = vmapped_model(padded_emissions, params)

    return {name: all_temps[j, :scen_info[j]] for j, name in enumerate(scen_names)}


# -----------------
# Model calibration
# -----------------

def build_emissions_by_agent(emis_dict_FaIR, agent_order=('CO2', 'CH4', 'N2O', 'Sulfur', 'BC')):
  """
  Convert emis_dict_JAX[scenario][agent] -> emissions_by_agent[scenario].

  For each scenario:
    - Create an array of shape (N_agents, T) where rows follow `agent_order`.
    - If an agent in `agent_order` is missing for that scenario, its row is zeros.
  """
  emis_by_agent = {}

  for scen, scen_emis in emis_dict_FaIR.items():
    if not scen_emis:
      raise ValueError(f"No emissions found for scenario '{scen}'")

    # Infer T from any available agent
    first_agent = next(iter(scen_emis.keys()))
    T = jnp.asarray(scen_emis[first_agent]).shape[0]

    rows = []
    for agent in agent_order:
      if agent in scen_emis:
        arr = jnp.asarray(scen_emis[agent], dtype=jnp.float32)
        if arr.shape[0] != T:
          raise ValueError(
            f"Length mismatch for agent '{agent}' in scenario '{scen}': "
            f"expected T={T}, got {arr.shape[0]}"
          )
      else:
        # Missing agent -> zero emissions
        arr = jnp.zeros((T,), dtype=jnp.float32)
      rows.append(arr)

    emis_by_agent[scen] = jnp.stack(rows, axis=0)  # (N_agents, T)

  return emis_by_agent

def generate_train_test(agents):

  emis_dict_train_FaIR, emis_dict_test_FaIR = run_fair.load_scenarioMIP_CMIP7(agents)
  emis_dict_train_JAX = build_emissions_by_agent(emis_dict_train_FaIR)
  emis_dict_test_JAX = build_emissions_by_agent(emis_dict_test_FaIR)

  return emis_dict_train_JAX, emis_dict_test_JAX


def generate_calib_data(agents, mode='FaIR', params_in=None):

  if mode == 'FaIR':
      params = params_FaIR
  elif mode == 'Calibrate':
      params = params_in
  else:
      raise ValueError(f'Error, mode {mode} not recognized.')

  emis_dict_tier1, emis_dict_tier2 = run_fair.load_scenarioMIP_CMIP7(agents)
  delT_dict_tier1 = run_fair.get_delT(emis_dict_tier1, list(emis_dict_tier1.keys()), agents, MIP='ScenarioMIP_tier1')
  delT_dict_tier2 = run_fair.get_delT(emis_dict_tier2, list(emis_dict_tier2.keys()), agents, MIP='ScenarioMIP_tier2')

  emis_dict_DECK = run_fair.load_DECK_CMIP7(agents)
  delT_dict_DECK = run_fair.get_delT(emis_dict_DECK, list(emis_dict_DECK.keys()), agents, MIP='DECK')

  emis_dict_DECK_subset = {}
  delT_dict_DECK_subset = {}
  for scen in emis_dict_DECK:
    if any(a in scen for a in agents):
      emis_dict_DECK_subset[scen] = {}
      for a in agents:
        emis_dict_DECK_subset[scen][a] = emis_dict_DECK[scen][a].copy()
        delT_dict_DECK_subset[scen] = delT_dict_DECK[scen].copy()
    else:
      continue

  emis_dict_calib_FaIR = emis_dict_tier1 | emis_dict_tier2 | emis_dict_DECK_subset
  delT_dict_calib_FaIR = delT_dict_tier1 | delT_dict_tier2 | delT_dict_DECK_subset

  delT_dict_calib_JAX = run_scenarios_batched(emis_dict_calib_FaIR, params)
  emis_dict_calib_JAX = build_emissions_by_agent(emis_dict_calib_FaIR)

  return emis_dict_calib_FaIR, emis_dict_calib_JAX, delT_dict_calib_FaIR, delT_dict_calib_JAX

def generate_JAX_data(agents, CS3=False, DAMIP=False, GeoMIP=False):

  emis_dict_tier1_FaIR, emis_dict_tier2_FaIR = run_fair.load_scenarioMIP_CMIP7(agents)
  emis_dict_DECK_FaIR = run_fair.load_DECK_CMIP7(agents)

  emis_dict_DECK_subset = {}
  for scen in emis_dict_DECK_FaIR:
    if any(a in scen for a in agents):
      emis_dict_DECK_subset[scen] = {}
      for a in agents:
        emis_dict_DECK_subset[scen][a] = emis_dict_DECK_FaIR[scen][a].copy()
    else:
      continue

  emis_dict_tier1_JAX = build_emissions_by_agent(emis_dict_tier1_FaIR)
  emis_dict_tier2_JAX = build_emissions_by_agent(emis_dict_tier2_FaIR)
  emis_dict_DECK_JAX = build_emissions_by_agent(emis_dict_DECK_subset)

  emis_dicts = [emis_dict_tier1_JAX, emis_dict_tier2_JAX, emis_dict_DECK_JAX]

  if CS3:
    emis_dict_CS3_FaIR = run_fair.load_CS3(agents=agents, emis_dict_tier1=emis_dict_tier1_FaIR)

    for scen in emis_dict_CS3_FaIR:
        for a in emis_dict_CS3_FaIR[scen]:
            if a not in agents:
                emis_dict_CS3_FaIR[scen][a] = emis_dict_CS3_FaIR[scen][a] * 0

    emis_dict_CS3_JAX = build_emissions_by_agent(emis_dict_CS3_FaIR)
    emis_dicts.append(emis_dict_CS3_JAX)

  if DAMIP:
    M_GHG = jnp.concat([emis_dict_tier1_JAX['historical'], emis_dict_tier1_JAX['M']], axis=1)
    M_GHG = M_GHG.at[3:, :].set(M_GHG[3:, :] * 0)
    M_AER = jnp.concat([emis_dict_tier1_JAX['historical'], emis_dict_tier1_JAX['M']], axis=1)
    M_AER = M_AER.at[0:3, :].set(M_AER[0:3, :] * 0)
    emis_dict_DAMIP = {'M_GHG':M_GHG, 'M_AER':M_AER}
    emis_dicts.append(emis_dict_DAMIP)

  if GeoMIP:
    path_GEO = 'data/GeoMIP/emis_G6sulfur.pickle'
    with open(path_GEO, "rb") as f:
        emis_GEO = pickle.load(f)
    emis_dict_GEO = {'G6sulfur': emis_GEO}
    emis_dicts.append(emis_dict_GEO)

  return emis_dicts

def loss_fn(theta, emis_matrix, target_matrix, base_params):
    params = params_from_theta(theta, base_params)
    # The entire batch of scenarios is computed in one parallel XLA call
    preds = jax.vmap(fair_emulator_core, in_axes=(0, None))(emis_matrix, params)
    return jnp.mean((preds - target_matrix)**2)


def prepare_calibration_data(emis_dict, target_dict):
    """
    Helper to transform dictionaries into padded matrices for batched calibration.
    Independent of specific agents in the historical scenario.
    """
    scen_names = [k for k in emis_dict.keys() if k != 'historical']
    historical_agents = emis_dict.get('historical', {})
    agent_mapping = {"CO2": 0, "CH4": 1, "N2O": 2, "Sulfur": 3, "BC": 4}

    # 1. Determine max length across all combined (hist + future) scenarios
    # We find lengths by looking at the first available agent in each dict
    max_len = 0
    h_len_global = len(next(iter(historical_agents.values()))) if historical_agents else 0
    for name in scen_names:
        # Future length
        f_len = len(next(iter(emis_dict[name].values()))) # future length
        h_len = h_len_global if name in needs_historical else 0 # historical length

        total_len = h_len + f_len
        if total_len > max_len:
            max_len = total_len

    # 2. Build padded matrices
    padded_emis = jnp.zeros((len(scen_names), max_len, 5))
    padded_targets = jnp.zeros((len(scen_names), max_len))
    loss_mask = jnp.zeros((len(scen_names), max_len))

    for i, name in enumerate(scen_names):
        f_len = len(next(iter(emis_dict[name].values())))
        h_len = h_len_global if name in needs_historical else 0

        for gas, col in agent_mapping.items():
            f_data = jnp.array(emis_dict[name].get(gas, []))

            if name in needs_historical:
                # If historical exists, prepend it. If gas is missing from history, pad with zeros.
                h_data = jnp.array(historical_agents.get(gas, jnp.zeros(h_len)))
                combined = jnp.concatenate([h_data, f_data])
            else:
                combined = f_data

            padded_emis = padded_emis.at[i, :len(combined), col].set(combined)

        # Pad targets (FaIR baseline temperatures)
        target_series = jnp.array(target_dict.get(name, []))
        padded_targets = padded_targets.at[i, h_len : h_len + f_len].set(target_series)
        loss_mask = loss_mask.at[i, h_len : h_len + f_len].set(1.0)

    return padded_emis, padded_targets, loss_mask

def calibrate_inverse(filepath, emis_dict, target_dict, theta0, learning_rate, n_steps, target='CO2'):
    """
    Automated parameter calibration using Optax and JAX gradients.
    """
    # Prepare data once before the loop (extremely important for speed)
    emis_matrix, target_matrix, loss_mask = prepare_calibration_data(emis_dict, target_dict)

    # Define the loss function
    def loss_fn(theta):
        params = params_from_theta(theta, params_FaIR)
        # Run all scenarios in parallel
        preds = jax.vmap(fair_emulator_core, in_axes=(0, None))(emis_matrix, params)
        sq_error = (preds - target_matrix)**2
        total_error = jnp.sum(sq_error * loss_mask)
        return total_error / jnp.sum(loss_mask)

    # Gradient function
    loss_and_grad = jax.value_and_grad(loss_fn)

    # Initialize Optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(theta0)

    # Mask to freeze non-target parameters
    mask = make_theta_mask(theta0, target=target)
    theta = theta0

    print(f"Starting calibration for target: {target}")
    for step in range(n_steps):
        loss_val, grads = loss_and_grad(theta)

        # Apply mask to gradients
        grads = grads * mask
        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

        if step % 10 == 0 or step == n_steps - 1:
            print(f"Step {step:4d} | MSE Loss: {loss_val:.8f}")
            # Save progress
            with open(filepath, 'wb') as f:
                pickle.dump(theta, f)

    return params_from_theta(theta, params_FaIR), loss_val

def calibrate_concentration_driven(filepath, conc_matrix, target_temp_matrix, loss_mask, theta0, learning_rate=1e-3, n_steps=100):
    """
    Calibrates ocean parameters (Climate Sensitivity) using prescribed concentrations.
    """

    def loss_fn(theta):
        # Reconstruct params - gradients will flow into ocean parameters via theta[0:7]
        params = params_from_theta(theta, params_FaIR)

        # Parallel execution across scenarios
        preds = jax.vmap(fair_concentration_core, in_axes=(0, None))(conc_matrix, params)

        sq_error = (preds - target_temp_matrix)**2
        total_error = jnp.sum(sq_error * loss_mask)
        return total_error / jnp.sum(loss_mask)

    # Gradient and Optimizer setup
    loss_and_grad = jax.value_and_grad(loss_fn)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(theta0)

    # Create the mask specifically for 'Climate' parameters (Indices 0-6)
    # Based on your make_theta_mask: targets 0-2 (capacity), 3-5 (transfer), 6 (efficacy)
    update_mask = make_theta_mask(theta0, target='Climate')
    theta = theta0

    print("Commencing Climate Parameter Calibration (Concentration Driven)...")

    for step in range(n_steps):
        loss_val, grads = loss_and_grad(theta)

        # Apply mask: only allow updates to indices 0 through 6
        grads = grads * update_mask

        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

        if step % 20 == 0 or step == n_steps - 1:
            # Check a sample parameter to ensure other values (like iIRF) aren't changing
            print(f"Step {step:4d} | MSE: {loss_val:.8f}")

            with open(filepath, 'wb') as f:
                pickle.dump(theta, f)

    return theta, loss_val


def process_custom_concentrations(co2_scenarios_dict, mode='FaIR'):
    """
    Converts a dict of 1D arrays into JAX-ready matrices.
    co2_scenarios_dict: {'ScenarioA': array([...]), 'ScenarioB': array([...])}
    """
    if mode == 'FaIR':
        params = params_FaIR
    else:
        raise ValueError(f'Error, mode {mode} not recognized.')
    names = list(co2_scenarios_dict.keys())

    # 1. Determine dimensions
    max_len = max(len(arr) for arr in co2_scenarios_dict.values())
    n_scens = len(names)

    # 2. Initialize matrices
    # Shape: (Scenarios, Time, 3 species)
    baselines = params['baseline_concentration'][:3]
    conc_matrix = jnp.ones((n_scens, max_len, 3)) * baselines
    # Shape: (Scenarios, Time)
    loss_mask = jnp.zeros((n_scens, max_len))

    # Pre-fill with baselines for CH4 and N2O (indices 1 and 2)
    conc_matrix = conc_matrix.at[:, :, 1].set(params['baseline_concentration'][1])
    conc_matrix = conc_matrix.at[:, :, 2].set(params['baseline_concentration'][2])

    for i, name in enumerate(names):
        data = co2_scenarios_dict[name]
        curr_len = len(data)

        # Insert CO2 data into column 0
        conc_matrix = conc_matrix.at[i, :curr_len, 0].set(data)

        # Create mask: 1.0 for valid data, 0.0 for padding
        loss_mask = loss_mask.at[i, :curr_len].set(1.0)

    return conc_matrix, loss_mask

def process_custom_targets(target_temp_dict):
    """
    Converts a dict of 1D temperature arrays into a JAX-ready matrix.
    target_temp_dict: {'ScenarioA': array([...]), 'ScenarioB': array([...])}
    """
    names = list(target_temp_dict.keys())

    # 1. Determine dimensions (must match the concentration matrix)
    max_len = max(len(arr) for arr in target_temp_dict.values())
    n_scens = len(names)

    # 2. Initialize matrix
    # Shape: (Scenarios, Time)
    target_matrix = jnp.zeros((n_scens, max_len))


    for i, name in enumerate(names):
        data = jnp.array(target_temp_dict[name])
        curr_len = len(data)

        # Insert temperature data
        target_matrix = target_matrix.at[i, :curr_len].set(data)

    return target_matrix

def plot_calibration_results(conc_matrix, target_matrix, loss_mask, theta0, theta_opt, scenario_names):
    """
    Plots the model performance before and after optimization.
    """
    # 1. Generate predictions
    params_initial = params_from_theta(theta0, params_FaIR)
    params_optimized = params_from_theta(theta_opt, params_FaIR)

    # Vmap the core over the scenario axis (0)
    vmap_model = jax.vmap(fair_concentration_core, in_axes=(0, None))

    preds_init = vmap_model(conc_matrix, params_initial)
    preds_opt = vmap_model(conc_matrix, params_optimized)

    # 2. Plotting
    n_scens = len(scenario_names)
    fig, axes = plt.subplots(1, n_scens, figsize=(6 * n_scens, 5), sharey=True)
    if n_scens == 1: axes = [axes]

    for i, name in enumerate(scenario_names):
        ax = axes[i]
        actual_len = int(jnp.sum(loss_mask[i]))
        years = jnp.arange(actual_len)

        ax.plot(years, target_matrix[i, :actual_len], color='black', label='Target (Data)', lw=2)
        ax.plot(years, preds_init[i, :actual_len], color='tab:red', linestyle=':', label='FaIR (Pre-Opt)')
        ax.plot(years, preds_opt[i, :actual_len], color='tab:red', linestyle='--', label='FaIR (Post-Opt)')

        ax.set_title(f'Scenario: {name}')
        ax.set_xlabel('Years')
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(r'$\Delta T$ (K)')
    axes[0].legend()

    plt.tight_layout()



def solve_sulfur_inverse(
    emissions_H_matrix,  # (T, 5) array: Full emissions for Scenario H (Time-Major)
    target_temp_M,       # (T,) array: Target GMST from Scenario M
    params=None,         # Parameter dictionary
    learning_rate=0.1,
    n_steps=2000,
    reg_weight=0.1
):
    """
    Solves for sulfur emissions (column 3) that force the model to match target_temp_M,
    given background emissions from Scenario H.

    Args:
        emissions_H_matrix: (Time, 5) array. Columns: [CO2, CH4, N2O, Sulfur, BC].
                            Note: Transpose your data if you have (5, Time).
        target_temp_M: (Time,) array of target temperatures.
        params: FaIR parameter dictionary (defaults to params_FaIR if None).
        learning_rate: Optimizer step size.
        n_steps: Number of optimization steps.
        reg_weight: Penalty weight for jaggedness (first-difference regularization).
    """
    if params is None:
        params = params_FaIR

    # Index for Sulfur in the new 5-species layout
    # Mapping: {"CO2": 0, "CH4": 1, "N2O": 2, "Sulfur": 3, "BC": 4}
    idx_Sulfur = 3

    # 1. Initialization
    # Extract initial guess from the background scenario (Column 3)
    # Shape: (T,)
    sulfur_initial_guess = emissions_H_matrix[:, idx_Sulfur]

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(sulfur_initial_guess)

    # 2. Define the Loss Function
    def inverse_loss_fn(sulfur_profile):
        # Construct the full emissions matrix (T, 5)
        # We take H emissions and use .at[:, 3] to update the Sulfur column
        emissions_current = emissions_H_matrix.at[:, idx_Sulfur].set(sulfur_profile)

        # Run the forward model (New Core)
        # Returns shape (T,) - the full temperature history
        T_pred = fair_emulator_core(emissions_current, params)

        # A. Primary Loss: MSE on Temperature
        mse_loss = jnp.mean((T_pred - target_temp_M)**2)

        # B. Regularization: Penalize first-order differences to ensure smoothness
        diff = jnp.diff(sulfur_profile)
        smoothness_penalty = jnp.mean(diff**2)

        return mse_loss + (reg_weight * smoothness_penalty)

    # 3. Optimization Loop (JIT Compiled)
    @jax.jit
    def step(sulfur_params, opt_state):
        loss, grads = jax.value_and_grad(inverse_loss_fn)(sulfur_params)
        updates, opt_state = optimizer.update(grads, opt_state, sulfur_params)
        new_sulfur = optax.apply_updates(sulfur_params, updates)
        return new_sulfur, opt_state, loss

    current_sulfur = sulfur_initial_guess

    print(f"Starting Inverse Solve (Target: M Temp, Background: H Emis)...")
    for i in range(n_steps):
        current_sulfur, opt_state, loss_val = step(current_sulfur, opt_state)

        if i % 200 == 0:
            print(f"Step {i}: Loss = {loss_val:.6f}")

    print(f"Final Loss: {loss_val:.6f}")

    # Generate final result
    final_emissions = emissions_H_matrix.at[:, idx_Sulfur].set(current_sulfur)
    final_T_pred = fair_emulator_core(final_emissions, params)

    return current_sulfur, final_T_pred