# -------
# Imports
# -------
import matplotlib.pyplot as plt
import run_fair
import pickle

import numpy as np

## Misc.
from typing import Iterable

## Local
from paths import DATA_DIR

# JAX
import jax
import jax.numpy as jnp
from jax import lax
import optax

# ------------------
# Default parameters
# ------------------
FAIR_PARAMS = {
  "a": jnp.array([0.1017451, 0.1209856, 0.3535364, 0.42373294]), # Carbon reservoir fractions [-]
  # default: jnp.array([0.2173, 0.2240, 0.2824, 0.2763]),

  "tau": jnp.array([4.4574850e+05, 2.1955202e+02, 1.4740475e+01, 2.2657001e+00]), # Reservoir time scales [yr]
  # default: jnp.array([1e6, 394.4, 36.54, 4.304]),
  "ch4_tau": 10,                                  # CH4 lifetime
  "k_ch4": 0.046,                                   # Radiative forcing from CH4

  # --- FaIR-style iIRF100 / alpha parameters for CO2 ---
  # Millar et al. (2017) / Smith et al. (2017) defaults
  "r0": 35.0,          # yr
  "rC": 0.019,         # yr / GtC
  "rT": 4.165,         # yr / K

  # Integrated impulse response horizon & max (FaIR convention)
  "iirf_h": 100.0,     # years over which iIRF is defined
  "iirf_max": 97.0,    # max allowed iIRF (avoids no-solution regime)

  "ch4_rT": 0.04,            # Sensitivity of lifetime to global-mean T [1/K]
  "ch4_ra": 1e-4,            # Sensitivity of lifetime to CH4 burden (ppb^-1)

  # N2O cycle
  "n2o_tau": 99.86245,          # years, FaIRv2 Table S2, default 100
  "n2o_emis2conc": 0.201,    # ppb per TgN/yr
  "k_n2o": 0.06395961,            # W/m2 per sqrt(ppb) difference, default 0.106

  # d_j: response timescales [years] - calibrated w/ prescribed concentrations
  "d": jnp.array([2.8593524, 21.964226 ,255.70793]),

  # q_j: equilibrium response coefficients [K / (W m^-2)] - calibrated w/ prescribed concentrations
  "q": jnp.array([0.16457078, 0.42236426, 0.47316828]),

  "f2_SO2": -0.003, # aerosol-radiation interaction
  "f2_BC": 0.0315142, # aerosol-radiation interaction
  "f1_aci": -0.8848617, # aerosol-cloud interaction
  "C0_SO2": 24.358507, # SO2 shape parameter
  "f2_aci": -0.00948581, # aerosol-cloud interaction

  "C0_PI": 278.0 # Preindustrial CO2 [ppm]
}

MESM_PARAMS = {
  "a": jnp.array([0.04127917, 0.04904277, 0.14516222, 0.76451576]), # Carbon reservoir fractions [-]
  "tau": jnp.array([2.1738056e+05, 1.0622746e+02, 6.8072896e+00, 1.1605897e+00]), # Reservoir time scales [yr]
  "ch4_tau": 10,                                  # CH4 lifetime
  "k_ch4": 0.046,                                   # Radiative forcing from CH4

  # --- FaIR-style iIRF100 / alpha parameters for CO2 ---
  # Millar et al. (2017) / Smith et al. (2017) defaults
  "r0": 35.0,          # yr
  "rC": 0.019,         # yr / GtC
  "rT": 4.165,         # yr / K

  # Integrated impulse response horizon & max (FaIR convention)
  "iirf_h": 100.0,     # years over which iIRF is defined
  "iirf_max": 97.0,    # max allowed iIRF (avoids no-solution regime)

  "ch4_rT": 0.04,            # Sensitivity of lifetime to global-mean T [1/K]
  "ch4_ra": 1e-4,            # Sensitivity of lifetime to CH4 burden (ppb^-1)

  # N2O cycle
  "n2o_tau": 99.86245,          # years, FaIRv2 Table S2, default 100
  "n2o_emis2conc": 0.201,    # ppb per TgN/yr
  "k_n2o": 0.06395961,            # W/m2 per sqrt(ppb) difference, default 0.106

  # d_j: response timescales [years]
  "d": jnp.array([0.60589135, 16.034082, 277.6066]),

  # q_j: equilibrium response coefficients [K / (W m^-2)]
  "q": jnp.array([0.21279761, 0.2657743, 0.4667166]),

  "f2_SO2": -0.003, # aerosol-radiation interaction
  "f2_BC": 0.0315142, # aerosol-radiation interaction
  "f1_aci": -0.8848617, # aerosol-cloud interaction
  "C0_SO2": 24.358507, # SO2 shape parameter
  "f2_aci": -0.00948581, # aerosol-cloud interaction

  "C0_PI": 286.4 # Preindustrial CO2 [ppm]
}

# Constants
SPY        = 3600.0 * 24.0 * 365.25     # Seconds per year
EARTHAREA  = 5.1e14                     # Surface area of Earth m^2
LFRAC      = 0.292                      # Land fraction (unitless)
LTHK       = 8.4                        # m
OTHK       = 100.0                      # m
OMHC       = 4186.0                     # J kg^-1 K^-1
ODENS      = 1000.0                     # kg m^-3
DTHK       = jnp.array([300., 300.,
                        1300., 1800.])  # m
C_MIN      = 1e-6                       # ppm, to keep log well-defined
PPM_TO_GTC = 2.12                       # 1 ppm = 2.12 GtC

GTCO2_TO_PPM_PER_YEAR = (12.0 / 44.0) / PPM_TO_GTC  # Roughly 0.1276 ppm per GtCO2

M0_CH4_PI      = 720.0      # preindustrial CH4 [ppb]
PPB_TO_MTCH4   = 2.78       # 1 ppb CH4 ≈ 2.78 Mt CH4 in the atmosphere
MTCH4_PER_PPB  = PPB_TO_MTCH4
PPB_PER_MTCH4  = 1.0 / MTCH4_PER_PPB

M0_N2O_PI       = 270.0   # preindustrial N2O [ppb], FaIRv2 default
N2O_EMIS2CONC   = 0.201   # ppb per TgN/yr, FaIRv2 Table S2

idx_CO2     = 0
idx_CH4     = 1
idx_N2O     = 2
idx_Sulfur  = 3
idx_BC      = 4

# ==================================================================
# Part 2a: JAX SCM physics core (per-agent cycle steps + thermal response)
# ==================================================================

# -----------------------------------
# Carbon cycle step (Joos-like 4-box)
# -----------------------------------
def _iirf100_from_alpha(alpha: float,
                        a: jnp.ndarray,
                        tau: jnp.ndarray,
                        iirf_h: float) -> float:
  """
  iIRF100(alpha) = Σ_i alpha a_i τ_i [1 - exp(-H / (alpha τ_i))]
  with H = iirf_h (usually 100 years).
  """
  alpha = jnp.asarray(alpha, dtype=jnp.float32)
  a     = jnp.asarray(a,     dtype=jnp.float32)
  tau   = jnp.asarray(tau,   dtype=jnp.float32)
  H     = jnp.float32(iirf_h)

  term = alpha * a * tau * (1.0 - jnp.exp(-H / (alpha * tau)))
  return jnp.sum(term)

def compute_alpha_co2(Cacc_GtC: float,
                      T_global: float,
                      params: dict) -> float:
  """
  Solve for the CO2 adjustment factor alpha given cumulative uptake and temperature.

  Cacc_GtC : cumulative carbon taken up by land+ocean (GtC)
  T_global : global mean temperature anomaly (K)
  """
  a       = params["a"]
  tau     = params["tau"]
  r0      = jnp.float32(params["r0"])
  rC      = jnp.float32(params["rC"])
  rT      = jnp.float32(params["rT"])
  iirf_h  = jnp.float32(params.get("iirf_h", 100.0))
  iirf_max = jnp.float32(params.get("iirf_max", 97.0))

  # Target iIRF100 from Millar/Smith parameterisation:
  # iIRF100_target = r0 + rC * Cacc + rT * T
  iirf_target = r0 + rC * jnp.asarray(Cacc_GtC) + rT * jnp.asarray(T_global)
  # Physical bounds
  iirf_target = jnp.clip(iirf_target, 0.0, iirf_max)

  # Bisection for α in [0.1, 100] (safe, monotonic)
  alpha_lo0 = jnp.float32(0.1)
  alpha_hi0 = jnp.float32(100.0)

  def body_fun(_, state):
    alpha_lo, alpha_hi = state
    alpha_mid = 0.5 * (alpha_lo + alpha_hi)
    iirf_mid  = _iirf100_from_alpha(alpha_mid, a, tau, iirf_h)

    # If iIRF(α_mid) > target, α is too large → move high bound down
    cond = iirf_mid > iirf_target
    alpha_hi_new = jnp.where(cond, alpha_mid, alpha_hi)
    alpha_lo_new = jnp.where(cond, alpha_lo, alpha_mid)
    return (alpha_lo_new, alpha_hi_new)

  alpha_lo, alpha_hi = lax.fori_loop(0, 30, body_fun, (alpha_lo0, alpha_hi0))
  alpha = 0.5 * (alpha_lo + alpha_hi)
  return alpha

def carbon_step(cpool: jnp.ndarray,
                Et: float,
                params: dict,
                dt: float,
                alpha: float) -> jnp.ndarray:
  """
  cpool: shape (4,), reservoir carbon anomalies (ppm) contributing to Catm
  Et:    emissions input in ppm/year, held constant within substep (matches original)
  """
  a   = params["a"]
  tau = params["tau"]
  eff_tau = alpha * tau
  dydt = a * Et - cpool / eff_tau
  return cpool + dt * dydt

# ---------------------------
# CO2 radiative forcing (AR6)
# ---------------------------
def radiative_forcing_co2(C: jnp.ndarray, params: dict) -> jnp.ndarray:
  """
  Piecewise polynomial coefficient in front of log(C/C0).
  """
  a1 = -2.4785e-7   # W m^-2 ppm^-2
  b1 = 7.5906e-4    # W m^-2 ppm^-1
  d1 = 5.2488       # W m^-2
  C0 = params["C0_PI"]

  Camax = C0 - b1 / (2.0 * a1)

  C = jnp.clip(C, C_MIN, jnp.inf)   # safety for log
  coef_hi = d1 - (b1 ** 2) / (d1 * a1)                             # C >= Camax
  coef_mid = d1 + a1 * (C - C0) ** 2 + b1 * (C - C0)               # C0 < C < Camax
  coef_lo = d1                                                     # C <= C0

  coefC = jnp.where(C >= Camax, coef_hi,
            jnp.where(C > C0, coef_mid, coef_lo))
  return coefC * jnp.log(C / C0)

# ------------------
# Methane cycle step
# ------------------
def methane_step(m_anom_ppb: float,
                 E_mtCH4_per_year: float,
                 T_global: float,
                 params: dict,
                 dt: float) -> jnp.ndarray:
  """
  Single-reservoir CH4 anomaly (in ppb) relative to PI.
  d(m)/dt = E_ppb - m/tau
    where E_ppb = E_mtCH4 / 2.78  (MtCH4->ppb)
  """
  tau = params["ch4_tau"]
  rT   = params.get("ch4_rT", 0.0)
  ra   = params.get("ch4_ra", 0.0)

  x = rT * T_global + ra * m_anom_ppb
  x = jnp.clip(x, -20.0, 20.0) # clamp exponent for calibration
  alpha = jnp.exp(x)

  tau_eff = tau * alpha
  tau_eff = jnp.clip(tau_eff, 0.1, 200.0) # clamp tau for calibration

  E_ppb_per_year = E_mtCH4_per_year * PPB_PER_MTCH4
  dm_dt = E_ppb_per_year - (m_anom_ppb / tau_eff)
  return m_anom_ppb + dt * dm_dt

# ---------------------------
# CH4 radiative forcing (AR6)
# ---------------------------
def radiative_forcing_ch4(M_ppb: jnp.ndarray, params: dict) -> jnp.ndarray:
  """
  Simple CH4 ERF without N2O overlap:
    RF = k_ch4 * (sqrt(M) - sqrt(M0))
  """
  k = params["k_ch4"]
  M = jnp.clip(M_ppb, 1e-6, jnp.inf)
  return k * (jnp.sqrt(M) - jnp.sqrt(M0_CH4_PI))

def radiative_forcing_SO2_BC(E_SO2: jnp.ndarray, E_BC: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    E_*: [T] arrays of annual emissions (same dt grid as other agents)
    theta_aer: dict or pytree with FaIR-like aerosol params
    Returns: [T] aerosol ERF time series (W/m^2)
    """
    f2_SO2 = params["f2_SO2"]
    f2_BC  = params["f2_BC"]
    f1_aci = params["f1_aci"]
    C0_SO2 = params["C0_SO2"]
    f2_aci = params["f2_aci"]

    E_SO2_eff = jnp.maximum(E_SO2, 0.0)

    ERFari = f2_SO2 * E_SO2_eff + f2_BC * E_BC
    ERFaci = f1_aci * jnp.log1p(E_SO2_eff / C0_SO2) + f2_aci * (E_BC)

    return ERFari + ERFaci

def n2o_step(n_anom_ppb: jnp.ndarray,
             E_TgN_per_year: jnp.ndarray,
             params: dict,
             dt: float) -> jnp.ndarray:
  """
  Single-reservoir N2O anomaly (in ppb) relative to PI (270 ppb).

  d(n)/dt = emis2conc * E_TgN - n / tau

  where E_TgN is N2O emissions in TgN/yr (FaIR v2.0 default units).
  """
  tau = params["n2o_tau"]                  # ~100 years
  emis2conc = params["n2o_emis2conc"]      # ~0.201 ppb / (TgN/yr)

  dn_dt = emis2conc * E_TgN_per_year - (n_anom_ppb / tau)
  return n_anom_ppb + dt * dn_dt

def radiative_forcing_n2o(N_ppb: jnp.ndarray, params: dict) -> jnp.ndarray:
  """
  Simple N2O ERF (FaIRv2-like, no interaction terms):
    RF = k_n2o * (sqrt(N) - sqrt(N0))
  """
  k = params["k_n2o"]              # ~0.106 W/m2 / sqrt(ppb)
  N = jnp.clip(N_ppb, 1e-6, jnp.inf)
  return k * (jnp.sqrt(N) - jnp.sqrt(M0_N2O_PI))

# ------------------
# Full climate step
# -----------------
def climate_step(S: jnp.ndarray,
                 RF_prev: float,
                 params: dict,
                 dt: float) -> tuple[jnp.ndarray, float]:
  """
  Three-timescale impulse-response thermal core (FaIRv2.0.0):
    dS_j/dt = q_j * F(t) - S_j / d_j
    T = sum_j S_j

  S        : shape (3,), thermal box responses S_j (K)
  RF_prev  : forcing F(t) in W m^-2 from previous step
  dt       : timestep in years
  returns  : (S_new, T_new) with T_new = sum(S_new)
  """
  d = params["d"]  # (3,) timescales [years]
  q = params["q"]  # (3,) response coeffs [K / (W m^-2)]

  # ODE step using previous-step forcing F(t)
  dSdt = (q * RF_prev - S) / d
  S_new = S + dt * dSdt

  T_new = jnp.sum(S_new)
  return S_new, T_new

# ----------------
# Time integration
# ----------------
def simulate_temp(
  years: jnp.ndarray,
  emissions_by_agent: jnp.ndarray,   # (N_agents, N_t)
  mode: str = 'FaIR',
  params: dict | None = None,
  dt: float = 0.1
) -> dict:
  """
  Forward-integrate the JAX SCM (carbon/CH4/N2O cycles + thermal response) from
  annual emissions to a full state-and-diagnostics dict, sampled back to `years`.

  emissions_by_agent rows follow (CO2, CH4, N2O, Sulfur, BC) order (see idx_* constants).
  Returns a dict with keys "years", "Catm_ppm", "Matm_ppb", "Natm_ppb", "RF_*", "GMST",
  plus cumulative-emissions diagnostics and sub-annual ("_sub") series.
  """
  if params is None:
    if mode == 'FaIR':
      params = FAIR_PARAMS
    elif mode == 'MESM':
      params = MESM_PARAMS
    else:
      raise ValueError(f'Error, mode {mode} not recognized.')

  years = jnp.asarray(years, dtype=jnp.float32)
  start = years[0]
  T = years.shape[0]
  nsteps = int(np.floor((T - 1) / dt) + 1)
  tvec = start + jnp.arange(nsteps + 1, dtype=jnp.float32) * dt

  emissions_by_agent = jnp.asarray(emissions_by_agent, dtype=jnp.float32)

  if emissions_by_agent.ndim != 2:
    raise ValueError("emissions_by_agent must have shape (N_agents, T)")

  n_agents, T_em = emissions_by_agent.shape
  if T_em != years.shape[0]:
    raise ValueError("Time dimension of emissions must match `years` length")

  # Initial states
  cpool0  = jnp.zeros((4,), dtype=jnp.float32) # carbon pool (ppm)
  m_anom0  = jnp.float32(0.0)   # CH4 anomaly (ppb)
  n_anom0  = jnp.float32(0.0)   # N2O anomaly (ppb)
  S0 = jnp.zeros((3,), dtype=jnp.float32)
  RF_prev0 = jnp.float32(0.0)
  cumulative_GtC0 = jnp.float32(0.0)

  cumulative_GtC0 = jnp.float32(0.0)
  carry0 = (cpool0, m_anom0, n_anom0, S0, RF_prev0, cumulative_GtC0)

  def step_fn(carry, t_curr):
    cpool, m_anom, n_anom, S, RF_prev, cumulative_GtC = carry

    T_prev = jnp.sum(S)

    # --- emissions lookup (hold-last) ---
    idx = jnp.searchsorted(years, t_curr + (dt/10.0), side='right') - 1
    idx = jnp.clip(idx, 0, years.shape[0]-1)

    Et_all                      = emissions_by_agent[:, idx]
    Et_co2_GtCO2_per_year       = Et_all[idx_CO2]
    Et_ch4_MtCH4_per_year       = Et_all[idx_CH4]
    Et_n2o_TgN_per_year         = Et_all[idx_N2O]
    Et_sulfur_MtSulfur_per_year = Et_all[idx_Sulfur]
    Et_bc_MtBC_per_year         = Et_all[idx_BC]

    # Conversions
    Et_co2_ppm_per_year = Et_co2_GtCO2_per_year * GTCO2_TO_PPM_PER_YEAR
    Et_co2_GtC_per_year = Et_co2_GtCO2_per_year * (12.0 / 44.0)

    C0_PI = params["C0_PI"]

    # --- CO2 carbon pools -> Catm (ppm) ---
    cumulative_GtC_new = cumulative_GtC + Et_co2_GtC_per_year * dt
    Catm_ppm_prev = C0_PI + jnp.sum(cpool)
    Cacc_GtC = cumulative_GtC_new - (Catm_ppm_prev - C0_PI) * PPM_TO_GTC
    alpha_co2 = compute_alpha_co2(Cacc_GtC, T_prev, params)
    cpool_new = carbon_step(cpool, Et_co2_ppm_per_year, params, dt, alpha_co2)
    Catm_ppm  = C0_PI + jnp.sum(cpool_new)

    # --- CH4 1-box -> Matm (ppb) ---
    m_anom_new = methane_step(m_anom, Et_ch4_MtCH4_per_year, T_prev, params, dt)
    Matm_ppb   = M0_CH4_PI + m_anom_new

    # --- N2O 1-box -> Natm (ppb) ---
    n_anom_new = n2o_step(n_anom, Et_n2o_TgN_per_year, params, dt)
    Natm_ppb   = M0_N2O_PI + n_anom_new

    # --- ERFs ---
    RF_co2  = radiative_forcing_co2(Catm_ppm, params)
    RF_ch4  = radiative_forcing_ch4(Matm_ppb, params)
    RF_n2o  = radiative_forcing_n2o(Natm_ppb, params)
    RF_aer  = radiative_forcing_SO2_BC(Et_sulfur_MtSulfur_per_year, Et_bc_MtBC_per_year, params)
    RF_curr = RF_co2 + RF_ch4 + RF_n2o + RF_aer

    # --- Thermal response: three-timescale impulse response ---
    S_new, T_new = climate_step(S, RF_prev, params, dt)

    # RF_curr becomes RF_prev for the *next* step
    new_carry = (cpool_new, m_anom_new, n_anom_new, S_new, RF_curr, cumulative_GtC_new)

    y = jnp.stack([Catm_ppm, Matm_ppb, Natm_ppb, RF_co2, RF_ch4, RF_n2o, RF_aer, T_new], axis=0)

    return new_carry, y

  carry0 = (cpool0, m_anom0, n_anom0, S0, RF_prev0, cumulative_GtC0)
  carry_f, Y = lax.scan(step_fn, carry0, tvec[:-1])  # (nsteps, 6)

  Catm_full = Y[:, 0]   # ppm
  Matm_full = Y[:, 1]   # ppb
  Natm_full = Y[:, 2]   # ppb
  RFc_full  = Y[:, 3]
  RFm_full  = Y[:, 4]
  RFn_full  = Y[:, 5]
  RFa_full  = Y[:, 6]
  GMST_full = Y[:, 7]

  # Sample to year ticks (hold-last)
  year_indices = jnp.searchsorted(tvec[:-1], years, side='right') - 1
  year_indices = jnp.clip(year_indices, 0, GMST_full.shape[0]-1)

  Catm_years = Catm_full[year_indices]
  Matm_years = Matm_full[year_indices]
  Natm_years = Natm_full[year_indices]
  RFc_years  = RFc_full[year_indices]
  RFm_years  = RFm_full[year_indices]
  RFn_years  = RFn_full[year_indices]
  RFa_years  = RFa_full[year_indices]
  GMST_years = GMST_full[year_indices]

  # Diagnostics
  emissions_ppm_co2 = emissions_by_agent[idx_CO2] * GTCO2_TO_PPM_PER_YEAR
  cumulative_GtC = jnp.cumsum(emissions_ppm_co2 * PPM_TO_GTC)     # GtC
  cumulative_MtCH4 = jnp.cumsum(emissions_by_agent[idx_CH4])                              # MtCH4

  return {
    "years": years,

    # Atmosphere
    "Catm_ppm": Catm_years,
    "Matm_ppb": Matm_years,
    "Natm_ppb": Natm_years,

    # Forcing
    "RF_co2": RFc_years,
    "RF_ch4": RFm_years,
    "RF_n2o": RFn_years,
    "RF_aer": RFa_years,
    "RF_total": RFc_years + RFm_years + RFa_years,

    # Temperature
    "GMST": GMST_years,

    # Emissions diagnostics
    "emissions_by_agent": emissions_by_agent,
    "cumulative_GtC": cumulative_GtC,
    "cumulative_MtCH4": cumulative_MtCH4,

    # Optional sub-annual outputs
    "t_sub": tvec[1:],
    "Catm_sub": Catm_full,
    "Matm_sub": Matm_full,
    "RF_co2_sub": RFc_full,
    "RF_ch4_sub": RFm_full,
    "GMST_sub": GMST_full,
  }

def simulate_temp_prescribed_conc(
    years: jnp.ndarray,
    concentrations_by_agent: jnp.ndarray, # (3, T) [CO2, CH4, N2O]
    emissions_by_agent: jnp.ndarray,      # (5, T) [Needed for Aerosols]
    params: dict | None = None,
    dt: float = 0.1
) -> jnp.ndarray:
    """
    Forward model: Concentration (+ Aer Emissions) -> Temperature.
    Bypasses gas cycles. Used for calibrating Climate Sensitivity (d, q).
    Returns GMST (K anomaly), one value per entry in `years`.
    """
    if params is None:
      params = FAIR_PARAMS

    years = jnp.asarray(years, dtype=jnp.float32)
    start = years[0]
    nsteps = int(np.floor((years.shape[0] - 1) / dt) + 1)
    tvec = start + jnp.arange(nsteps + 1, dtype=jnp.float32) * dt

    # Initial Thermal State
    S0 = jnp.zeros((3,), dtype=jnp.float32)
    RF_prev0 = jnp.float32(0.0)
    carry0 = (S0, RF_prev0)

    def step_fn(carry, t_curr):
        S, RF_prev = carry

        # Lookup index
        idx = jnp.searchsorted(years, t_curr + (dt/10.0), side='right') - 1
        idx = jnp.clip(idx, 0, years.shape[0]-1)

        # 1. Get Prescribed Concentrations
        # concentrations_by_agent expected order: 0:CO2(ppm), 1:CH4(ppb), 2:N2O(ppb)
        C_ppm = concentrations_by_agent[0, idx]
        M_ppb = concentrations_by_agent[1, idx]
        N_ppb = concentrations_by_agent[2, idx]

        # 2. Get Aerosol Emissions (Sulfur=3, BC=4 in standard layout)
        E_sulf = emissions_by_agent[idx_Sulfur, idx]
        E_bc   = emissions_by_agent[idx_BC, idx]

        # 3. Calculate Forcing
        RF_co2 = radiative_forcing_co2(C_ppm, params)
        RF_ch4 = radiative_forcing_ch4(M_ppb, params)
        RF_n2o = radiative_forcing_n2o(N_ppb, params)
        RF_aer = radiative_forcing_SO2_BC(E_sulf, E_bc, params)
        RF_curr = RF_co2 + RF_ch4 + RF_n2o + RF_aer

        # 4. Thermal Step
        S_new, T_new = climate_step(S, RF_prev, params, dt)

        return (S_new, RF_curr), T_new

    _, GMST_full = lax.scan(step_fn, carry0, tvec[:-1])

    # Downsample to annual
    year_indices = jnp.searchsorted(tvec[:-1], years, side='right') - 1
    year_indices = jnp.clip(year_indices, 0, GMST_full.shape[0]-1)

    return GMST_full[year_indices]

# ==================================================================
# Part 1 / 2a: FaIR <-> JAX emissions data plumbing, and evaluation metrics
# ==================================================================

# ----------------------
# Misc. helper functions
# ----------------------
# (plot_FaIR_v_JAX moved to utils_plotting.py; see needs_historical below, which it uses)

tier1 = ['H-ext','M','ML','L','VLLO-ext','VLHO']

needs_historical = ['H-ext','H-ext-OS','M',
                    'M-ext','ML','ML-ext','L',
                    'L-ext','VLLO-ext','VLHO','VLHO-ext']

def get_emissions(emis_dict: dict, agents: Iterable[str]) -> dict:
  """Convert emis_dict[scenario][agent] (numpy) into the same nested structure as jnp arrays."""
  emis_dict_JAX = {}
  for scen in emis_dict:
    if scen == 'historical' and 'historical' in emis_dict_JAX:
      continue

    emis_dict_JAX[scen] = {}

    for a in agents:
      #if scen in needs_historical:
      #  emis = jnp.concatenate((emis_dict['historical'][a], emis_dict[scen][a]))
      #else:
      emis = jnp.array(emis_dict[scen][a])

      emis_dict_JAX[scen][a] = emis

  return emis_dict_JAX

def build_emissions_by_agent(emis_dict_JAX: dict, agent_order: tuple[str, ...] = ('CO2', 'CH4', 'N2O', 'Sulfur', 'BC')) -> dict:
  """
  Convert emis_dict_JAX[scenario][agent] -> emissions_by_agent[scenario].

  For each scenario:
    - Create an array of shape (N_agents, T) where rows follow `agent_order`.
    - If an agent in `agent_order` is missing for that scenario, its row is zeros.
  """
  emis_by_agent = {}

  for scen, scen_emis in emis_dict_JAX.items():
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

def run_scenarios(emis_by_agent_dict: dict, mode: str = 'FaIR', params: dict | None = None, dt: float = 0.1) -> dict:
    """
    Batched execution of scenarios using simulate_temp logic.

    Replicates the exact I/O of 'run_scenarios' but uses jax.vmap to run
    all scenarios in parallel on the GPU/TPU.

    Inputs:
        emis_by_agent_dict: { 'ScenarioName': array(N_agents, Time) }

    Returns:
        outputs: { 'ScenarioName': simulate_temp_output }
                 (Output trimmed to match input length)
    """

    # 1. Setup & Dimensions
    scen_names = list(emis_by_agent_dict.keys())
    n_scens = len(scen_names)

    # Get N_agents from the first scenario
    n_agents = next(iter(emis_by_agent_dict.values())).shape[0]

    # Determine lengths for padding/slicing
    lengths = [emis_by_agent_dict[name].shape[1] for name in scen_names]
    max_len = max(lengths)

    # 2. Construct Padded Batch Inputs
    # Emissions: (N_Scenarios, N_Agents, Max_Time)
    padded_emis = jnp.zeros((n_scens, n_agents, max_len))

    # Years: (N_Scenarios, Max_Time)
    # We tile the years array so vmap sees a consistent shape
    years_template = jnp.arange(max_len, dtype=jnp.float32)
    padded_years = jnp.tile(years_template, (n_scens, 1))

    for i, name in enumerate(scen_names):
        curr_len = lengths[i]
        curr_data = emis_by_agent_dict[name]

        # Fill the emission data into the padded tensor
        padded_emis = padded_emis.at[i, :, :curr_len].set(curr_data)

        # (Note: The 'years' array is already filled 0..Max.
        # The extra years beyond curr_len will be simulated but discarded.)

    # 3. Parallel Execution via VMAP
    # Map over axis 0 of 'years' and 'emissions'.
    # 'params' and 'dt' are broadcasted (None).
    vmapped_sim = jax.vmap(simulate_temp, in_axes=(0, 0, None, None, None))

    # Execute batch
    # Output structure will mirror simulate_temp's return, but with an added leading dimension
    batch_results = vmapped_sim(padded_years, padded_emis, mode, params, dt)

    # 4. Unpack and Slice Results
    outputs = {}

    # Helper to correctly slice the time dimension if it exists
    def slice_leaf(batch_arr, index, valid_len):
        # If array matches max_len in dim 1, it's a time series -> Slice it.
        # If not (e.g. scalar summary), just take the batch item.
        if batch_arr.ndim > 1 and batch_arr.shape[1] == max_len:
            return batch_arr[index, :valid_len]
        return batch_arr[index]

    # We use jax.tree_util to handle whatever structure (dict, tuple, etc.) simulate_temp returns
    for i, name in enumerate(scen_names):
        valid_len = lengths[i]

        # Extract the i-th slice for this scenario from the batched PyTree
        scen_result = jax.tree_util.tree_map(
            lambda x: slice_leaf(x, i, valid_len),
            batch_results
        )

        outputs[name] = scen_result

    return outputs


def calc_nrmse(T_model: jnp.ndarray, T_true: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
  """RMSE(T_model, T_true) normalized by max(|T_true|), eps floor - same convention
  as utils_inverse._nrmse (which delegates here), so calibration loss and emulator-
  evaluation loss use one shared NRMSE definition."""
  diff = T_model - T_true
  rmse = jnp.sqrt(jnp.mean(diff**2))
  max_abs = jnp.maximum(jnp.max(jnp.abs(T_true)), eps)
  return rmse / max_abs

def mean_nrmse_from_outputs(
    delT_dict_JAX: dict,
    delT_dict_FaIR: dict,
) -> tuple[list, jnp.ndarray]:
  """
  Using precomputed model outputs:
    - compute per-scenario NRMSE
    - return their mean
  """
  losses = []
  for scen in delT_dict_JAX:
    T_model = jnp.asarray(delT_dict_JAX[scen]["GMST"], dtype=jnp.float32)
    T_true  = jnp.asarray(delT_dict_FaIR[scen], dtype=jnp.float32)
    loss_s  = calc_nrmse(T_model, T_true)
    losses.append(loss_s)

  return losses, jnp.mean(jnp.stack(losses, axis=0))

# -----------------
# Model calibration
# -----------------
def generate_calib_data(agents: Iterable[str], mode: str = 'FaIR') -> tuple[dict, dict, dict, dict]:
  """
  Build a FaIR-vs-JAX calibration dataset for `agents`: ScenarioMIP tier1+tier2
  plus the DECK scenarios relevant to those agents.

  Returns (emis_dict_calib_FaIR, emis_dict_calib_JAX, delT_dict_calib_FaIR, delT_dict_calib_JAX).
  """

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

  emis_dict_JAX = get_emissions(emis_dict_calib_FaIR, agents)
  emis_dict_calib_JAX = build_emissions_by_agent(emis_dict_JAX)
  delT_dict_calib_JAX = run_scenarios(emis_dict_calib_JAX, mode=mode)

  return emis_dict_calib_FaIR, emis_dict_calib_JAX, delT_dict_calib_FaIR, delT_dict_calib_JAX

def generate_JAX_data(
    agents: Iterable[str], DECK: bool = True, CS3: bool = False, DAMIP: bool = False, GeoMIP: bool = False
) -> list[dict]:
  """
  Build the list of JAX-format emissions dicts (one per scenario group requested)
  used to drive the SCM: ScenarioMIP tier1/tier2 always, plus DECK/CS3/DAMIP/GeoMIP
  groups when their flag is set.
  """

  emis_dict_tier1_FaIR, emis_dict_tier2_FaIR = run_fair.load_scenarioMIP_CMIP7(agents)

  emis_dict_JAX = get_emissions(emis_dict_tier1_FaIR, agents)
  emis_dict_tier1_JAX = build_emissions_by_agent(emis_dict_JAX)

  emis_dict_JAX = get_emissions(emis_dict_tier2_FaIR, agents)
  emis_dict_tier2_JAX = build_emissions_by_agent(emis_dict_JAX)

  emis_dicts = [emis_dict_tier1_JAX, emis_dict_tier2_JAX]

  if DECK:
    emis_dict_DECK_FaIR = run_fair.load_DECK_CMIP7(agents)

    emis_dict_DECK_subset = {}
    for scen in emis_dict_DECK_FaIR:
      if any(a in scen for a in agents):
        emis_dict_DECK_subset[scen] = {}
        for a in agents:
          emis_dict_DECK_subset[scen][a] = emis_dict_DECK_FaIR[scen][a].copy()
      else:
        continue

    emis_dict_JAX = get_emissions(emis_dict_DECK_subset, agents)
    emis_dict_DECK_JAX = build_emissions_by_agent(emis_dict_JAX)
    emis_dicts.append(emis_dict_DECK_JAX)

  if CS3:
    emis_dict_JAX = run_fair.load_CS3(agents=agents, emis_dict_tier1=emis_dict_tier1_FaIR)

    for scen in emis_dict_JAX:
        for a in emis_dict_JAX[scen]:
            if a not in agents:
                emis_dict_JAX[scen][a] = emis_dict_JAX[scen][a] * 0

    emis_dict_CS3_JAX = build_emissions_by_agent(emis_dict_JAX)
    emis_dicts.append(emis_dict_CS3_JAX)

  if DAMIP:
    M_GHG = jnp.concat([emis_dict_tier1_JAX['historical'], emis_dict_tier1_JAX['M']], axis=1)
    M_GHG = M_GHG.at[3:, :].set(M_GHG[3:, :] * 0)
    M_AER = jnp.concat([emis_dict_tier1_JAX['historical'], emis_dict_tier1_JAX['M']], axis=1)
    M_AER = M_AER.at[0:3, :].set(M_AER[0:3, :] * 0)
    emis_dict_DAMIP = {'M_GHG':M_GHG, 'M_AER':M_AER}
    emis_dicts.append(emis_dict_DAMIP)

  if GeoMIP:
    path_GEO = str(DATA_DIR / 'GeoMIP' / 'emis_G6sulfur.pickle')
    with open(path_GEO, "rb") as f:
        emis_GEO = pickle.load(f)
    emis_dict_GEO = {'G6sulfur': emis_GEO}
    emis_dicts.append(emis_dict_GEO)

  return emis_dicts

# ==================================================================
# Part 2a: calibration (theta <-> params mapping, loss, optimizers)
# ==================================================================

# ----------------------------------
# Calibrate model parameters to FaIR
# ----------------------------------
def make_theta0(mode: str = 'FaIR') -> jnp.ndarray:
  """
  Build an initial parameter vector theta from a params dict.

  theta layout (length 11):
    [0]  r0
    [1]  rC
    [2]  rT
    [3]  ch4_rT
    [4]  ch4_ra
    [5:8]  log(d_j)
    [8:11] log(q_j)
  """

  if mode == 'FaIR':
    params = FAIR_PARAMS
  elif mode == 'MESM':
    params = MESM_PARAMS
  else:
    raise ValueError(f'Error, mode {mode} not recognized.')

  r0   = jnp.float32(params["r0"])
  rC   = jnp.float32(params["rC"])
  rT   = jnp.float32(params["rT"])
  ch4_rT = jnp.float32(params["ch4_rT"])
  ch4_ra = jnp.float32(params["ch4_ra"])

  n2o_tau = jnp.float32(params["n2o_tau"])
  k_n2o = jnp.float32(params["k_n2o"])

  d0 = jnp.asarray(params["d"], dtype=jnp.float32)  # shape (3,)
  q0 = jnp.asarray(params["q"], dtype=jnp.float32)  # shape (3,)
  a0 = jnp.asarray(params["a"],   dtype=jnp.float32)  # (4,)
  tau0 = jnp.asarray(params["tau"], dtype=jnp.float32)  # (4,)

  log_d0 = jnp.log(d0)
  log_q0 = jnp.log(q0)
  log_tau0 = jnp.log(tau0)

  a_logits0 = jnp.log(a0)

  theta0 = jnp.concatenate(
    [
      jnp.array([r0, rC, rT, ch4_rT, ch4_ra], dtype=jnp.float32),
      log_d0,
      log_q0,
      a_logits0,
      log_tau0,
      jnp.array([
        params["f2_SO2"],
        params["f2_BC"],
        params["f1_aci"],
        params["C0_SO2"],
        params["f2_aci"],
      ], dtype=jnp.float32),
      jnp.array([n2o_tau, k_n2o], dtype=jnp.float32),
    ],
    axis=0,
  )
  return theta0

def params_from_theta(theta: jnp.ndarray,
                      base_params: dict = FAIR_PARAMS) -> dict:
  """
  Map the unconstrained theta vector back into a params dict that can be
  passed into simulate_temp.

  We copy FAIR_PARAMS and overwrite the tuned fields.
  """
  # Unpack
  r0      = theta[0]
  rC      = theta[1]
  rT      = theta[2]
  ch4_rT  = theta[3]
  ch4_ra  = theta[4]
  log_d   = theta[5:8]   # (3,)
  log_q   = theta[8:11]  # (3,)

  # Carbon cycle
  a_logits = theta[11:15]  # (4,)
  log_tau  = theta[15:19]  # (4,)

  d = jnp.exp(log_d)
  q = jnp.exp(log_q)
  tau = jnp.exp(log_tau)
  a   = jax.nn.softmax(a_logits)

  f2_SO2  = theta[19]
  f2_BC   = theta[20]
  f1_aci  = theta[21]
  C0_SO2  = theta[22]
  f2_aci  = theta[23]

  n2o_tau = theta[24]
  k_n2o = theta[25]

  # Shallow copy of base params, then overwrite tuned entries
  params = dict(base_params)
  params["r0"]      = r0
  params["rC"]      = rC
  params["rT"]      = rT
  params["ch4_rT"]  = ch4_rT
  params["ch4_ra"]  = ch4_ra
  params["d"]       = d
  params["q"]       = q
  params["a"]       = a
  params["tau"]     = tau
  params["f2_SO2"]  = f2_SO2
  params["f2_BC"]   = f2_BC
  params["f1_aci"]  = f1_aci
  params["C0_SO2"]  = C0_SO2
  params["f2_aci"]  = f2_aci
  params["n2o_tau"] = n2o_tau
  params["k_n2o"]   = k_n2o

  # ecs is implied by q; keep ecs consistent if you want
  # F_2xCO2 from your AR6 RF at 2xCO2:
  C0_PI = params["C0_PI"]
  F2x = radiative_forcing_co2(2.0 * C0_PI, params)
  params["ecs"] = F2x * jnp.sum(q)

  return params

def make_theta_mask(theta: jnp.ndarray,
                    target: str = "all") -> jnp.ndarray:
  """
  Build a mask for theta so we can zero-out gradients for everything
  except the chosen target group.

  target ∈ {"all", "Climate", "Carbon", "CO2", "CH4", "Aer"}.
  """
  mask = jnp.zeros_like(theta)

  if target == "all":
    mask = jnp.ones_like(theta)

  # 1. Climate / Thermal Only (d, q) -> Calibrates ECS/TCR
  if target == "Climate":
    # log_d (5:8) and log_q (8:11)
    clim_idx = jnp.array(list(range(5, 11)), dtype=jnp.int32)
    mask = mask.at[clim_idx].set(1.0)

  # 2. Carbon Cycle Only (r parameters, a, tau) -> Calibrates CO2 airborne fraction
  elif target == "Carbon":
    # r0, rC, rT (0:3)
    mask = mask.at[0:3].set(1.0)
    # a_logits (11:15), log_tau (15:19)
    mask = mask.at[11:19].set(1.0)

  # 3. Gas specific
  elif target == "CO2":
    co2_idx = jnp.array(
      [0, 1, 2]              # r0, rC, rT
      + list(range(11, 15))  # a_logits
      + list(range(15, 19)), # log_tau
      dtype=jnp.int32,
    )
    mask = mask.at[co2_idx].set(1.0)
  elif target == "CH4":
    ch4_idx = jnp.array([3, 4], dtype=jnp.int32)
    mask = mask.at[ch4_idx].set(1.0)
  elif target == "N2O":
    n2o_idx = jnp.array([24, 25], dtype=jnp.int32)
    mask = mask.at[n2o_idx].set(1.0)
  elif target == "Aer":
    aer_idx = jnp.array(list(range(19, 24)), dtype=jnp.int32)  # sulfur + BC
    mask = mask.at[aer_idx].set(1.0)
  else:
    raise ValueError(f"Unknown calibration target: {target}")

  return mask

def loss_fn(theta: jnp.ndarray,
            emis_dict_JAX: dict,
            delT_dict_FaIR: dict,
            dt: float = 0.1) -> jnp.ndarray:
  """
  Compute mean NRMSE between this SCM (with params derived from theta)
  and the FaIR ground-truth delT time series, across all calibration scenarios.

  This is the scalar objective for JAX-based optimization.
  """
  # 1) Map theta -> params dict
  params = params_from_theta(theta, base_params=FAIR_PARAMS)

  # 2) Run SCM on all calibration scenarios
  delT_dict_JAX = run_scenarios(emis_dict_JAX, params=params, dt=dt)

  # 3) Compute per-scenario NRMSE and their mean
  _, loss_avg = mean_nrmse_from_outputs(
    delT_dict_JAX,
    delT_dict_FaIR,
  )
  return loss_avg

# ----------------------------------------
# Mode A: Calibrate Carbon Cycle (Emis -> Conc)
# ----------------------------------------
def calibrate_carbon_cycle(
    filepath: str,
    emis_dict_JAX: dict,
    target_conc_dict: dict,
    theta0: jnp.ndarray,
    dt: float,
    n_steps: int,
    learning_rate: float = 1e-2,
    mode: str = 'FaIR',
    base_params: dict = FAIR_PARAMS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calibrates 'r0', 'rC', 'rT', 'a', 'tau' to match target CO2 concentrations.
    Ignores Temperature output/errors. Checkpoints theta to `filepath` every step.
    Non-tuned theta fields (everything outside the "Carbon" mask) are pulled from
    `base_params` - typically FAIR_PARAMS; pass MESM_PARAMS explicitly when this
    is one step of an MESM calibration (see scripts/2c_calibrate_MESM.py).
    Returns (theta_opt, final_loss).
    """
    # Pre-pad data for JAX (similar to run_scenarios)
    scen_names = list(emis_dict_JAX.keys())

    def loss_carbon(theta):
        params = params_from_theta(theta, base_params)
        # Run standard forward model
        res_dict = run_scenarios(emis_dict_JAX, params=params, dt=dt, mode=mode)

        total_mse = 0.0
        # Calculate MSE only on CO2 concentrations
        for scen in scen_names:
            pred_co2 = res_dict[scen]["Catm_ppm"]
            true_co2 = target_conc_dict[scen][0,:] # Expecting array of shape (T,)
            total_mse += jnp.mean((pred_co2 - true_co2)**2)
        return total_mse

    loss_and_grad = jax.value_and_grad(loss_carbon)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(theta0)
    theta = theta0

    # Mask: Update ONLY Carbon params
    mask = make_theta_mask(theta0, target="Carbon")

    print("Starting Carbon Cycle Calibration (Emis -> Conc)...")
    for step in range(n_steps):
        loss, grads = loss_and_grad(theta)
        grads = grads * mask # Apply mask
        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

        if step % 100 == 0 or step == n_steps - 1:
            print(f"Step {step} | Loss (Conc MSE): {loss:.4f}")

    with open(filepath, 'wb') as f: pickle.dump(theta, f)
    return theta, params_from_theta(theta, base_params)

# ----------------------------------------
# Mode B: Calibrate Climate (Conc -> Temp)
# ----------------------------------------
def calibrate_climate_sensitivity(
    filepath: str,
    emis_dict_JAX: dict,
    conc_dict_JAX: dict,
    target_temp_dict: dict,
    theta0: jnp.ndarray,
    dt: float,
    n_steps: int,
    learning_rate: float = 1e-2,
    base_params: dict = FAIR_PARAMS,
) -> tuple[jnp.ndarray, dict]:
    """
    Calibrates the thermal response params (d, q) to match target GMST given
    prescribed concentrations (via simulate_temp_prescribed_conc), bypassing the
    carbon cycle entirely. Non-tuned theta fields (everything outside the
    "Climate" mask) are pulled from `base_params` - typically FAIR_PARAMS; pass
    MESM_PARAMS explicitly when this is one step of an MESM calibration (see
    scripts/2c_calibrate_MESM.py). Checkpoints theta to `filepath`.
    Returns (theta_opt, params_opt).
    """
    scen_names = list(emis_dict_JAX.keys())

    # Helper to vmap the new prescribed function
    def run_prescribed_batch(theta):
        params = params_from_theta(theta, base_params)
        losses = []
        for scen in scen_names:
            # We need both Emis (for Aerosols) and Conc (for GHGs)
            emis = emis_dict_JAX[scen] # Shape (2, T)
            conc = conc_dict_JAX[scen] # Shape (3, T)
            target = target_temp_dict[scen]
            years = jnp.arange(emis.shape[1]) # Assuming year 0..T

            pred_T = simulate_temp_prescribed_conc(years, conc, emis, params, dt)
            losses.append(jnp.mean((pred_T - target)**2))
        return jnp.sum(jnp.array(losses))

    loss_and_grad = jax.value_and_grad(run_prescribed_batch)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(theta0)
    theta = theta0

    # Mask: Update ONLY Climate params (d, q)
    mask = make_theta_mask(theta0, target="Climate")

    print("Starting Climate Sensitivity Calibration (Conc -> Temp)...")
    for step in range(n_steps):
        loss, grads = loss_and_grad(theta)
        grads = grads * mask
        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

        if step % 100 == 0 or step == n_steps - 1:
            print(f"Step {step} | Loss (Temp MSE): {loss:.4f}")

    with open(filepath, 'wb') as f: pickle.dump(theta, f)
    return theta, params_from_theta(theta, base_params)

# ----------------
# Train/test split
# ----------------

def generate_train_test(agents: Iterable[str], mode: str = 'FaIR') -> tuple[dict, dict, dict, dict, dict, dict, dict, dict]:
  """
  Build a ScenarioMIP tier1 (train) / tier2 (test) split of FaIR + JAX emissions and delT.

  Returns (emis_dict_train_FaIR, emis_dict_test_FaIR, emis_dict_train_JAX, emis_dict_test_JAX,
  delT_dict_train_FaIR, delT_dict_test_FaIR, delT_dict_train_JAX, delT_dict_test_JAX).
  """

  emis_dict_train_FaIR, emis_dict_test_FaIR = run_fair.load_scenarioMIP_CMIP7(agents)
  delT_dict_train_FaIR = run_fair.get_delT(emis_dict_train_FaIR, list(emis_dict_train_FaIR.keys()), agents, MIP='ScenarioMIP_tier1')
  delT_dict_test_FaIR = run_fair.get_delT(emis_dict_test_FaIR, list(emis_dict_test_FaIR.keys()), agents, MIP='ScenarioMIP_tier2')

  emis_dict_JAX = get_emissions(emis_dict_train_FaIR, agents)
  emis_dict_train_JAX = build_emissions_by_agent(emis_dict_JAX)
  delT_dict_train_JAX = run_scenarios(emis_dict_train_JAX, mode=mode)

  emis_dict_JAX = get_emissions(emis_dict_test_FaIR, agents)
  emis_dict_test_JAX = build_emissions_by_agent(emis_dict_JAX)
  delT_dict_test_JAX = run_scenarios(emis_dict_test_JAX, mode=mode)

  return emis_dict_train_FaIR, emis_dict_test_FaIR, emis_dict_train_JAX, emis_dict_test_JAX, delT_dict_train_FaIR, delT_dict_test_FaIR, delT_dict_train_JAX, delT_dict_test_JAX

def calibrate_inverse(
    filepath: str,
    emis_dict_JAX: dict,
    delT_dict_FaIR: dict,
    theta0: jnp.ndarray,
    dt: float,
    n_steps: int,
    target: str = "CO2",
    learning_rate: float = 1e-2,
) -> tuple[dict, jnp.ndarray]:
  """
  Calibrate the `target` group of theta (see make_theta_mask) against FaIR delT via
  Adam, checkpointing theta to `filepath` every 10 steps. Returns (params_opt, final_loss).

  Note: theta's entries span very different natural scales (e.g. ch4_rT ~ 4e-2,
  ch4_ra ~ 1e-4 vs. r0 ~ 35), so a single learning_rate can be stable for one
  `target` group and diverge for another (observed for target='CH4' at the
  default 1e-2 - use a smaller learning_rate, e.g. 1e-4, for that target).
  """

  def loss_theta(theta):
    return loss_fn(
      theta,
      emis_dict_JAX=emis_dict_JAX,
      delT_dict_FaIR=delT_dict_FaIR,
      dt=dt,
    )

  loss_and_grad = jax.value_and_grad(loss_theta)

  optimizer = optax.adam(learning_rate=learning_rate)
  opt_state = optimizer.init(theta0)

  theta = theta0
  mask = make_theta_mask(theta0, target=target)

  for step in range(n_steps):
      loss_value, grads = loss_and_grad(theta)
      grads = grads * mask
      updates, opt_state = optimizer.update(grads, opt_state, theta)
      theta = optax.apply_updates(theta, updates)

      if step % 10 == 0:
          print(f"step {step:4d} | loss_avg = {float(loss_value):.5f}")
          with open(filepath, 'wb') as f:
              pickle.dump(theta, f)

  theta_opt = theta
  print("Final loss_avg:", float(loss_value))
  return params_from_theta(theta_opt), loss_value

# ==================================================================
# Part 2b: GeoMIP sulfur-injection inverse solve
# ==================================================================

def solve_sulfur_inverse(
    emissions_H_jax: jnp.ndarray,    # (5, T) array: Full emissions for Scenario H
    target_temp_M: jnp.ndarray,      # (T,) array: Target GMST from Scenario M
    years: jnp.ndarray,              # (T,) array: Year coordinates
    mode: str = 'FaIR',
    params: dict | None = None,
    learning_rate: float = 0.1,
    n_steps: int = 2000,
    reg_weight: float = 0.1      # Regularization weight to prevent jagged emissions
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solves for sulfur emissions that force the model to match target_temp_M,
    given background emissions from Scenario H.
    """

    # 1. Initialization
    # We start with the Sulfur emissions from H as our initial guess.
    # We want to optimize only the sulfur row (index 3).
    sulfur_initial_guess = emissions_H_jax[idx_Sulfur]

    # We define the optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(sulfur_initial_guess)

    # 2. Define the Loss Function
    def inverse_loss_fn(sulfur_profile):
        # Construct the full emissions matrix
        # Take H emissions, but swap out the Sulfur row with our optimization variable
        emissions_current = emissions_H_jax.at[idx_Sulfur].set(sulfur_profile)

        # Run the forward model
        res = simulate_temp(years, emissions_current, mode=mode, params=params)
        T_pred = res["GMST"]

        # A. Primary Loss: Mean Squared Error on Temperature
        mse_loss = jnp.mean((T_pred - target_temp_M)**2)

        # B. Regularization (Crucial for inverse problems)
        # Without this, the solver might produce physically unrealistic,
        # highly oscillating emissions to fit numerical noise.
        # We penalize large changes in emissions (first-order difference).
        diff = jnp.diff(sulfur_profile)
        smoothness_penalty = jnp.mean(diff**2)

        return mse_loss + (reg_weight * smoothness_penalty)

    # 3. Optimization Loop
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

    final_emissions = emissions_H_jax.at[idx_Sulfur].set(current_sulfur)
    final_res = simulate_temp(years, final_emissions, mode=mode, params=params)
    final_T_pred = final_res["GMST"]

    return current_sulfur, final_T_pred


