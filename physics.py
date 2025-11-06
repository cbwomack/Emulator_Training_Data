# physics.py
from __future__ import annotations
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

# ----------------------------
# Constants / defaults
# ----------------------------
C0_DEFAULT = 277.15  # ppm (preindustrial)
A_DEFAULT = jnp.array([0.2173, 0.2240, 0.2824, 0.2763], dtype=jnp.float64)
TAU_DEFAULT = jnp.array([1e6, 394.4, 36.54, 4.304], dtype=jnp.float64)

# AR6-like forcing coefficient polynomial coefficients (as in your code)
A1 = jnp.float64(-2.4785e-7)
B1 = jnp.float64( 7.5906e-4)
D1 = jnp.float64( 5.2488)

# climate geometry/props
EARTH_AREA = jnp.float64(5.1e14)      # m^2
LFRAC      = jnp.float64(0.292)       # land fraction [-]
LTHK       = jnp.float64(8.4)         # land layer thickness [m]
OTHK       = jnp.float64(100.0)       # ocean mixed layer thickness [m]
OMHC       = jnp.float64(4186.0)      # J kg-1 K-1
ODENS      = jnp.float64(1000.0)      # kg m-3
DTHK       = jnp.array([300.0, 300.0, 1300.0, 1800.0], dtype=jnp.float64)  # deep ocean [m]
SPY        = jnp.float64(3600.0*24.0*365.25)

class PhysicsParams(NamedTuple):
    ecs: float
    ohtr: float
    a: jnp.ndarray   # (4,)
    tau: jnp.ndarray # (4,)
    dt: float        # years
    C0: float = C0_DEFAULT

class PhysicsOut(NamedTuple):
    T: jnp.ndarray     # (T,) surface anomaly [K]
    Catm: jnp.ndarray  # (T,) ppm
    RF: jnp.ndarray    # (T,) W/m^2

def _coefC_piecewise(C: jnp.ndarray, C0: float) -> jnp.ndarray:
    """
    Vectorized piecewise rule used inside RF = coefC(C)*log(C/C0).
    Mirrors your numpy logic, but JAX-friendly.
    """
    Camax = C0 - B1/(2.0*A1)
    # regions:
    # mask1: C >= Camax
    # mask2: C < Camax & C > C0
    # mask3: C <= C0
    coef_default = jnp.full_like(C, D1)
    coef1 = D1 - (B1**2) / (D1*A1)
    coef2 = D1 + A1*(C - C0)**2 + B1*(C - C0)

    # select via where cascades:
    coef = jnp.where(C >= Camax, coef1, coef_default)    # mask1
    coef = jnp.where((C < Camax) & (C > C0), coef2, coef) # mask2 (middle)
    # mask3 keeps D1 (already in coef_default)
    return coef

def _radiative_forcing(C: jnp.ndarray, C0: float) -> jnp.ndarray:
    coef = _coefC_piecewise(C, C0)
    return coef * jnp.log(C / C0)

class ClimateCarry(NamedTuple):
    cpool: jnp.ndarray     # (4,)
    tanm_ao: jnp.float64   # scalar surf anomaly
    dQ: jnp.ndarray        # (4,) deep-ocean heat content (per-area) [J m^-2]
    aoQ: jnp.float64       # surface heat content (per-area) [J m^-2]

class ClimateConsts(NamedTuple):
    ao_hc: jnp.float64     # surf heat capacity per-area [J m^-2 K^-1]
    d_hc: jnp.ndarray      # (4,) deep layer heat cap per-area
    mlayert: jnp.ndarray   # (4,) mid-layer thicknesses used in htr profile
    feedback: jnp.float64  # W m^-2 K^-1 from ECS
    htc: jnp.float64       # W m^-2 K^-1 * m (transport coeff times mixed-layer mid-thk)

def _make_climate_consts(ecs: float, ohtr: float) -> ClimateConsts:
    # per-area heat capacities
    ao_vol = EARTH_AREA * (LFRAC*LTHK + (1.0 - LFRAC)*OTHK)
    ao_vhc = OMHC * ODENS
    ao_hc  = (ao_vol * ao_vhc) / EARTH_AREA

    d_vol  = (1.0 - LFRAC) * EARTH_AREA * DTHK
    d_vhc  = ao_vhc
    d_hc   = (d_vol * d_vhc) / EARTH_AREA

    # layers thickness vector (surf + deep)
    layert  = jnp.concatenate([jnp.array([OTHK]), DTHK])
    mlayert = (layert[:-1] + layert[1:]) / 2.0

    # feedback parameter (RF@2x / ECS)
    rf_2x = _radiative_forcing(jnp.array([2.0*C0_DEFAULT]), C0_DEFAULT)[0]
    feedback = rf_2x / ecs

    # ocean heat transport coeff scaled by first mid-layer thickness
    htc = ohtr * mlayert[0]
    return ClimateConsts(ao_hc=ao_hc, d_hc=d_hc, mlayert=mlayert, feedback=feedback, htc=htc)

def _carbon_step(cpool: jnp.ndarray, Et: jnp.float64, params: PhysicsParams) -> Tuple[jnp.ndarray, jnp.float64]:
    # Joos multi-box linear uptake
    dydt = params.a * Et - cpool / params.tau
    cpool_next = cpool + params.dt * dydt
    Catm = params.C0 + cpool_next.sum()
    return cpool_next, Catm

def _climate_step(
    tanm_ao: jnp.float64, dQ: jnp.ndarray, aoQ: jnp.float64,
    RF_prev: jnp.float64, consts: ClimateConsts, dt: float
) -> Tuple[jnp.float64, jnp.ndarray, jnp.float64, jnp.float64]:
    """
    One explicit step (mirrors your _climateModel):
    - inputs RF_prev (forcing at previous time), surface T anomaly, deep Q, surface Q
    - returns updated (tanm_ao, dQ, aoQ, RF_used)
    """
    # cooling from feedback
    feedcool = tanm_ao * consts.feedback

    # inter-layer heat transports
    # htr[0] between surface and top deep layer (use anomalies)
    htr0 = (tanm_ao - (dQ[0] / consts.d_hc[0])) * consts.htc / consts.mlayert[0]

    # remaining transports between deep layers (from enthalpies converted to temps)
    dT = dQ / consts.d_hc
    dT_up = dT[:-1]
    dT_dn = dT[1:]
    htr_rest = (dT_up - dT_dn) * consts.htc / consts.mlayert[1:]  # shape (3,)

    # surface heat content step
    aoQ_new = aoQ + dt * (RF_prev - feedcool - htr0) * SPY

    # deep heat content step
    dQ_new = dQ.at[:-1].set(dQ[:-1] + dt * (jnp.concatenate([jnp.array([htr0]), htr_rest[:-1]]) - htr_rest) * SPY)
    dQ_new = dQ_new.at[-1].set(dQ[-1] + dt * htr_rest[-1] * SPY)

    # new surface anomaly
    tanm_ao_new = aoQ_new / consts.ao_hc
    return tanm_ao_new, dQ_new, aoQ_new, RF_prev

def emissions_to_temperature(U: jnp.ndarray, params: PhysicsParams) -> PhysicsOut:
    """
    U: (T,) emissions (ppm/yr) → ΔT(t), Catm(t), RF(t)
    - exact defaults and forcing rule preserved
    - all with lax.scan → differentiable
    """
    consts = _make_climate_consts(params.ecs, params.ohtr)

    def step(carry: Tuple[ClimateCarry, jnp.float64], Et: jnp.float64):
        climate, RF_prev = carry

        # Carbon step
        cpool_next, Catm = _carbon_step(climate.cpool, Et, params)

        # New RF from new Catm (for next step; pass previous RF to climate step)
        RF_curr = _radiative_forcing(jnp.array([Catm]), params.C0)[0]

        # Climate step uses RF_prev (consistent with your code)
        tanm_ao_next, dQ_next, aoQ_next, _ = _climate_step(
            climate.tanm_ao, climate.dQ, climate.aoQ, RF_prev, consts, params.dt
        )
        climate_next = ClimateCarry(cpool_next, tanm_ao_next, dQ_next, aoQ_next)

        # outputs at this time: T (new), Catm (new), RF (new)
        y = jnp.array([tanm_ao_next, Catm, RF_curr])
        return (climate_next, RF_curr), y

    # initial state
    carry0 = ClimateCarry(
        cpool=jnp.zeros((4,), dtype=jnp.float64),
        tanm_ao=jnp.float64(0.0),
        dQ=jnp.zeros((4,), dtype=jnp.float64),
        aoQ=jnp.float64(0.0)
    )
    RF0 = jnp.float64(0.0)
    (carryT, _), ys = lax.scan(step, (carry0, RF0), U)
    T = ys[:, 0]
    Catm = ys[:, 1]
    RF = ys[:, 2]
    return PhysicsOut(T=T, Catm=Catm, RF=RF)

# Batched wrapper for many U sequences
batched_emissions_to_temperature = jax.vmap(emissions_to_temperature, in_axes=(0, None))
