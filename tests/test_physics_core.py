# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5 and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Tier 1: pure JAX SCM physics-core functions (utils_FaIR_JAX.py). No I/O,
deterministic, and the load-bearing numerical building blocks under every
downstream pipeline function - highest value per the Phase 4 test plan.

Each test checks one of: zero-forcing/zero-emissions steady state (no
spurious constant term in the reservoir ODEs), a single-step response against
a hand-computed value (verified against the actual function output before
being pinned here, not derived independently), monotonicity where physically
expected, or output shape/dtype.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import utils_FaIR_JAX as m

PARAMS = m.FAIR_PARAMS


# ---------------------------
# carbon_step
# ---------------------------

def test_carbon_step_zero_emissions_steady_state():
    # zero emissions in, empty reservoirs -> no spurious constant term, stays at zero
    cpool0 = jnp.zeros(4)
    out = m.carbon_step(cpool0, 0.0, PARAMS, dt=1.0, alpha=1.0)
    np.testing.assert_allclose(out, np.zeros(4))


def test_carbon_step_single_step_matches_hand_computed_value():
    # Et=1, alpha=1, dt=1, cpool=0 -> dydt = a*1 - 0 = a, so cpool_new = a exactly
    cpool0 = jnp.zeros(4)
    out = m.carbon_step(cpool0, 1.0, PARAMS, dt=1.0, alpha=1.0)
    np.testing.assert_allclose(out, np.asarray(PARAMS["a"]), rtol=1e-6)


def test_carbon_step_shape_and_dtype():
    cpool0 = jnp.zeros(4)
    out = m.carbon_step(cpool0, 1.0, PARAMS, dt=0.1, alpha=1.0)
    assert out.shape == (4,)
    assert jnp.issubdtype(out.dtype, jnp.floating)


# ---------------------------
# radiative_forcing_co2
# ---------------------------

def test_radiative_forcing_co2_zero_at_preindustrial():
    # C == C0 -> log(C/C0) == 0 -> RF == 0 regardless of the piecewise coefficient
    rf = m.radiative_forcing_co2(jnp.array(PARAMS["C0_PI"]), PARAMS)
    np.testing.assert_allclose(rf, 0.0, atol=1e-6)


def test_radiative_forcing_co2_monotonically_increasing_with_concentration():
    # more CO2 -> more forcing, at both the mid-range and high-range piecewise branches
    C0 = PARAMS["C0_PI"]
    rf_lo = m.radiative_forcing_co2(jnp.array(1.5 * C0), PARAMS)
    rf_hi = m.radiative_forcing_co2(jnp.array(2.0 * C0), PARAMS)
    assert 0.0 < float(rf_lo) < float(rf_hi)


# ---------------------------
# methane_step
# ---------------------------

def test_methane_step_zero_emissions_steady_state():
    out = m.methane_step(0.0, 0.0, 0.0, PARAMS, dt=1.0)
    np.testing.assert_allclose(out, 0.0, atol=1e-8)


def test_methane_step_single_step_matches_hand_computed_value():
    # m=0, T=0 -> alpha=exp(0)=1, tau_eff=tau (unclamped); dm/dt = E_ppb - 0
    out = m.methane_step(0.0, 1.0, 0.0, PARAMS, dt=1.0)
    np.testing.assert_allclose(out, m.PPB_PER_MTCH4, rtol=1e-6)


# ---------------------------
# radiative_forcing_ch4
# ---------------------------

def test_radiative_forcing_ch4_zero_at_preindustrial():
    rf = m.radiative_forcing_ch4(jnp.array(m.M0_CH4_PI), PARAMS)
    np.testing.assert_allclose(rf, 0.0, atol=1e-6)


def test_radiative_forcing_ch4_monotonically_increasing_with_concentration():
    rf_lo = m.radiative_forcing_ch4(jnp.array(m.M0_CH4_PI + 100.0), PARAMS)
    rf_hi = m.radiative_forcing_ch4(jnp.array(m.M0_CH4_PI + 500.0), PARAMS)
    assert 0.0 < float(rf_lo) < float(rf_hi)


# ---------------------------
# radiative_forcing_SO2_BC
# ---------------------------

def test_radiative_forcing_SO2_BC_zero_at_zero_emissions():
    rf = m.radiative_forcing_SO2_BC(jnp.array(0.0), jnp.array(0.0), PARAMS)
    np.testing.assert_allclose(rf, 0.0, atol=1e-6)


def test_radiative_forcing_SO2_BC_sulfur_cools_and_bc_warms():
    # SO2 (sulfate aerosols) is a net cooling agent here, BC (soot) a net warming
    # one - opposite signs, both away from zero as emissions increase.
    rf_so2 = m.radiative_forcing_SO2_BC(jnp.array(10.0), jnp.array(0.0), PARAMS)
    rf_bc = m.radiative_forcing_SO2_BC(jnp.array(0.0), jnp.array(10.0), PARAMS)
    assert float(rf_so2) < 0.0
    assert float(rf_bc) > 0.0


# ---------------------------
# n2o_step
# ---------------------------

def test_n2o_step_zero_emissions_steady_state():
    out = m.n2o_step(jnp.array(0.0), jnp.array(0.0), PARAMS, dt=1.0)
    np.testing.assert_allclose(out, 0.0, atol=1e-8)


def test_n2o_step_single_step_matches_hand_computed_value():
    # n=0 -> dn/dt = emis2conc * E_TgN - 0
    out = m.n2o_step(jnp.array(0.0), jnp.array(1.0), PARAMS, dt=1.0)
    np.testing.assert_allclose(out, PARAMS["n2o_emis2conc"], rtol=1e-6)


# ---------------------------
# radiative_forcing_n2o
# ---------------------------

def test_radiative_forcing_n2o_zero_at_preindustrial():
    rf = m.radiative_forcing_n2o(jnp.array(m.M0_N2O_PI), PARAMS)
    np.testing.assert_allclose(rf, 0.0, atol=1e-6)


def test_radiative_forcing_n2o_monotonically_increasing_with_concentration():
    rf_lo = m.radiative_forcing_n2o(jnp.array(m.M0_N2O_PI + 10.0), PARAMS)
    rf_hi = m.radiative_forcing_n2o(jnp.array(m.M0_N2O_PI + 50.0), PARAMS)
    assert 0.0 < float(rf_lo) < float(rf_hi)


# ---------------------------
# climate_step
# ---------------------------

def test_climate_step_zero_forcing_steady_state():
    # zero forcing in, empty thermal boxes -> no spurious constant term
    S0 = jnp.zeros(3)
    S_new, T_new = m.climate_step(S0, 0.0, PARAMS, dt=1.0)
    np.testing.assert_allclose(S_new, np.zeros(3))
    assert float(T_new) == pytest.approx(0.0)


def test_climate_step_single_step_matches_hand_computed_value():
    # S=0, RF=1, dt=1 -> dS/dt = q*1/d, so S_new = q/d exactly; T_new = sum(S_new)
    S0 = jnp.zeros(3)
    S_new, T_new = m.climate_step(S0, 1.0, PARAMS, dt=1.0)
    expected_S = np.asarray(PARAMS["q"]) / np.asarray(PARAMS["d"])
    np.testing.assert_allclose(S_new, expected_S, rtol=1e-6)
    assert float(T_new) == pytest.approx(float(np.sum(expected_S)), rel=1e-6)


def test_climate_step_shape_and_dtype():
    S0 = jnp.zeros(3)
    S_new, T_new = m.climate_step(S0, 2.5, PARAMS, dt=0.1)
    assert S_new.shape == (3,)
    assert jnp.issubdtype(S_new.dtype, jnp.floating)
    assert jnp.ndim(T_new) == 0
