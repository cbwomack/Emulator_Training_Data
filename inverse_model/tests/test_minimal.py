# tests/test_minimal.py
import jax
import jax.numpy as jnp
from physics import PhysicsParams, emissions_to_temperature
from inverse import demo_minimal
from features import standardize_fit, standardize_apply

jax.config.update("jax_enable_x64", True)

def test_shapes_and_dtypes():
    T = 16
    U = jnp.zeros((T,), dtype=jnp.float64)
    p = PhysicsParams(
        ecs=3.0, ohtr=1.23,
        a=jnp.array([0.2173, 0.2240, 0.2824, 0.2763], dtype=jnp.float64),
        tau=jnp.array([1e6, 394.4, 36.54, 4.304], dtype=jnp.float64),
        dt=1.0
    )
    out = emissions_to_temperature(U, p)
    assert out.T.shape == (T,)
    assert out.Catm.shape == (T,)
    assert out.RF.shape == (T,)
    assert out.T.dtype == jnp.float64

def test_scaler():
    X = jnp.arange(12.0, dtype=jnp.float64).reshape(6,2)
    st = standardize_fit(X)
    Xs = standardize_apply(X, st)
    assert Xs.shape == (6,2)
    assert jnp.isfinite(Xs).all()

def test_grad_through_physics():
    T = 16
    U = jnp.linspace(0, 1, T, dtype=jnp.float64)
    p = PhysicsParams(
        ecs=3.0, ohtr=1.23,
        a=jnp.array([0.2173, 0.2240, 0.2824, 0.2763], dtype=jnp.float64),
        tau=jnp.array([1e6, 394.4, 36.54, 4.304], dtype=jnp.float64),
        dt=1.0
    )
    def f(U_):
        return emissions_to_temperature(U_, p).T.sum()
    g = jax.grad(f)(U)
    assert g.shape == (T,)
    assert jnp.isfinite(g).all()

def test_demo_runs():
    # Just ensure it executes without error and prints two floats
    demo_minimal(T=64, K=10)
