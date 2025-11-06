# inverse.py
from __future__ import annotations
from functools import partial
from typing import Dict, Any, Tuple, Sequence
import jax
import jax.numpy as jnp
import optax

from physics import PhysicsParams, emissions_to_temperature
from features import make_training_data_from_U, make_eval_data_from_targets, ScalerState
from lstm import init_lstm, apply_lstm, TinyLSTM
from train_unroll import train_lstm_through

jax.config.update("jax_enable_x64", True)

# ---------- Regularizers on U ----------
def penalty_cumsum_nonneg(U: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jax.nn.softplus(-jnp.cumsum(U)))

def tv(U: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.abs(U[1:] - U[:-1]))

def l2diff(U: jnp.ndarray) -> jnp.ndarray:
    d = U[1:] - U[:-1]
    return jnp.sum(d * d)

def eval_rmse_on_targets(
    params_K, model, scaler: ScalerState,
    W_list,  # tuple/list of (Te, 1, D)
    Y_list,  # tuple/list of (Te,)
) -> jnp.ndarray:
    from features import standardize_apply
    D = W_list[0].shape[-1]

    def _rmse_one(W, ytrue):
        Te = W.shape[0]
        W2 = W.reshape(Te, D)
        W2s = standardize_apply(W2, scaler).reshape(Te, 1, D)
        # model returns (Te, 1); squeeze ONCE → (Te,)
        yh = model.apply({"params": params_K}, W2s, rng=jax.random.PRNGKey(0), train=False)
        yh = yh.squeeze(-1)
        return jnp.sqrt(jnp.mean((yh - ytrue) ** 2))

    rmses = jnp.array([_rmse_one(Wi, Yi) for Wi, Yi in zip(W_list, Y_list)])
    return rmses.mean()

# ---------- Inverse objective ----------
def inverse_objective(
    U: jnp.ndarray,
    *,
    lstm_init, opt_init_state, K: int,
    train_cfg: Dict[str, Any],
    gen_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    physics_params: PhysicsParams,
    model: TinyLSTM,
    optimizer,
    rng_key,
    lambda_tv: float = 1e-4,
    lambda_l2: float = 1e-5,
    lambda_cum: float = 1.0,
) -> jnp.ndarray:
    # Train through (U)
    params_K, opt_state_K, scaler = train_lstm_through(
        U, lstm_init, opt_init_state, K,
        train_cfg, make_training_data_from_U, gen_cfg, physics_params,
        model, optimizer, rng_key,
    )

    # Evaluate on targets (provided by caller, already in pipeline convention)
    W_list = eval_cfg["W_list"]  # tuple of (Te,1,D)
    Y_list = eval_cfg["Y_list"]  # tuple of (Te,)
    rmse = eval_rmse_on_targets(params_K, model, scaler, W_list, Y_list)

    # Regularization
    reg = (lambda_tv * tv(U) +
           lambda_l2 * l2diff(U) +
           lambda_cum * penalty_cumsum_nonneg(U))

    return rmse + reg

# ---------- Minimal runnable demo ----------
def demo_minimal(T: int = 750, K: int = 50) -> None:
    # Physics defaults as requested
    p = PhysicsParams(
        ecs=3.0, ohtr=1.23,
        a=jnp.array([0.2173, 0.2240, 0.2824, 0.2763], dtype=jnp.float64),
        tau=jnp.array([1e6, 394.4, 36.54, 4.304], dtype=jnp.float64),
        dt=1.0,
        C0=277.15
    )

    # Model/opt
    rng = jax.random.PRNGKey(0)
    hidden, dense, dropout = 16, 32, 0.0
    input_dim = 3  # default features: U, Catm, RF
    model, params0 = init_lstm(rng, input_dim, hidden, dense, dropout, train=True)
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-3)
    opt_state0 = optimizer.init(params0)
    lstm_init = (model, params0)

    # Training config (deterministic batches)
    train_cfg = dict(batch_size=32, dropout_rate=0.0)
    gen_cfg   = dict(features=("U", "Catm", "RF"))

    # Build synthetic target scenarios W_i (just to exercise the path):
    # For two targets, e.g., two different emissions → temps we want the LSTM to predict well.
    U_tgt1 = jnp.zeros((T,), dtype=jnp.float64).at[100:200].set(0.5)  # small pulse
    U_tgt2 = jnp.linspace(0.0, 1.0, T, dtype=jnp.float64) * 0.2        # gentle ramp
    # Build features the same way as training features (so eval uses same columns)
    from features import make_training_data_from_U
    W1, Y1, _ = make_training_data_from_U(U_tgt1, gen_cfg, p)
    W2, Y2, _ = make_training_data_from_U(U_tgt2, gen_cfg, p)
    eval_cfg = dict(W_list=(W1, W2), Y_list=(Y1, Y2))

    # Initialize U to zero (train emissions)
    U0 = jnp.zeros((T,), dtype=jnp.float64)

    # One meta step on U:
    @jax.jit
    def step_on_U(U, optU_state, key):
        def obj(U_):
            return inverse_objective(
                U_, lstm_init=lstm_init, opt_init_state=opt_state0, K=K,
                train_cfg=train_cfg, gen_cfg=gen_cfg, eval_cfg=eval_cfg,
                physics_params=p, model=model, optimizer=optimizer, rng_key=key,
                lambda_tv=1e-4, lambda_l2=1e-5, lambda_cum=1.0
            )
        val, gU = jax.value_and_grad(obj)(U)
        updates, optU_state = optax.adam(1e-2).update(gU, optU_state, U)
        U = optax.apply_updates(U, updates)
        return U, optU_state, val

    optU = optax.adam(1e-2)
    optU_state = optU.init(U0)
    key = jax.random.PRNGKey(42)

    U1, optU_state, val0 = step_on_U(U0, optU_state, key)
    U2, optU_state, val1 = step_on_U(U1, optU_state, key)

    print("Objective before:", float(val0))
    print("Objective after: ", float(val1))

    return U0, U1, U2

if __name__ == "__main__":
    demo_minimal()
