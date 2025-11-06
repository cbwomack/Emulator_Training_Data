# train_unroll.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from features import standardize_fit, standardize_apply, ScalerState

jax.config.update("jax_enable_x64", True)

def _make_train_step(model, optimizer):
    # model and optimizer are closed over (static for JIT)
    def mse_loss(params, batch_x, batch_y, rng, train: bool):
        yhat = model.apply({"params": params}, batch_x, rng=rng, train=train)  # (B, 1)
        yhat = yhat.squeeze(-1)  # (B,)
        return jnp.mean((yhat - batch_y) ** 2)

    @jax.jit
    def train_step(params, opt_state, batch_x, batch_y, rng):
        loss, grads = jax.value_and_grad(mse_loss)(params, batch_x, batch_y, rng, True)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step

def _make_epoch_batches(X: jnp.ndarray, y: jnp.ndarray, batch_size: int, perm: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Deterministic batching: apply a fixed permutation, then split into batches
    (drop the last incomplete batch to keep shapes static).
    X: (N, 1, D), y: (N,)
    """
    N = X.shape[0]
    Nb = (N // batch_size) * batch_size
    idx = perm[:Nb]
    Xb = X[idx]
    yb = y[idx]
    Xb = Xb.reshape(-1, batch_size, *Xb.shape[1:])  # (Batches, B, 1, D)
    yb = yb.reshape(-1, batch_size)                # (Batches, B)
    return Xb, yb

def train_lstm_through(
    U: jnp.ndarray,
    lstm_init: Tuple[Any, FrozenDict],
    opt_init_state: optax.OptState,
    K: int,
    train_cfg: Dict[str, Any],
    gen_fn,             # callable(U, gen_cfg, physics_params) -> (X, y, aux)
    gen_cfg: Dict[str, Any],
    physics_params,
    model,
    optimizer,
    rng_key,
):
    """
    Unrolled optimizer over exactly K steps. Grads flow into U through the data path.
    Steps:
      1) (X_raw, y_raw) from gen_fn(U, ...)
      2) Standardize features (fit on train-only)
      3) Deterministic batching; cycle through batches during the K steps.
    """
    batch_size = int(train_cfg.get("batch_size", 32))

    # Generate training data deterministically from U
    X_raw, y_raw, aux = gen_fn(U, gen_cfg, physics_params)    # X_raw (T,1,D), y_raw (T,)

    # Standardize (fit on train)
    T, _, D = X_raw.shape
    X2d = X_raw.reshape(T, D)
    scaler = standardize_fit(X2d)
    X_std = standardize_apply(X2d, scaler).reshape(T, 1, D)

    # Create fixed permutation for determinism
    key_perm, key_loop = jax.random.split(rng_key, 2)
    perm = jax.random.permutation(key_perm, T)

    # Batch once (static shapes)
    Xb, yb = _make_epoch_batches(X_std, y_raw, batch_size, perm)
    n_batches = Xb.shape[0]

    # Unpack/prepare trainables
    params = lstm_init[1]
    opt_state = opt_init_state

    # Build a jitted train_step that closes over model/optimizer
    train_step = _make_train_step(model, optimizer)

    def body(carry, k):
        params, opt_state = carry
        b = k % n_batches
        params, opt_state, _loss = train_step(params, opt_state, Xb[b], yb[b], key_loop)
        return (params, opt_state), _loss

    (params_K, opt_state_K), losses = jax.lax.scan(body, (params, opt_state), jnp.arange(K))
    return params_K, opt_state_K, scaler
