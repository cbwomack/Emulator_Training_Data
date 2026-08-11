# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5 and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

# -------
# Imports
# -------
import utils_FaIR_JAX
import numpy as np
import matplotlib.pyplot as plt
import pickle
import xarray as xr
import json
from pathlib import Path

# JAX
import jax
import jax.numpy as jnp
from jax import lax
import optax
from jax import tree as _jtree

import os, pickle, numpy as np

## Setup plots
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "sans-serif",
  "font.sans-serif": ["Helvetica Light"],
})


# Choose the row order for the emissions matrix passed to simulate_temp
AGENTS_DEFAULT = ("CO2", "CH4", "N2O", "Sulfur", "BC")  # rows in this order
# (simulate_temp references idx_CO2 / idx_CH4; keep these consistent)
idx_CO2, idx_CH4, idx_N2O, idx_Sulfur, idx_BC = 0, 1, 2, 3, 4

# ==================================================================
# Part 3: feature engineering, dataset construction, and the MLP emulator
# ==================================================================

def _as_jnp(x: jnp.ndarray | np.ndarray | float, dtype: type = jnp.float32) -> jnp.ndarray:
    """Coerce x to a jnp array of the given dtype."""
    return jnp.asarray(x, dtype=dtype)

def _prev_and_cumu_prev(E_curr: jnp.ndarray, E_hist: jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Causal previous-year emission and cumulative-to-previous-year."""
  E_curr = _as_jnp(E_curr).reshape(-1)
  T  = E_curr.shape[0]

  if E_hist is not None and _as_jnp(E_hist).size > 0:
    E_hist = _as_jnp(E_hist).reshape(-1)
    H  = E_hist.shape[0]
    E_all = jnp.concatenate([E_hist, E_curr], axis=0)

    # indices (H-1) .. (H+T-2) — clamp for safety if H==0
    start = jnp.maximum(H - 1, 0)
    E_prev = E_all[start:start+T]
    Cumu_all = jnp.cumsum(E_all)
    Cumu_prev = Cumu_all[start:start+T]
  else:
    zero = jnp.array([0.0], dtype=E_curr.dtype)
    E_prev   = jnp.concatenate([zero, E_curr[:-1]], axis=0)        # (T,)
    Cumu_curr = jnp.cumsum(E_curr)
    Cumu_prev = jnp.concatenate([zero, Cumu_curr[:-1]], axis=0)  # (T,)
  return E_prev, Cumu_prev

def _ema(x: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
  """
  Causal EMA via lax.scan: y_t = (1-alpha) y_{t-1} + alpha x_t
  Returns y for all t (same length as x). Init y_0 = x_0 * alpha (light bias).
  """
  x = _as_jnp(x)
  alpha = _as_jnp(alpha, dtype=x.dtype)
  y0 = alpha * x[0]  # small bias toward first value
  def step(y_prev, x_t):
    y_t = (1.0 - alpha) * y_prev + alpha * x_t
    return y_t, y_t
  _, ys = lax.scan(step, y0, x[1:])
  ys = jnp.concatenate([jnp.expand_dims(y0, 0), ys], axis=0)
  return ys

def _ema_prev(E_curr: jnp.ndarray, E_hist: jnp.ndarray | None, alpha: jnp.ndarray) -> jnp.ndarray:
  """
  EMA computed on concatenated history+current, then aligned causally:
  feature at t uses EMA up to t-1 (no leakage).
  """
  E_curr = _as_jnp(E_curr).reshape(-1)
  if E_hist is not None and _as_jnp(E_hist).size > 0:
    E_hist = _as_jnp(E_hist).reshape(-1)
    E_all = jnp.concatenate([E_hist, E_curr], axis=0)
    ema_all = _ema(E_all, alpha)
    H = E_hist.shape[0]
    # want EMA at indices (H-1) .. (H+T-2)

    start = jnp.maximum(H - 1, 0)
    T = E_curr.shape[0]
    ema_prev = ema_all[start:start+T]
  else:
    ema_curr = _ema(E_curr, alpha)
    # shift by one with zero at t=0
    zero = jnp.array([0.0], dtype=E_curr.dtype)
    ema_prev = jnp.concatenate([zero, ema_curr[:-1]], axis=0)
  return ema_prev

def make_features_emissions_generic(
    emis_curr_dict: dict,              # dict: agent -> (T,) emissions
    emis_hist_dict: dict | None = None,         # dict or None: agent -> (H,) emissions
    agents: tuple = AGENTS_DEFAULT,        # tuple/list of agents to include (order = column grouping)
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
    dt_years: float = 1.0,
    zero_fill_missing: bool = True,
) -> jnp.ndarray:
  """
  Builds causal, per-agent features:
    For each agent a in `agents`, columns (in this order):
      1) E_prev[a]            (previous-year emissions)
      2) EMA_short_prev[a]    (EMA over ~5 years, causal, aligned to t-1)
      3) EMA_long_prev[a]     (EMA over ~30 years, causal, aligned to t-1)
      4) EMA_long_prev[a]     (EMA over ~100 years, causal, aligned to t-1)
      5) CumE_prev[a]         (cumulative emissions up to t-1)

  - Handles any subset of agents (e.g., just CO2, just CH4, both).
  - If `zero_fill_missing=True`, missing agents are zeroed (keeps column layout stable).
  - Uses dt_years to set EMA alphas: alpha = 1 - exp(-dt / window).
  - Returns:
      X: (T, 5*len(agents)) feature matrixX
  """
  # Determine T from the first available current series
  T = None
  for a in agents:
    if emis_curr_dict.get(a, None) is not None:
      T = _as_jnp(emis_curr_dict[a]).reshape(-1).shape[0]
      break
  if T is None:
    raise ValueError("No current emissions provided for any agent in `agents`.")

  # Precompute EMA alphas
  w_short, w_long, w_vlong = ema_windows_years
  alpha_short = 1.0 - jnp.exp(-_as_jnp(dt_years) / _as_jnp(w_short))
  alpha_long  = 1.0 - jnp.exp(-_as_jnp(dt_years) / _as_jnp(w_long))
  alpha_vlong  = 1.0 - jnp.exp(-_as_jnp(dt_years) / _as_jnp(w_vlong))

  feats = []
  for a in agents:
    E_curr = emis_curr_dict.get(a, None)
    if E_curr is None:
      if not zero_fill_missing:
        raise ValueError(f"Missing current emissions for agent '{a}' and zero_fill_missing=False.")
      E_curr = jnp.zeros((T,), dtype=jnp.float32)
    else:
      E_curr = _as_jnp(E_curr).reshape(-1)
      if E_curr.shape[0] != T:
        raise ValueError(f"Agent '{a}' length mismatch: got {E_curr.shape[0]} != {T}.")

    E_hist = None
    if emis_hist_dict is not None and a in emis_hist_dict and emis_hist_dict[a] is not None:
        E_hist = _as_jnp(emis_hist_dict[a]).reshape(-1)

    # 2a) previous-year + cumulative-to-previous
    E_prev, Cum_prev = _prev_and_cumu_prev(E_curr, E_hist)

    # 2b) EMAs (short ~5y, long ~30y), aligned causally to t-1
    emaS_prev = _ema_prev(E_curr, E_hist, alpha_short)
    emaL_prev = _ema_prev(E_curr, E_hist, alpha_long)
    emavL_prev = _ema_prev(E_curr, E_hist, alpha_vlong)

    # stack per-agent in required order
    agent_X = jnp.stack([E_prev, emaS_prev, emaL_prev, emavL_prev, Cum_prev], axis=1)  # (T, 4)
    feats.append(agent_X)

  X = jnp.concatenate(feats, axis=1) if len(feats) > 1 else feats[0]
  return X

def _infer_step(yrs: jnp.ndarray) -> jnp.ndarray:
    """Median spacing between consecutive years (1.0 if fewer than 2 points)."""
    yrs = _as_jnp(yrs).reshape(-1)
    if yrs.size <= 1:
        return jnp.array(1.0, dtype=jnp.float32)
    return jnp.median(jnp.diff(yrs))

def _make_contiguous_years(yrs_hist: jnp.ndarray, yrs_curr: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Rebase yrs_curr so its first point is exactly one step after yrs_hist[-1].
    Keeps the *spacing* of yrs_curr. If yrs_curr already follows yrs_hist,
    we leave it unchanged.
    """
    yrs_h = _as_jnp(yrs_hist).reshape(-1)
    yrs_c = _as_jnp(yrs_curr).reshape(-1)

    if (yrs_h.size == 0) or (yrs_c.size == 0):
        return jnp.concatenate([yrs_h, yrs_c]), yrs_c

    # If already contiguous (or strictly after), do nothing
    if yrs_c[0] > yrs_h[-1]:
        yrs_all = jnp.concatenate([yrs_h, yrs_c])
        return yrs_all, yrs_c

    # Otherwise, shift current so it starts one "curr step" after hist end
    step_c   = _infer_step(yrs_c)
    shift    = (yrs_h[-1] + step_c) - yrs_c[0]
    yrs_c_rb = yrs_c + shift
    yrs_all  = jnp.concatenate([yrs_h, yrs_c_rb])
    return yrs_all, yrs_c_rb

def _stack_emissions(agents: tuple, emis_dict: dict, T: int, dtype: type = jnp.float32) -> jnp.ndarray:
    """
    Build (N_agents, T) with rows in `agents` order.
    Missing agents are zero-filled.
    """
    rows = []
    for a in agents:
        arr = emis_dict.get(a, None)
        if arr is None:
            rows.append(jnp.zeros((T,), dtype=dtype))
        else:
            arr = _as_jnp(arr).reshape(-1)
            if arr.shape[0] != T:
                raise ValueError(f"Length mismatch for agent '{a}': {arr.shape[0]} != {T}")
            rows.append(arr.astype(dtype))
    return jnp.stack(rows, axis=0)  # (N_agents, T)

def simulate_targets_gmst(
    years_curr: jnp.ndarray,                         # (T_cur,)
    emis_curr_dict: dict,                     # dict: {"CO2": (T_cur,), "CH4": (T_cur,), ...}
    years_hist: jnp.ndarray | None = None,                    # (T_hist,) or None
    emis_hist_dict: dict | None = None,                # dict or None: {"CO2": (T_hist,), "CH4": (T_hist,), ...}
    agents: tuple = AGENTS_DEFAULT,              # tuple/list: which agents to include and their row order
    mode: str = 'FaIR',
    dt: float = 0.1
) -> jnp.ndarray:
    """
    Returns GMST over the current scenario's years using the new multi-agent simulator.

    Inputs:
      - emis_curr_dict values are in native units per agent:
          CO2: GtCO2/yr, CH4: MtCH4/yr (consistent with simulate_temp)
      - Missing agents in emis_*_dict are zero-filled (keeps column layout stable).
    """

    yrs_c = _as_jnp(years_curr).reshape(-1)
    Tcur  = yrs_c.shape[0]

    has_hist = (years_hist is not None) and (emis_hist_dict is not None)

    if has_hist:
        yrs_h = _as_jnp(years_hist).reshape(-1)
        yrs_all, _yrs_c_rb = _make_contiguous_years(yrs_h, yrs_c)

        # Build per-agent arrays for history/current, zero-filling as needed
        # Determine T per segment
        Th = yrs_h.shape[0]

        # Prepare per-agent concatenated series into one matrix (N_agents, Th+Tcur)
        emis_all = []
        for a in agents:
            Eh = _as_jnp(emis_hist_dict.get(a, jnp.zeros((Th,), jnp.float32))).reshape(-1)
            Ec = _as_jnp(emis_curr_dict.get(a, jnp.zeros((Tcur,), jnp.float32))).reshape(-1)
            if Eh.shape[0] != Th or Ec.shape[0] != Tcur:
                raise ValueError(f"Agent '{a}' length mismatch (hist {Eh.shape[0]} vs {Th}, curr {Ec.shape[0]} vs {Tcur})")
            emis_all.append(jnp.concatenate([Eh, Ec], axis=0))
        emissions_by_agent_all = jnp.stack(emis_all, axis=0)  # (N_agents, Th+Tcur)

        out_all = utils_FaIR_JAX.simulate_temp(
            years=yrs_all,
            emissions_by_agent=emissions_by_agent_all,
            mode=mode,
            dt=dt
        )
        GMST_curr = out_all["GMST"][-Tcur:]
    else:
        # No history: stack current only (zero-fill missing)
        emissions_by_agent_c = _stack_emissions(agents, emis_curr_dict, Tcur)
        out_c = utils_FaIR_JAX.simulate_temp(
            years=yrs_c,
            emissions_by_agent=emissions_by_agent_c,
            mode=mode,
            dt=dt
        )
        GMST_curr = out_c["GMST"]

    return GMST_curr

def extract_years_and_emis(emis_entry_for_scenario: jnp.ndarray, agents: tuple = AGENTS_DEFAULT) -> tuple[jnp.ndarray, dict]:
    """
    Input:
      emis_entry_for_scenario: array-like with shape (N_agents, T),
        rows correspond to agents in `agents` order (e.g., CO2, CH4).
        Units: CO2 in GtCO2/yr, CH4 in MtCH4/yr.

    Output:
      years: (T,) float32
      emis_curr_dict: {agent: (T,) float32}
    """
    A = jnp.asarray(emis_entry_for_scenario, dtype=jnp.float32)
    if A.ndim != 2:
        raise ValueError("emis_entry_for_scenario must have shape (N_agents, T).")
    n_rows, T = int(A.shape[0]), int(A.shape[1])

    # Align rows to the requested agents list
    if n_rows < len(agents):
        # pad missing agents with zeros
        pad = jnp.zeros((len(agents) - n_rows, T), dtype=A.dtype)
        A = jnp.concatenate([A, pad], axis=0)
    elif n_rows > len(agents):
        # drop extra rows (assumes the first len(agents) rows match the requested agents)
        A = A[:len(agents), :]

    years = jnp.asarray(np.arange(T), dtype=jnp.float32)
    emis_curr_dict = {agent: A[i, :] for i, agent in enumerate(agents)}
    return years, emis_curr_dict


scens_with_hist = ['H-ext','H-ext-OS','M',
                   'M-ext','ML','ML-ext','L',
                   'L-ext','VLLO-ext','VLHO','VLHO-ext',
                   'AA','CT']

def build_dataset_from_runfair_dict(
    emis_dict: dict,
    historical_name: str = "historical",
    agents: tuple = AGENTS_DEFAULT,
    mode: str = 'FaIR',
    ema_windows_years: tuple = (5.0, 30.0, 100.0)
) -> list[tuple[jnp.ndarray, jnp.ndarray, str]]:
    """
    Returns list of (X_features, y_target, scenario_name), using:
      - X_features built by make_features_emissions_generic (per-agent prev, 5y/30y EMA, cum_prev)
      - y_target via simulate_targets_gmst (multi-agent simulator wrapper)
    """
    # Historical series
    years_hist, emis_hist_dict = (None, None)
    if historical_name in emis_dict:
        years_hist, emis_hist_dict = extract_years_and_emis(emis_dict[historical_name], agents=agents)

    dataset = []
    skip_hist = False
    for scen in emis_dict.keys():
        if skip_hist and scen == historical_name:
          continue

        if scen == historical_name and not skip_hist:
            skip_hist = True

        yrs_cur, emis_cur_dict = extract_years_and_emis(emis_dict[scen], agents=agents)
        needs_history = (scen != historical_name) and (years_hist is not None) and (scen in scens_with_hist)

        if needs_history:
            years_hist, emis_hist_dict = CS3_hist_modifier(scen, years_hist, emis_hist_dict)

        # --- Features (native units; causal, zero-fills handled upstream) ---
        X = make_features_emissions_generic(
            emis_curr_dict=emis_cur_dict,
            emis_hist_dict=(emis_hist_dict if needs_history else None),
            agents=agents,
            ema_windows_years=ema_windows_years,
            dt_years=1.0,
            zero_fill_missing=True,
        )

        # --- Targets ---
        y = simulate_targets_gmst(
            years_curr=yrs_cur,
            emis_curr_dict=emis_cur_dict,
            years_hist=(years_hist if needs_history else None),
            emis_hist_dict=(emis_hist_dict if needs_history else None),
            mode=mode
        )

        N = min(X.shape[0], y.shape[0])
        dataset.append((X[:N], y[:N], scen))

    return dataset

def fit_scaler(X: jnp.ndarray, eps: float = 1e-8) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    """Per-column standardize X (zero mean, unit var). Returns (Xs, (mu, sd))."""
    mu = jnp.mean(X, axis=0)
    sd = jnp.sqrt(jnp.var(X, axis=0) + eps)
    Xs = (X - mu) / sd
    return Xs, (mu, sd)

def apply_scaler(X: jnp.ndarray, stats: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """Apply a (mu, sd) scaler fit elsewhere (e.g. by fit_scaler) to X."""
    mu, sd = stats
    return (X - mu) / sd

def _infer_feat_dim(train_dataset: list, test_dataset: list) -> int:
    """Feature width of the first (X, ...) row found across train/test datasets, else 0."""
    for ds in (train_dataset, test_dataset):
        for (X, *_rest) in ds:
            return int(X.shape[1])
    return 0

def split_and_scale(train_dataset: list, test_dataset: list) -> tuple[list, list, tuple[jnp.ndarray, jnp.ndarray]]:
    """
    JAX-safe: lists of (X, y, scen) -> scaled train/test and (mu, sd) from train.
    Works for any feature width (e.g., 4*len(agents)).
    """
    D = _infer_feat_dim(train_dataset, test_dataset)
    if D == 0:
        # Degenerate case: nothing to scale
        return [], [], (jnp.zeros((0,)), jnp.ones((0,)))

    # Concatenate train features along rows
    Xtr_list = [jnp.asarray(X, dtype=jnp.float32) for (X, _, _) in train_dataset]
    Xtr = jnp.concatenate(Xtr_list, axis=0) if Xtr_list else jnp.zeros((0, D), jnp.float32)

    # Fit scaler on the concatenated train features
    Xtr_s, stats = fit_scaler(Xtr)

    # Slice standardized rows back per scenario
    train_scaled = []
    offset = 0
    for (X, y, scen) in train_dataset:
        n = int(X.shape[0])
        Xs_slice = Xtr_s[offset:offset+n]
        train_scaled.append((Xs_slice, jnp.asarray(y, dtype=jnp.float32), scen))
        offset += n

    # Apply scaler to test features
    test_scaled = []
    for (X, y, scen) in test_dataset:
        Xs = apply_scaler(jnp.asarray(X, dtype=jnp.float32), stats)
        test_scaled.append((Xs, jnp.asarray(y, dtype=jnp.float32), scen))

    return train_scaled, test_scaled, stats

def init_mlp_params(key: jax.random.PRNGKey, input_dim: int, hidden_sizes: list[int]) -> list[dict]:
    """
    Initialize params for a standard MLP.

    Args:
        key: jax.random.PRNGKey
        input_dim: size of the input feature vector (flattened)
        hidden_sizes: list of integers defining nodes per layer, e.g. [64, 64]
                      len = num. of hidden layers, value in each layer = num. neurons

    Returns:
        List of dicts [{'W':.., 'b':..}, ...] including the final output layer.
    """
    params = []
    # The architecture flows from input -> hidden_1 -> ... -> hidden_n -> output (scalar)
    layer_dims = [input_dim] + hidden_sizes + [1]

    keys = jax.random.split(key, len(layer_dims) - 1)

    for i in range(len(layer_dims) - 1):
        in_d, out_d = layer_dims[i], layer_dims[i+1]

        # Xavier/Glorot initialization
        lim = jnp.sqrt(6.0 / (in_d + out_d))
        W = jax.random.uniform(keys[i], (in_d, out_d), minval=-lim, maxval=lim)
        b = jnp.zeros((out_d,))

        params.append({'W': W, 'b': b})

    return params

def mlp_forward(params: list[dict], X: jnp.ndarray) -> jnp.ndarray:
    """
    Standard Feedforward Neural Network.

    Args:
        params: List of layer dicts initialized by init_mlp_params
        X: Input features. Shape (N, D) or (N, T, D).
           If time (T) is present, it is flattened into the feature dimension.

    Returns:
        (N,) scalar output array
    """
    # 1. Ensure input is float32 (or matches param dtype)
    # We grab the dtype from the first layer's weights
    dtype = params[0]["W"].dtype
    X = X.astype(dtype)

    # 2. Flatten inputs
    # If X is (N, T, D), this becomes (N, T*D).
    # If X is (N, D), this stays (N, D).
    N = X.shape[0]
    activations = X.reshape(N, -1)

    # 3. Forward pass through hidden layers (all but the last)
    for layer in params[:-1]:
        linear = activations @ layer['W'] + layer['b']
        activations = jnp.tanh(linear)

    # 4. Final Output Layer (Linear, no activation)
    final_layer = params[-1]
    y = activations @ final_layer['W'] + final_layer['b']

    # Squeeze to return shape (N,) matching the old output format
    return y.squeeze(-1)

def _mse(pred: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error, casting pred to y's dtype first."""
    pred = pred.astype(y.dtype)
    return jnp.mean((pred - y)**2)

_tree_map    = _jtree.map
def train_mlp_sgd(
    params0: list[dict], Xtr: jnp.ndarray, ytr: jnp.ndarray, K: int = 400, lr: float = 5e-2, weight_decay: float = 1e-2,
    batch_size: int | None = None, key: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> tuple[list[dict], jnp.ndarray]:
    """
    Train an MLP (mlp_forward) with clipped-gradient SGD + weight decay for K steps,
    scanning the whole loop with jax.lax.scan (checkpointed) for speed/memory.
    Returns (params_final, losses) where losses has one entry per step.

    `batch_size`: default None reproduces the prior behavior exactly - every
    step uses the complete Xtr/ytr (despite the "SGD" name, this was always
    full-batch gradient descent with momentum, not minibatch-stochastic; see
    0a_inner_loop_sgd_pilot). When set to an int, each of the K steps samples
    `batch_size` rows without replacement from Xtr/ytr using `key` (split
    fresh every step), making this genuinely stochastic gradient descent in
    the classical sense. `batch_size >= Xtr.shape[0]` falls back to full-batch.
    """
    pdt = params0[0]["W"].dtype
    Xtr = Xtr.astype(pdt); ytr = ytr.astype(pdt)
    N = Xtr.shape[0]
    use_minibatch = batch_size is not None and batch_size < N

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.add_decayed_weights(weight_decay),
        optax.sgd(lr)
    )
    opt_state = optimizer.init(params0)

    @jax.checkpoint
    def step(carry, _):
        params, opt_state, step_key = carry

        if use_minibatch:
            step_key, sample_key = jax.random.split(step_key)
            idx = jax.random.choice(sample_key, N, shape=(batch_size,), replace=False)
            Xb, yb = Xtr[idx], ytr[idx]
        else:
            Xb, yb = Xtr, ytr

        def loss_fn(p):
            yhat = mlp_forward(p, Xb)
            return _mse(yhat, yb)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = _tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6), grads)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        return (params, opt_state, step_key), loss

    (paramsK, _, _), losses = jax.lax.scan(step, (params0, opt_state, key), xs=None, length=K)
    return paramsK, losses

# ==================================================================
# Part 3/4: core bilevel inverse optimization (shared by single- and
# multi-agent, FaIR- and MESM-calibrated experiments)
# ==================================================================

def freeze_inactive_agents(emis_dict: dict, active_agents: tuple = ("CO2",), agents: tuple = AGENTS_DEFAULT) -> dict:
    """Stop gradients flowing through agents not in `active_agents` (values unchanged, just detached)."""
    out = {}
    for a in agents:
        x = emis_dict[a]
        out[a] = x if (a in active_agents) else jax.lax.stop_gradient(x)
    return out

def _apply_active_mask_to_emis(U_pytree: dict, active_agents: tuple, inactive_mode: str = "zeros") -> dict:
    """Zero (optionally stop-gradient) every agent in U_pytree not in active_agents."""
    def squash(u, active):
        if active:
            return u
        if inactive_mode == "stop_grad_zeros":
            return jax.lax.stop_gradient(jnp.zeros_like(u))
        # default: pure zeros (still fine if we also mask grads/updates):
        return jnp.zeros_like(u)
    return {ag: squash(U_pytree[ag], ag in active_agents) for ag in U_pytree}

# Reuse the helper from earlier to accept dict OR (N_agents, T) arrays
def _normalize_emissions_input(U_in: dict | jnp.ndarray, agents: tuple, T: int, dtype: type = jnp.float32) -> dict:
    """Coerce U_in (dict of per-agent series, or a (N_agents, T) array) to {agent: (T,)}, zero-filling/padding as needed."""
    if isinstance(U_in, dict):
        out = {}
        for a in agents:
            v = U_in.get(a, None)
            if v is None:
                out[a] = jnp.zeros((T,), dtype=dtype)
            else:
                v = jnp.asarray(v, dtype=dtype).reshape(-1)
                if v.shape[0] != T:
                    raise ValueError(f"Input for agent '{a}' has length {v.shape[0]} != {T}.")
                out[a] = v
        return out
    else:
        A = jnp.asarray(U_in, dtype=dtype)
        if A.ndim != 2:
            raise ValueError("Emissions input must be dict or a (N_agents, T) array.")
        if A.shape[1] != T:
            raise ValueError(f"Input time length {A.shape[1]} != {T}.")
        if A.shape[0] < len(agents):
            pad = jnp.zeros((len(agents) - A.shape[0], T), dtype=dtype)
            A = jnp.concatenate([A, pad], axis=0)
        A = A[:len(agents), :]
        return {a: A[i, :] for i, a in enumerate(agents)}

def build_train(
    U_in: dict | jnp.ndarray,                              # dict {agent:(T,)} OR array (N_agents,T)
    agents: tuple = AGENTS_DEFAULT,
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
    dtype: type = jnp.float32,
    years_hist: jnp.ndarray | None = None,
    emis_hist_dict: dict | None = None,
    mode: str = 'FaIR'
) -> list[tuple[jnp.ndarray, jnp.ndarray, str]]:
    """
    Returns a single-scenario train dataset:
      [(X, y, 'opt_scen')]

    If years_hist and emis_hist_dict are provided, they are used as
    historical context (just like build_dataset_from_runfair_dict),
    so features and GMST are built with history prepended.
    """
    # Infer T from input
    if isinstance(U_in, dict):
        for a in agents:
            if a in U_in and U_in[a] is not None:
                T = int(jnp.asarray(U_in[a]).reshape(-1).shape[0])
                break
        else:
            raise ValueError("Cannot infer T: no agent series provided in dict.")
    else:
        A = jnp.asarray(U_in)
        if A.ndim != 2:
            raise ValueError("U_in must be dict or (N_agents, T) array.")
        T = int(A.shape[1])

    years_cur = jnp.arange(T, dtype=dtype)

    # Normalize emissions to dict {agent:(T,)}
    emis_curr_dict = _normalize_emissions_input(U_in, agents=agents, T=T, dtype=dtype)

    has_hist = (years_hist is not None) and (emis_hist_dict is not None)

    # Features (with history if available)
    X = make_features_emissions_generic(
        emis_curr_dict=emis_curr_dict,
        emis_hist_dict=(emis_hist_dict if has_hist else None),
        agents=agents,
        ema_windows_years=ema_windows_years,
        dt_years=1.0,
        zero_fill_missing=True,
    )

    # Targets (GMST with history if available)
    y = simulate_targets_gmst(
        years_curr=years_cur,
        emis_curr_dict=emis_curr_dict,
        years_hist=(years_hist if has_hist else None),
        emis_hist_dict=(emis_hist_dict if has_hist else None),
        mode=mode
    ).astype(dtype)

    N = min(X.shape[0], y.shape[0])
    return [(X[:N], y[:N], "opt_scen")]

def build_valid(
    emis_dict_valid: dict,
    historical_name: str = "historical",
    agents: tuple = AGENTS_DEFAULT,
    mode: str = 'FaIR',
    ema_windows_years: tuple = (5.0, 30.0, 100.0)
) -> list[tuple[jnp.ndarray, jnp.ndarray, str]]:
    """
    Combined validation set:
      - Tier-1 with historical context (where available)
      - Tier-2 without adding a historical sample (skip_hist=True)
      - DECK subset with NO historical context
    """
    valid = build_dataset_from_runfair_dict(
        emis_dict_valid,
        historical_name=historical_name, agents=agents, mode=mode, ema_windows_years=ema_windows_years
    )
    return valid

def make_objective_over_emissions(
    scen_name_tr1: str,
    train_data: list, test_data: list,
    emis_dict: dict,
    params0: list[dict],
    historical_name: str = "historical",
    dtype: type = jnp.float32,
    agents: tuple = AGENTS_DEFAULT,
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
    active_agents: tuple = ("CO2",),
    mode: str = 'FaIR'
) -> callable:
    """
    Build a closure `objective_over_emissions(U_in)` that: substitutes U_in as the
    `scen_name_tr1` training scenario's emissions, rebuilds that scenario's
    features/targets, trains a fresh MLP on the updated training set, and returns
    the resulting mean test NRMSE. This is the single-scenario building block used
    by optimize_emissions_inverse's outer loop (see make_inverse_objective_single_train
    for the multi-scenario version actually used there).
    """
    # --- Historical context (if present) ---
    years_hist, emis_hist_dict = (None, None)
    if historical_name in emis_dict:
        years_hist, emis_hist_dict = extract_years_and_emis(
            emis_entry_for_scenario=emis_dict[historical_name],
            agents=agents
        )
        years_hist = jnp.asarray(years_hist, dtype=dtype)

    # --- Current scenario years (emissions provided at call-time) ---
    years_cur, _emis_cur_ignored = extract_years_and_emis(
        emis_entry_for_scenario=emis_dict[scen_name_tr1],
        agents=agents
    )
    years_cur = jnp.asarray(years_cur, dtype=dtype)
    T = int(years_cur.shape[0])
    needs_history = (scen_name_tr1 != historical_name) and (years_hist is not None)

    if needs_history:
        years_hist, emis_hist_dict = CS3_hist_modifier(scen_name_tr1, years_hist, emis_hist_dict)

    def objective_over_emissions(U_in):
        """
        U_in: either dict {agent: (T,)} or array (N_agents, T) in `agents` order.
        Returns scalar mean test MSE after inner MLP train.
        """
        # 0) Normalize incoming emissions for the chosen training scenario
        emis_tr1_dict = _normalize_emissions_input(U_in, agents=agents, T=T, dtype=dtype)
        emis_tr1_dict = freeze_inactive_agents(emis_tr1_dict, active_agents=active_agents, agents=agents)

        # 1) Features for chosen training scenario (causal, with EMAs & cumulative-to-prev)
        X_tr1_new = make_features_emissions_generic(
            emis_curr_dict=emis_tr1_dict,
            emis_hist_dict=(emis_hist_dict if needs_history else None),
            agents=agents,
            ema_windows_years=ema_windows_years,
            dt_years=1.0,
            zero_fill_missing=True,
        )

        # 2) Targets from the simulator wrapper (GMST)
        y_tr1_new = simulate_targets_gmst(
            years_curr=years_cur,
            emis_curr_dict=emis_tr1_dict,
            years_hist=(years_hist if needs_history else None),
            emis_hist_dict=(emis_hist_dict if needs_history else None),
            mode=mode
        ).astype(dtype)

        # 3) Rebuild the full training dataset (this scenario updated, others unchanged)
        train_dataset_updated = [(X_tr1_new, y_tr1_new, scen_name_tr1)]
        for (X, y, scen) in train_data:
            if scen == scen_name_tr1:
                continue
            train_dataset_updated.append((
                jnp.asarray(X, dtype=dtype),
                jnp.asarray(y, dtype=dtype),
                scen
            ))

        # 4) Scale with updated train stats; apply to tests
        train_s_updated, test_s_updated, _stats = split_and_scale(
            train_dataset_updated, test_data
        )

        # 5) Concatenate updated train tensors
        Xtr_u = jnp.concatenate([X for (X, _, _) in train_s_updated], axis=0).astype(dtype)
        ytr_u = jnp.concatenate([y for (_, y, _) in train_s_updated], axis=0).astype(dtype)

        # 6) Inner MLP training
        paramsK_u, _ = train_mlp_sgd(
            params0, Xtr_u, ytr_u, K=400, lr=5e-2, weight_decay=1e-4
        )

        # 7) Evaluate mean test NRMSE
        return avg_nrmse_over_tests(paramsK_u, test_s_updated).astype(dtype)

    return objective_over_emissions

def init_constant_emissions(
    T: int,
    emis_val: float = 50.0,
    dtype: type = jnp.float32,
) -> jnp.ndarray:
    """Constant initial emissions."""
    return jnp.ones(T, dtype=dtype) * emis_val

def init_ramp_emissions(
    T: int,
    final_val: float = 50.0,
    dtype: type = jnp.float32,
) -> jnp.ndarray:
    """Ramp initial emissions."""
    return jnp.linspace(0, int(final_val), T)

def init_gaussian_emissions(
    T: int,
    peak_emis: float = 50.0,
    mu: float = 350.0,
    sigma: float = 100.0,
    floor: float = 0.0,
    dtype: type = jnp.float32,
) -> jnp.ndarray:
    """Gaussian-shaped initial emissions."""
    t = jnp.arange(T, dtype=dtype)
    z = (t - jnp.array(mu, dtype=dtype)) / jnp.array(sigma, dtype=dtype)
    E = floor + jnp.array(peak_emis, dtype=dtype) * jnp.exp(-0.5 * z * z)
    return jnp.maximum(jnp.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0), 0.0).astype(dtype)

def init_sine_emissions(
    T: int,
    peak_emis: float = 50.0,
    period: float = 150.0,
    dtype: type = jnp.float32,
) -> jnp.ndarray:
    """Ramp initial emissions."""
    t = jnp.arange(T, dtype=dtype)
    return peak_emis * jnp.sin(2 * jnp.pi * t / period).astype(dtype)


def make_init_emissions_pytree(
    T: int,
    agents: tuple = AGENTS_DEFAULT,
    active_agents: tuple | None = None,
    init_cond: str | jnp.ndarray | None = None,               # str (keyword) | array-like (matrix) | None
    inactive_mode: str = "zeros", # "zeros" or "stop_grad_zeros"
    dtype: type = jnp.float32,
) -> dict:
    """
    Build initial emissions as a PyTree {agent: (T,)}.

    init_cond can be:
      - str: keyword ("gaussian", "constant", "ramp", "cos") applied to all active agents,
             using distinct default parameters per agent.
      - array-like (N_agents, T): rows correspond to `agents` order.
      - None: defaults to "gaussian".

    Inactive agents are zeroed.
    """
    # 1. Setup
    if active_agents is None:
        active_agents = tuple(agents)

    # Default parameters per agent for different initialization modes
    # Modify these values to tune the "different parameters" per forcing agent
    AGENT_CONFIGS = {
        "CO2":    {"peak": 60.0,  "mu": 350, "sig_frac": 0.15, "period": 500},
        "CH4":    {"peak": 750.0, "mu": 450, "sig_frac": 0.25, "period": 500},
        "N2O":    {"peak": 10.0,  "mu": 350, "sig_frac": 0.15, "period": 250},
        "Sulfur": {"peak": 60.0,  "mu": 350, "sig_frac": 0.15, "period": 250},
        "BC":     {"peak": 10.0,  "mu": 350, "sig_frac": 0.15, "period": 250},
        # Fallback for unknown agents
        "DEFAULT": {"peak": 50.0, "mu": 350, "sig_frac": 0.15, "period": 250},
    }

    def _get_zero():
        z = jnp.zeros((T,), dtype=dtype)
        return jax.lax.stop_gradient(z) if inactive_mode == "stop_grad_zeros" else z

    def _make_from_keyword(agent, keyword):
        cfg = AGENT_CONFIGS.get(agent, AGENT_CONFIGS["DEFAULT"])
        key = keyword.lower().strip()

        if key == "constant":
            return init_constant_emissions(
                T=T,
                emis_val=cfg["peak"],
                dtype=dtype
            )
        elif key == "ramp":
            return init_ramp_emissions(
                T=T,
                final_val=cfg["peak"],
                dtype=dtype
            )
        elif key == "gaussian":
            return init_gaussian_emissions(
                T=T,
                peak_emis=cfg["peak"],
                mu=cfg["mu"],
                sigma=cfg["sig_frac"] * T,
                floor=0.0,
                dtype=dtype
            )
        elif key == "sine":
            return init_sine_emissions(
                T=T,
                peak_emis=cfg["peak"],
                period=cfg["period"],
                dtype=dtype
            )

        else:
            raise ValueError(f"Unknown init_cond keyword: '{keyword}'. Supported: constant, ramp, gaussian, sine.")

    # 2. Handle Matrix Input (N_agents, T)
    if hasattr(init_cond, "shape") or isinstance(init_cond, (list, tuple, np.ndarray)):
        # Assume array-like
        arr = jnp.asarray(init_cond, dtype=dtype)
        if arr.ndim != 2:
            raise ValueError(f"init_cond matrix must be (N_agents, T), got shape {arr.shape}")
        if arr.shape[0] != len(agents):
            raise ValueError(f"init_cond rows {arr.shape[0]} != len(agents) {len(agents)}")
        if arr.shape[1] != T:
            raise ValueError(f"init_cond cols {arr.shape[1]} != T {T}")

        U = {}
        for i, ag in enumerate(agents):
            if ag in active_agents:
                U[ag] = arr[i].reshape(-1)
            else:
                U[ag] = _get_zero()
        return U

    # 3. Handle Keyword Input (str) or None
    if init_cond is None:
        init_cond = "constant"

    if isinstance(init_cond, str):
        U = {}
        for ag in agents:
            if ag in active_agents:
                U[ag] = _make_from_keyword(ag, init_cond)
            else:
                U[ag] = _get_zero()
        return U

    raise TypeError(f"init_cond must be a keyword string, a matrix, or None. Got: {type(init_cond)}")


def make_time_weights(T: int, power: float = 2.0, min_scale: float = 0.1) -> jnp.ndarray:
    """Per-timestep optimizer weight ramping from min_scale (t=0) to 1.0 (t=T-1), as t^power."""
    x = jnp.linspace(0.0, 1.0, T)
    w = x ** power
    return min_scale + (1.0 - min_scale) * w  # (T,)

def make_time_weights_pytree(T: int, agents: tuple = AGENTS_DEFAULT, power: float = 2.0, min_scale: float = 0.15) -> dict:
    """Same weight vector (see make_time_weights) broadcast to every agent."""
    w = make_time_weights(T, power=power, min_scale=min_scale)
    return {a: w for a in agents}

def avg_nrmse_over_tests(params: list[dict], test_list: list, eps: float = 1e-8) -> jnp.ndarray:
    """
    For each scenario:
      NRMSE = RMSE(yhat, ytrue) / max(|ytrue|)
    Then average NRMSE across scenarios weighted by scenario length.

    eps prevents division by zero when ytrue is all zeros.
    """
    if not test_list:
        return jnp.array(0.0, dtype=jnp.float32)

    def nrmse_one(Xte, yte):
        yhat = mlp_forward(params, Xte).astype(yte.dtype)
        return _nrmse(yhat, yte, eps)

    vals = [nrmse_one(X, y) for (X, y, _) in test_list]
    weights = jnp.array([len(y) for (_, y, _) in test_list])
    return jnp.average(jnp.stack(vals), weights=weights).astype(jnp.float32)


def make_inverse_objective_single_train(
    params0: list[dict],
    test_dataset_all: list,
    K_inner: int = 400,
    lr_inner: float = 5e-2,
    wd_inner: float = 1e-2,
    agents: tuple = AGENTS_DEFAULT,
    active_agents: tuple = ("CO2",),
    inactive_mode: str = "zeros",
    smoothness_weight: float = 0.0,
    mode: str = 'FaIR',
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
    batch_size: int | None = None,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> callable:
    """
    Build the bilevel objective `objective(U_pytree) -> (loss, aux)` at the heart of
    optimize_emissions_inverse: given a candidate emissions trajectory U_pytree,
    (1) mask inactive agents, (2) build a single-scenario train set from U_pytree
    (build_train), (3) train a fresh MLP emulator on it for K_inner SGD steps
    (train_mlp_sgd) - this is the "inner" optimization, (4) evaluate that emulator's
    mean NRMSE on `test_dataset_all`, plus an optional first-difference smoothness
    penalty on U_pytree. The outer gradient (w.r.t. U_pytree) is obtained by
    differentiating through the entire inner training loop.
    aux = (paramsK, test_s, train_temp_raw) for logging/checkpointing.

    `batch_size`/`key` are forwarded to train_mlp_sgd's inner loop (default
    None = full-batch, unchanged behavior; see 0a_inner_loop_sgd_pilot).
    Known limitation of this pilot-stage passthrough: `key` is fixed for the
    life of one optimize_emissions_inverse call, so every outer step's inner
    minibatch draws are seeded identically - true per-outer-step-varying
    minibatches would require threading a key through the outer loop's own
    JIT'd carry, which is out of scope until 0a's pilot results are in.
    """
    def objective(U_pytree):
        U_eff = _apply_active_mask_to_emis(U_pytree, active_agents, inactive_mode)

        reg_loss = 0.0
        if smoothness_weight > 0.0:
            for agent_name in active_agents:
                arr = U_eff[agent_name]
                # First-difference penalty: Sum of squared changes between years
                # Penalizes "jaggedness" directly.
                diffs = jnp.diff(arr)
                reg_loss += jnp.sum(diffs**2)

        train_updated = build_train(
            U_eff,
            agents=agents,
            ema_windows_years=ema_windows_years,
            dtype=jnp.float32,
            years_hist=None,
            emis_hist_dict=None,
            mode=mode
        )

        train_temp_raw = [y for (_, y, _) in train_updated]

        # 2) Scale with updated train stats; apply to ALL tests
        train_s, test_s, _stats = split_and_scale(train_updated, test_dataset_all)

        # 3) Concatenate train tensors
        Xtr = jnp.concatenate([X for (X, _, _) in train_s], axis=0).astype(jnp.float32)
        ytr = jnp.concatenate([y for (_, y, _) in train_s], axis=0).astype(jnp.float32)

        # 4) Inner training
        paramsK, _ = train_mlp_sgd(
            params0, Xtr, ytr, K=K_inner, lr=lr_inner, weight_decay=wd_inner,
            batch_size=batch_size, key=key,
        )

        # 5) Average NRMSE across all scenarios, weighted by scenario length
        nrmse_val = avg_nrmse_over_tests(paramsK, test_s)
        loss = nrmse_val + (smoothness_weight * reg_loss)

        aux  = (paramsK, test_s, train_temp_raw)
        return loss, aux
    return objective


def _tree_to_numpy(tree: dict) -> dict:
    """Recursively convert every jax array leaf in a pytree to a host numpy array."""
    return jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x)) if isinstance(x, (jnp.ndarray, jax.Array)) else x, tree
    )

def _tree_to_jnp(tree: dict) -> dict:
    """Recursively convert every numpy array leaf in a pytree to a jnp array."""
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, tree
    )

def save_inverse_ckpt(path: str, state: dict) -> None:
    """Pickle an optimize_emissions_inverse checkpoint dict to `path` (host numpy arrays)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    safe = {
        "U_traj": [_tree_to_numpy(u) for u in state["U_traj"]],
        "errors": np.asarray(state["errors"]),
        "opt_state": _tree_to_numpy(state["opt_state"]),
        "paramsK_k": _tree_to_numpy(state['paramsK_k']),
        "step_count": int(state["step_count"]),
        "time_weights": _tree_to_numpy(state.get("time_weights", None)),
        "meta": state.get("meta", {}),
        "preds_traj": state["preds_traj"],
        "train_temp_traj": state["train_temp_traj"],
    }

    with open(path, "wb") as f:
        pickle.dump(safe, f)

def load_inverse_ckpt(path: str) -> dict:
    """Load a checkpoint saved by save_inverse_ckpt, restoring jnp arrays."""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    out = {
        "U_traj": [_tree_to_jnp(u) for u in raw["U_traj"]],
        "errors": jnp.asarray(raw["errors"], dtype=jnp.float32),
        "opt_state": _tree_to_jnp(raw["opt_state"]),
        "step_count": int(raw["step_count"]),
        "time_weights": None if raw["time_weights"] is None else _tree_to_jnp(raw["time_weights"]),
        "meta": raw.get("meta", {}),
        "preds_traj": raw["preds_traj"],
        "train_temp_traj": raw["train_temp_traj"],
    }
    return out

def scale_by_coord_pytree(weights_pytree: dict) -> optax.GradientTransformation:
    """An optax transform that elementwise-multiplies gradients by a matching pytree of weights."""
    def _mul(g, w): return g * w
    def init_fn(_): return ()
    def update_fn(updates, state, params=None):
        return jax.tree_util.tree_map(_mul, updates, weights_pytree), state
    return optax.GradientTransformation(init_fn, update_fn)

def preds_by_scenario(params: list[dict], test_list: list) -> list[tuple[str, jnp.ndarray, jnp.ndarray]]:
    """Run the MLP on every (X, y, scenario) test row; returns [(scenario, y_hat, y_true), ...]."""
    out = []
    for (Xte, yte, scen) in test_list:
        yhat = mlp_forward(params, Xte)
        out.append((scen, yhat, yte))
    return out

def make_agent_mask_pytree(T: int, agents: tuple, active_agents: tuple) -> dict:
    """{agent: ones(T)} for agents in active_agents, {agent: zeros(T)} otherwise."""
    one  = lambda: jnp.ones((T,), jnp.float32)
    zero = lambda: jnp.zeros((T,), jnp.float32)
    return {ag: (one() if ag in active_agents else zero()) for ag in agents}

def mul_tree(a: dict, b: dict) -> dict:
    """Elementwise-multiply two matching pytrees."""
    return jax.tree_util.tree_map(lambda x, m: x * m, a, b)

def optimize_emissions_inverse(
    emis_dict: dict,
    params0: list[dict],
    num_updates: int = 500,
    step_size: float | dict = 1e3,
    momentum: float = 0.9,
    nesterov: bool = True,
    K_inner: int = 500,
    lr_inner: float = 5e-2,
    wd_inner: float = 1e-2,
    historical_name: str = "historical",
    agents: tuple = AGENTS_DEFAULT,
    active_agents: tuple | None = None,
    init_cond: str | jnp.ndarray | None = None,
    inactive_mode: str = "zeros",
    T: int = 750,
    n_nonneg_prefix: int | None = None,
    filter_hist: bool = False,
    smoothness_weight: float = 0.0,
    mode: str = 'FaIR',
    checkpoint_path: str | None = None,
    checkpoint_every: int = 50,
    resume_if_exists: bool = True,
    preds_every: int = 50,
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
    batch_size: int | None = None,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> dict:
    """
    The core bilevel/outer-loop optimizer: finds an emissions trajectory U (one
    (T,) series per agent, starting from `init_cond`) that minimizes the objective
    built by make_inverse_objective_single_train - i.e. the trajectory whose
    resulting training data yields the best-generalizing emulator on
    `emis_dict`'s scenarios (via build_valid). Optimizes with SGD+momentum
    (per-agent time-weighted, see make_time_weights_pytree/step_size), clips
    gradients to unit global norm, and optionally projects the first
    `n_nonneg_prefix` timesteps to be non-negative.

    Checkpoints (U trajectory, optimizer state, losses, prediction snapshots) are
    written to `checkpoint_path` every `checkpoint_every` steps and resumed from
    there if `resume_if_exists` and the file exists - this is what makes long runs
    safe to schedule/interrupt/resume.

    Returns a dict with keys "U_traj", "errors", "updates_done", "paramsK_k",
    "checkpoint_path", "preds_traj", "train_temp_traj".
    """
    import gc # Import garbage collection

    def hostify_tree(tree):
        return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)

    def hostify_preds(step_list):
        out = []
        for (sc, yh, yt) in step_list:
            yh_h = np.asarray(jax.device_get(yh))
            yt_h = np.asarray(jax.device_get(yt))
            out.append((sc, yh_h, yt_h))
        return out

    # --- Build combined test dataset ---
    test_dataset_all = build_valid(
        emis_dict,
        historical_name=historical_name,
        agents=agents,
        mode=mode,
        ema_windows_years=ema_windows_years,
    )

    if filter_hist:
        test_dataset_all = [row for row in test_dataset_all if row[2] != historical_name]

    # Extract scenario names for reconstruction later (names cannot be JIT-ed)
    test_scen_names = [row[2] for row in test_dataset_all]

    objective = make_inverse_objective_single_train(
        params0,
        test_dataset_all,
        K_inner=K_inner,
        lr_inner=lr_inner,
        wd_inner=wd_inner,
        agents=agents,
        active_agents=active_agents,
        inactive_mode=inactive_mode,
        smoothness_weight=smoothness_weight,
        mode=mode,
        batch_size=batch_size,
        key=key,
        ema_windows_years=ema_windows_years
    )

    # --- 1. Create a Pure Loss Function (No Strings) ---
    # JAX cannot JIT functions that return strings in `aux`.
    # We wrap the objective to strip the strings from 'test_s'.
    def loss_fn_pure(U):
        loss, (paramsK, test_s, train_temp_raw) = objective(U)
        # test_s is [(X, y, name), ...]. Strip 'name' for JIT.
        test_arrays = [(X, y) for (X, y, _) in test_s]
        return loss, (paramsK, test_arrays, train_temp_raw)

    # --- Optimizer Setup ---
    time_weights = make_time_weights_pytree(T=T, agents=agents, power=2.0, min_scale=0.15)

    if isinstance(step_size, dict):
        optimizer_lr = 1.0
        def apply_agent_lr(agent_name, weight_array):
            lr = step_size.get(agent_name, step_size.get('default', 1e-2))
            return weight_array * lr
        time_weights = {a: apply_agent_lr(a, w) for a, w in time_weights.items()}
    else:
        optimizer_lr = step_size

    if active_agents is not None:
        agent_mask = make_agent_mask_pytree(T, agents, active_agents)
        time_weights = mul_tree(time_weights, agent_mask)
    else:
        agent_mask = make_agent_mask_pytree(T, agents, agents)

    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        scale_by_coord_pytree(time_weights),
        optax.sgd(learning_rate=optimizer_lr, momentum=momentum, nesterov=nesterov),
    )

    # ----------------------------------------------------------------------
    # Projection Logic
    # ----------------------------------------------------------------------
    def project_nonneg_prefix(U_tree):
        if n_nonneg_prefix is None: return U_tree
        def _project_1d(u):
            prefix = jnp.maximum(u[:n_nonneg_prefix], 0.0)
            return u.at[:n_nonneg_prefix].set(prefix)
        return jax.tree_util.tree_map(_project_1d, U_tree)

    # --- 2. JIT-Compile the Update Step ---
    @jax.jit
    def update_step(U, opt_state):
        (loss, aux_pure), grads = jax.value_and_grad(loss_fn_pure, has_aux=True)(U)

        grads = mul_tree(grads, agent_mask)
        updates, new_opt_state = opt.update(grads, opt_state, params=U)
        updates = mul_tree(updates, agent_mask)

        new_U = optax.apply_updates(U, updates)
        new_U = project_nonneg_prefix(new_U)

        return new_U, new_opt_state, loss, aux_pure

    # --- Initialization / Resume ---
    preds_traj = []
    train_temp_traj = []
    updates_done = 0

    if resume_if_exists and checkpoint_path and os.path.isfile(checkpoint_path):
        print("Resuming from checkpoint...")
        ckpt = load_inverse_ckpt(checkpoint_path)
        U_traj = ckpt["U_traj"]
        errors = ckpt["errors"].tolist()
        U_pytree = U_traj[-1]
        opt_state = ckpt["opt_state"]
        updates_done = ckpt["step_count"]
        preds_traj = ckpt.get("preds_traj", [])
        train_temp_traj = ckpt.get("train_temp_traj", [])
    else:
        if not resume_if_exists and checkpoint_path and os.path.isfile(checkpoint_path):
            print("Overwriting save data...")
        else:
            print("No save data found, starting fresh...")
        U_pytree = make_init_emissions_pytree(
            T=T, agents=agents, active_agents=active_agents,
            init_cond=init_cond, inactive_mode=inactive_mode
        )
        U_pytree = project_nonneg_prefix(U_pytree)

        loss0, (paramsK0, test_arrays0, train_temp0) = loss_fn_pure(U_pytree)

        # Reconstruct string metadata for logging
        test_s0 = []
        for i, (X, y) in enumerate(test_arrays0):
            test_s0.append((X, y, test_scen_names[i]))

        rmse0 = float(loss0)
        opt_state = opt.init(U_pytree)
        U_traj = [U_pytree]
        errors = [rmse0]

        preds0 = preds_by_scenario(paramsK0, test_s0)
        preds_traj.append(hostify_preds(preds0))
        train_temp_traj.append(hostify_tree(train_temp0))

    # --- Outer Loop ---
    remaining = max(0, num_updates - updates_done)

    for _ in range(remaining):
        # 3. Call JIT-compiled step
        U_pytree, opt_state, loss_k, aux_pure = update_step(U_pytree, opt_state)

        # 4. Block until ready to prevent dispatch queue from consuming all RAM
        loss_k.block_until_ready()

        paramsK_k, test_arrays_k, train_temp_k = aux_pure
        rmse_k = float(loss_k)

        errors.append(rmse_k)
        U_traj.append(U_pytree)
        updates_done += 1

        if updates_done % preds_every == 0:
            # Re-attach scenario names
            test_s_k = []
            for i, (X, y) in enumerate(test_arrays_k):
                test_s_k.append((X, y, test_scen_names[i]))

            preds_k = preds_by_scenario(paramsK_k, test_s_k)
            preds_traj.append(hostify_preds(preds_k))
            train_temp_traj.append(hostify_tree(train_temp_k))

        # Checkpoint
        if checkpoint_path and ((updates_done % checkpoint_every) == 0):
            # Save logic...
            state = {
                "U_traj": U_traj, "errors": errors, "opt_state": opt_state,
                "step_count": updates_done, "time_weights": time_weights,
                "paramsK_k": _tree_to_numpy(paramsK_k),
                "preds_traj": preds_traj, "train_temp_traj": train_temp_traj,
            }
            save_inverse_ckpt(checkpoint_path, state)

            # Explicit GC
            gc.collect()

    # Final save logic (same as original)...
    if remaining == 0: paramsK_k = 0 # Handle case where no updates happened

    return {
        "U_traj": U_traj,
        "errors": jnp.asarray(errors, jnp.float32),
        "updates_done": updates_done,
        "paramsK_k": _tree_to_numpy(paramsK_k) if remaining > 0 else 0,
        "checkpoint_path": checkpoint_path,
        "preds_traj": preds_traj,
        "train_temp_traj": train_temp_traj,
    }

# ==================================================================
# Part 3/4: baseline & optimal emulator evaluation, notebook-facing wrappers
# ==================================================================

def prepare_baseline_data(
    emis_dict_train: dict,
    emis_dict_test: dict,
    historical_name: str = "historical",
    mode: str = 'FaIR',
    ema_windows_years: tuple = (5.0, 30.0, 100.0)
) -> tuple[list, list, tuple[jnp.ndarray, jnp.ndarray]]:
    """Build baseline train/test datasets with the same feature construction as inverse."""
    train_data = build_dataset_from_runfair_dict(emis_dict_train, historical_name=historical_name, mode=mode, ema_windows_years=ema_windows_years)
    test_data  = build_dataset_from_runfair_dict(emis_dict_test, historical_name=historical_name, mode=mode, ema_windows_years=ema_windows_years)
    # scale (fit on train, apply to test) — same as inverse
    train_s, test_s, stats = split_and_scale(train_data, test_data)
    return train_s, test_s, stats

def train_baseline_emulator(
    train_scaled: list,                 # list[(X_s, y, scen), ...]
    key: jax.random.PRNGKey,
    hidden_sizes: list[int] = [16],
    K: int = 400,
    lr: float = 5e-2,
    weight_decay: float = 1e-2,
    dtype: type = jnp.float32
) -> tuple[list[dict], tuple[jnp.ndarray, jnp.ndarray], dict]:
    """Concatenate train tensors and train via the same MLP+optimizer as inverse."""
    Xtr = jnp.concatenate([X for (X, _, _) in train_scaled], axis=0).astype(dtype)
    ytr = jnp.concatenate([y for (_, y, _) in train_scaled], axis=0).astype(dtype)

    input_dim = int(Xtr.shape[1])
    params0 = init_mlp_params(key, input_dim=input_dim, hidden_sizes=hidden_sizes)

    train_mlp_sgd_jit = jax.jit(train_mlp_sgd, static_argnames=("K",))
    paramsK, losses = train_mlp_sgd_jit(params0, Xtr, ytr, K=K, lr=lr, weight_decay=weight_decay)

    meta = dict(in_dim=input_dim, hidden_sizes=hidden_sizes, K=K, lr=lr, weight_decay=weight_decay)
    return paramsK, (jnp.mean(losses), losses), meta

def evaluate_emulator_nrmse(
    params: list[dict],
    dataset: list,          # list[(X, y, scen)]
    stats: tuple[jnp.ndarray, jnp.ndarray]             # (mu, sd) from fit on the baseline train set
) -> jnp.ndarray:
    """Mean NRMSE of an MLP over `dataset`, rescaling features with `stats` (no refit)."""
    # Apply the SAME scaler used in training
    test_scaled = []
    for (X, y, scen) in dataset:
        Xs = apply_scaler(jnp.asarray(X, dtype=jnp.float32), stats)
        test_scaled.append((Xs, jnp.asarray(y, dtype=jnp.float32), scen))
    return avg_nrmse_over_tests(params, test_scaled)

def _nrmse(yhat: jnp.ndarray, ytrue: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """RMSE(yhat, ytrue) normalized by max(|ytrue|) - delegates to
    utils_FaIR_JAX.calc_nrmse so calibration and emulator-evaluation loss share
    one NRMSE implementation."""
    return utils_FaIR_JAX.calc_nrmse(yhat, ytrue, eps=eps)

# --- helper: apply train stats to a test dataset list[(X, y, scen)] ---------
def _apply_stats_to_test(test_dataset: list, stats: tuple[jnp.ndarray, jnp.ndarray]) -> list:
    """Apply a (mu, sd) scaler to every X in a list of (X, y, scen) rows."""
    out = []
    for (X, y, scen) in test_dataset:
        Xs = apply_scaler(jnp.asarray(X, jnp.float32), stats)
        out.append((Xs, jnp.asarray(y, jnp.float32), scen))
    return out

# --- main wrapper ------------------------------------------------------------
def evaluate_baseline_over_multiple_tests(
    emis_dict_train: dict,
    eval_sets: dict,              # {"H": emis_dict_test_H, "S": emis_dict_test_S, ...}
    historical_name: str = "historical",
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    hidden_sizes: list[int] = [16],
    K: int = 400,
    lr: float = 5e-2,
    weight_decay: float = 1e-2,
    mode: str = 'FaIR',
    ema_windows_years: tuple = (5.0, 30.0, 100.0)
) -> tuple[list[dict], dict, dict, dict]:
    """
    Trains the baseline emulator ONCE using emis_dict_train,
    then evaluates on each set in `eval_sets` using the same scaler and MLP params.

    Returns:
      paramsK_base,
      per_designation_results where each value is:
        {
          "baseline_error_dict_<designation>": {scenario -> NRMSE},
          "baseline_error_mean": float,
          "test_scaled": [(Xs, y, scen), ...]  # scaled test set used
        }
    """
    # 1) Prepare baseline train data (and also one test pass, but we only keep stats)
    train_s, _, stats = prepare_baseline_data(
        emis_dict_train=emis_dict_train,
        emis_dict_test=emis_dict_train,
        historical_name=historical_name,
        mode=mode
    )

    # 2) Train baseline emulator once
    paramsK_base, (_mean_train_loss, _train_losses), meta = train_baseline_emulator(
        train_scaled=train_s,
        key=key,
        hidden_sizes=hidden_sizes,
        K=K,
        lr=lr,
        weight_decay=weight_decay,
    )

    # 3) Evaluate on each test designation
    baseline_results, baseline_preds, ground_truth = {}, {}, {}
    for eval_set, emis_dict_test in eval_sets.items():
        year_weights = []
        # build raw test dataset
        test_raw = build_dataset_from_runfair_dict(
            emis_dict_test, historical_name=historical_name, mode=mode, ema_windows_years=ema_windows_years
        )
        # scale with train stats (no refit!)
        test_s_scaled = _apply_stats_to_test(test_raw, stats)

        # compute per-scenario NRMSE and mean
        baseline_results[eval_set], baseline_preds[eval_set], ground_truth[eval_set] = {}, {}, {}
        for (Xte, yte, scen_name) in test_s_scaled:
            if eval_set not in ['Tier 1','All'] and scen_name == 'historical':
                continue
            yhat = mlp_forward(paramsK_base, Xte)
            r_s  = _nrmse(jnp.asarray(yhat), jnp.asarray(yte))

            ground_truth[eval_set][scen_name] = yte
            baseline_preds[eval_set][scen_name] = yhat
            baseline_results[eval_set][scen_name] = r_s
            year_weights.append(len(jnp.asarray(yhat)))

        baseline_results[eval_set]['mean'] = np.average(list(baseline_results[eval_set].values()), weights=year_weights)

    return paramsK_base, baseline_results, baseline_preds, ground_truth

def create_baseline(
    train_s: list, test_s: list, hidden_sizes: list[int] = [16], idx_demo: int | None = None, verbose: bool = False,
    seed: int = 0
) -> list[dict]:
    """
    Train a baseline MLP on `train_s` and report per-scenario NRMSE on `test_s`.

    `seed` controls the MLP's random initialization (default 0 reproduces the
    prior hardcoded behavior exactly). Used by multi-seed uncertainty sweeps
    (see 6a_seed_uncertainty_sweep) to vary emulator-training randomness.
    """

    def eval_test_nrmse_by_scenario(params, test_list):
        """Returns list of (scenario_name, rmse_value) for all tests."""
        out = []
        for (Xte, yte, scen_name) in test_list:
            yhat = mlp_forward(params, Xte)
            out.append((scen_name, _nrmse(yhat, yte)))
        return out

    (Xs, y, scen) = train_s[0]
    X_for_mlp = Xs.reshape((Xs.shape[0], 1, Xs.shape[1]))

    Xtr = jnp.concatenate([X for (X, _, _) in train_s], axis=0).astype(jnp.float32)  # (N, 4)
    ytr = jnp.concatenate([y for (_, y, _) in train_s], axis=0).astype(jnp.float32)

    key    = jax.random.PRNGKey(seed)
    input_dim = Xtr.shape[1]

    params0 = init_mlp_params(key, input_dim=input_dim, hidden_sizes=hidden_sizes)

    train_mlp_sgd_jit = jax.jit(train_mlp_sgd, static_argnames=("K",))
    paramsK, losses = train_mlp_sgd_jit(params0, Xtr, ytr, K=400, lr=5e-2, weight_decay=1e-2)

    if idx_demo is not None:
        import utils_plotting  # deferred: utils_plotting imports this module, so this must not be a module-level import
        (Xs_demo, y_demo, scen_demo) = test_s[idx_demo]
        utils_plotting.plot_mlp_predictions(paramsK, Xs_demo, y_demo, metric="RMSE", title_prefix=f"{scen_demo}")

    rmse_list = eval_test_nrmse_by_scenario(paramsK, test_s)
    avg_rmse  = jnp.mean(jnp.stack([r for (_, r) in rmse_list]))

    if verbose:
        # Print results
        for name, r in rmse_list:
            print(f"{name:30s}  RMSE = {float(r):.6f}")
        print(f"\nAverage test RMSE across {len(rmse_list)} scenarios: {float(avg_rmse):.6f}")

    return params0

def test_get_grad(
    scen: str, params0: list[dict], emis_dict_JAX: dict, train_data: list, test_data: list,
    agents: tuple = AGENTS_DEFAULT, active_agents=('CO2'), mode: str = 'FaIR'
) -> None:
  """
  Debug helper: print the gradient norm of make_objective_over_emissions w.r.t.
  each agent's emissions for `scen`.

  Note: default active_agents=('CO2') is the *string* 'CO2', not a 1-tuple
  (missing trailing comma) - preserved as-is; flagged for Phase 5.
  """

  years_tr1, E0 = extract_years_and_emis(emis_dict_JAX[scen])

  objective_over_emissions = make_objective_over_emissions(
      scen_name_tr1=scen,
      train_data=train_data,
      test_data=test_data,
      emis_dict=emis_dict_JAX,
      params0=params0,
      historical_name="historical",
      agents=agents,
      active_agents=active_agents,
      dtype=jnp.float32,
      mode=mode
  )

  # gradients wrt both series:
  val, g = jax.value_and_grad(objective_over_emissions, argnums=0, has_aux=False)((E0))

  # Gradient check (should be non-zero)
  print(f'Getting gradient for {scen}...')
  for key in g:
      print(f"\t||grad {key}||:", float(jnp.linalg.norm(g[key])))

  return

all_scens = {'tier1': ['historical', 'H-ext', 'M', 'ML', 'L', 'VLLO-ext', 'VLHO'],
             'tier2': ['H-ext-OS', 'M-ext', 'ML-ext', 'L-ext', 'VLHO-ext'],
             'DECK': ['abrupt-4xCO2', '1pctCO2'],
             'CS3': ['AA', 'CT'],
             'all': ['historical', 'H-ext', 'M', 'ML', 'L', 'VLLO-ext', 'VLHO', 'H-ext-OS', 'M-ext', 'ML-ext', 'L-ext', 'VLHO-ext', 'abrupt-4xCO2', '1pctCO2', 'AA', 'CT'],
}

def evaluate_optimal_emulator(
    training_paths: list[str],
    train_scenarios: list[str],
    eval_sets: dict,
    params0: list[dict] = None,
    agents: tuple = AGENTS_DEFAULT,
    active_agents: tuple | None = None,
    inactive_mode: str = "zeros",
    historical_name: str = "historical",
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    K: int = 400,
    lr: float = 5e-2,
    weight_decay: float = 1e-2,
    mode: str = 'FaIR',
    ind_effects: bool = False,
    ema_windows_years: tuple = (5.0, 30.0, 100.0)
) -> dict | tuple[dict, dict]:
    """
    For each optimize_emissions_inverse checkpoint in `training_paths` (final U in
    the trajectory), train a fresh MLP on the resulting optimal emissions and
    evaluate NRMSE against every eval_sets entry. Returns results_out
    (results_out[train_label][eval_set][scenario] = NRMSE, plus 'mean'), and also
    y_hat_all (raw predictions) if ind_effects=True.
    """
    results_out = {}

    if ind_effects:
        y_hat_all = {}

    # Ensure active_agents is iterable; default to all if None
    if active_agents is None:
        active_agents = tuple(agents)

    for i, path in enumerate(training_paths):
        train_label = train_scenarios[i]
        # 1) Load dataset
        with open(path, "rb") as f:
            raw = pickle.load(f)

        # Extract final optimized emissions (dict)
        U_traj = [_tree_to_jnp(u) for u in raw["U_traj"]]
        U_final_dict = U_traj[-1]

        # 2) Mask inactive agents (returns dict)
        U_eff_dict = _apply_active_mask_to_emis(U_final_dict, active_agents, inactive_mode)

        # 3) Stack into array (N_agents, T) for build_train
        #    Use the first agent's length to determine T
        T = U_eff_dict[agents[0]].shape[0]
        U_eff_array = _stack_emissions(agents, U_eff_dict, T)

        # 4) Build training data using the stacked array
        train_updated = build_train(
            U_eff_array,
            agents=agents,
            ema_windows_years=ema_windows_years,
            dtype=jnp.float32,
            years_hist=None,
            emis_hist_dict=None,
            mode=mode
        )

        # Scale stats on updated train
        train_s, _, stats = split_and_scale(train_updated, [])

        # Flatten training tensors
        Xtr = jnp.concatenate([X for (X, _, _) in train_s], axis=0).astype(jnp.float32)
        ytr = jnp.concatenate([y for (_, y, _) in train_s], axis=0).astype(jnp.float32)

        # 5) Inner Training
        k1, k2 = jax.random.split(key)
        in_dim = int(Xtr.shape[1])

        paramsK, _ = train_mlp_sgd(
            params0, Xtr, ytr, K=K, lr=lr, weight_decay=weight_decay
        )

        results_out[train_label] = {}
        if ind_effects:
            y_hat_all[train_label] = {}
        for test_name, emis_dict_test in eval_sets.items():
            test_raw = build_dataset_from_runfair_dict(
                emis_dict_test, historical_name=historical_name, ema_windows_years=ema_windows_years
            )
            test_s_scaled = _apply_stats_to_test(test_raw, stats)

            weights = []
            results_out[train_label][test_name] = {}
            if ind_effects:
                y_hat_all[train_label][test_name] = {}
            for (Xte, yte, scen_name) in test_s_scaled:
                if test_name not in ['Tier 1', 'All'] and scen_name == 'historical':
                    continue
                yhat = mlp_forward(paramsK, Xte)
                nrmse = float(_nrmse(jnp.asarray(yhat), jnp.asarray(yte)))
                results_out[train_label][test_name][scen_name] = nrmse
                weights.append(len(yhat))

                if ind_effects:
                    y_hat_all[train_label][test_name][scen_name] = yhat

            mean_err = np.average(list(results_out[train_label][test_name].values()), weights=weights)
            results_out[train_label][test_name]['mean'] = float(mean_err)

    if ind_effects:
        return results_out, y_hat_all

    return results_out

def CS3_hist_modifier(scen_name: str, years_hist: jnp.ndarray, emis_hist_dict: dict) -> tuple[jnp.ndarray, dict]:
    """For CS3 scenarios (AA, CT), truncate the shared historical series to their shorter 256-year record."""
    # Define the scenarios that need partial history
    SPECIAL_SCENS = ["AA", "CT"]

    if scen_name in SPECIAL_SCENS:
        SUBSET_LEN = 256

        # Slice years
        years_new = years_hist[:256]

        emis_new = {
            agent: arr[-SUBSET_LEN:]
            for agent, arr in emis_hist_dict.items()
        }
        return years_new, emis_new

    return years_hist, emis_hist_dict


# -------------------------------
# Wrapper functions for notebooks
# -------------------------------

def generate_init_params_and_train_data(
    agents: tuple, active_agents: tuple, test_scen: str, hidden_sizes: list[int] = [16],
    idx_demo: int | None = None, verbose: bool = False, mode: str = 'FaIR', ema_windows_years: tuple = (5.0, 30.0, 100.0),
    seed: int = 0
) -> tuple[list[dict], dict]:
    """
    Notebook-facing wrapper: build the tier1/tier2 train/test split (via
    utils_FaIR_JAX.generate_train_test), train the initial baseline MLP params
    (create_baseline) used to seed optimize_emissions_inverse, and print a
    gradient sanity check (test_get_grad) for `test_scen`.

    `seed` controls the MLP's random initialization (default 0 reproduces the
    prior hardcoded behavior exactly); forwarded to create_baseline.
    Returns (params0, emis_dict_train_JAX).
    """

    emis_dict_train_FaIR, emis_dict_test_FaIR, emis_dict_train_JAX, emis_dict_test_JAX, delT_dict_train_FaIR, delT_dict_test_FaIR, delT_dict_train_JAX, delT_dict_test_JAX = utils_FaIR_JAX.generate_train_test(agents, mode=mode)

    train_data = build_dataset_from_runfair_dict(emis_dict_train_JAX, mode=mode, ema_windows_years=ema_windows_years)
    test_data = build_dataset_from_runfair_dict(emis_dict_test_JAX, mode=mode, ema_windows_years=ema_windows_years)
    train_s, test_s, stats = split_and_scale(train_data, test_data)

    params0 = create_baseline(train_s, test_s, hidden_sizes=hidden_sizes, idx_demo=idx_demo, verbose=verbose, seed=seed)
    test_get_grad(test_scen, params0, emis_dict_train_JAX, train_data, test_data, active_agents=active_agents, mode=mode)

    return params0, emis_dict_train_JAX

def generate_eval_data(
    agents: tuple, DECK: bool = True, CS3: bool = False, DAMIP: bool = False, GeoMIP: bool = False
) -> tuple:
    """
    Notebook-facing wrapper (the most-reused function in this file - used across
    nearly every 3x/4x/SI notebook): build the named eval_sets dict ("Tier 1",
    "Tier 2", and whichever of "DECK"/"CS3"/"DAMIP"/"GeoMIP" are requested, plus
    "All") from utils_FaIR_JAX.generate_JAX_data.
    Returns (eval_sets, *eval_data, emis_dict_all_JAX) - the individual eval_data
    dicts are also returned positionally, in the same order they went into eval_sets.
    """

    eval_data = utils_FaIR_JAX.generate_JAX_data(agents, DECK=DECK, CS3=CS3, DAMIP=DAMIP, GeoMIP=GeoMIP)

    keys = ["Tier 1", "Tier 2"]
    if DECK: keys.append("DECK")
    if CS3: keys.append("CS3")
    if DAMIP: keys.append("DAMIP")
    if GeoMIP: keys.append("GeoMIP")

    # TEMPORARY!
    if len(agents) == 5 and "DECK" in keys:
        for a in agents:
            if a == 'CO2':
                continue
            eval_data[2].pop(f'abrupt-4x{a}')
            eval_data[2].pop(f'1pct{a}')

    eval_sets = dict(zip(keys, eval_data))

    emis_dict_all_JAX = {}
    for d in eval_data:
        emis_dict_all_JAX |= d

    eval_sets["All"] = emis_dict_all_JAX

    return eval_sets, *eval_data, emis_dict_all_JAX

def generate_and_eval_baseline_emulator(
    emis_dict_train: dict, eval_sets: dict, save_path: str | None = None, verbose: bool = False,
    hidden_sizes: list[int] = [16], mode: str = 'FaIR', ema_windows_years: tuple = (5.0, 30.0, 100.0),
    seed: int = 0, K: int = 400, lr: float = 5e-2, weight_decay: float = 1e-2,
) -> tuple[dict, dict, dict]:
    """
    Notebook-facing wrapper around evaluate_baseline_over_multiple_tests: train
    the baseline emulator on emis_dict_train, evaluate NRMSE on every eval_sets
    entry, optionally pickle baseline_results to save_path, optionally print a
    per-scenario summary. `seed` controls the baseline MLP's random
    initialization (default 0 reproduces the prior hardcoded behavior exactly).
    `K`/`lr`/`weight_decay` are the baseline's own training hyperparameters
    (independently tunable from the optimized emulator's - see Stage 6e,
    data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json);
    defaults reproduce the prior hardcoded values exactly.
    Returns (baseline_results, baseline_pred_delT, ground_truth_delT).
    """

    paramsK_base, baseline_results, baseline_pred_delT, ground_truth_delT = evaluate_baseline_over_multiple_tests(
    emis_dict_train=emis_dict_train,
    eval_sets=eval_sets,
    historical_name="historical",
    key=jax.random.PRNGKey(seed),
    hidden_sizes=hidden_sizes,
    K=K,
    lr=lr,
    weight_decay=weight_decay,
    mode=mode,
    ema_windows_years=ema_windows_years
)

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(baseline_results, f)

    if verbose:
        for eval_set in baseline_results:
            per_scen = baseline_results[eval_set]
            mean_err = baseline_results[eval_set]['mean']
            print(f"\n{eval_set} mean NRMSE: {mean_err:.4f}")
            for scen, val in per_scen.items():
                if scen == 'mean':
                    continue
                print(f"  {scen:30s}  {val:.4f}")

    return baseline_results, baseline_pred_delT, ground_truth_delT

def run_inverse_experiment_setup(
    agents: list[str],
    active_agents: tuple[str, ...],
    mode: str = 'FaIR',
    hidden_sizes: list[int] = [16],
    idx_demo: int = 1,
    test_scen: str = 'historical',
    verbose: bool = False,
    CS3: bool = True,
    DAMIP: bool = False,
    GeoMIP: bool = False,
    baseline_save_path: str | None = None,
    seed: int = 0,
    baseline_K: int = 400, baseline_lr: float = 5e-2, baseline_weight_decay: float = 1e-2,
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
) -> dict:
    """
    Shared setup for the 3x/4a/SI_1 inverse-optimization companion scripts and
    notebooks: build the initial baseline MLP params + training data
    (generate_init_params_and_train_data), the named eval_sets
    (generate_eval_data), and the trained+evaluated baseline emulator
    (generate_and_eval_baseline_emulator) that every inverse-optimization
    experiment in the notebook is compared against.

    `seed` controls MLP random initialization and is passed to *both*
    generate_init_params_and_train_data and generate_and_eval_baseline_emulator,
    so the optimized-emissions emulator and the baseline emulator keep starting
    from the same initial params within a given seed (the existing fairness
    property documented in PROGRESS.md) while varying together across seeds
    (used by 6a_seed_uncertainty_sweep). Default 0 reproduces prior behavior.

    `baseline_K`/`baseline_lr`/`baseline_weight_decay` are the baseline
    emulator's own, independently-tunable training hyperparameters (see
    Stage 6e); defaults reproduce the prior hardcoded values exactly.

    `ema_windows_years` is forwarded to both generate_init_params_and_train_data
    and generate_and_eval_baseline_emulator so the baseline emulator's features
    can be matched to whatever EMA convention the caller needs to compare
    against (e.g. Figure 4's optimized-emulator re-evaluation - see
    regenerate_fig4_co2_only_cache). Default reproduces the prior hardcoded
    value exactly.

    Returns a dict with keys: agents, active_agents, mode, params0,
    emis_dict_train_JAX, eval_sets, baseline_results, baseline_pred_delT,
    ground_truth_delT. Pass this dict straight into run_inverse_experiment.
    """
    params0, emis_dict_train_JAX = generate_init_params_and_train_data(
        agents, active_agents, test_scen, hidden_sizes=hidden_sizes,
        idx_demo=idx_demo, verbose=verbose, mode=mode, seed=seed,
        ema_windows_years=ema_windows_years,
    )
    eval_sets, *_ = generate_eval_data(agents, CS3=CS3, DAMIP=DAMIP, GeoMIP=GeoMIP)
    baseline_results, baseline_pred_delT, ground_truth_delT = generate_and_eval_baseline_emulator(
        eval_sets["Tier 1"], eval_sets, save_path=baseline_save_path,
        verbose=verbose, hidden_sizes=hidden_sizes, mode=mode, seed=seed,
        K=baseline_K, lr=baseline_lr, weight_decay=baseline_weight_decay,
        ema_windows_years=ema_windows_years,
    )
    return {
        "agents": agents,
        "active_agents": active_agents,
        "mode": mode,
        "params0": params0,
        "emis_dict_train_JAX": emis_dict_train_JAX,
        "eval_sets": eval_sets,
        "baseline_results": baseline_results,
        "baseline_pred_delT": baseline_pred_delT,
        "ground_truth_delT": ground_truth_delT,
    }

def build_group_emis_dicts(emis_dict_train_JAX: dict, eval_sets: dict) -> dict[str, dict]:
    """
    Build the {group_name: emis_dict} lookup used by run_inverse_experiment.
    'H-ext' is the single-scenario optimization target used by every
    notebook's "2a" section (H-ext + historical, taken from the training
    split, matching ScenarioMIP's requirement that historical always
    precede a future scenario); every other group name is a .copy() of the
    correspondingly-named eval_sets entry, only included if that eval_sets
    entry was actually built (i.e. the DAMIP/GeoMIP/CS3 flags passed to
    run_inverse_experiment_setup).
    """
    eval_set_names = {
        'tier1': 'Tier 1', 'tier2': 'Tier 2', 'DECK': 'DECK', 'CS3': 'CS3',
        'DAMIP': 'DAMIP', 'GeoMIP': 'GeoMIP', 'all': 'All',
    }
    groups = {
        'H-ext': {
            'H-ext': emis_dict_train_JAX['H-ext'].copy(),
            'historical': emis_dict_train_JAX['historical'].copy(),
        }
    }
    for group, eval_key in eval_set_names.items():
        if eval_key in eval_sets:
            groups[group] = eval_sets[eval_key].copy()
    return groups

def run_inverse_experiment(
    setup: dict,
    group: str,
    checkpoint_dir: str,
    tag: str,
    num_updates: int,
    step_size: float | dict,
    momentum: float,
    nesterov: bool,
    K_inner: int,
    lr_inner: float,
    wd_inner: float,
    init_cond: str,
    T: int,
    filter_hist: bool,
    checkpoint_every: int,
    resume_if_exists: bool,
    preds_every: int,
    smoothness_weight: float = 0.0,
    active_agents: tuple[str, ...] | None = None,
    mode: str | None = None,
    batch_size: int | None = None,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> dict:
    """
    Run one optimize_emissions_inverse experiment against `group` - either
    'H-ext' (the single-scenario target used by every notebook's "2a"
    section) or one of the named eval_sets groups built by
    run_inverse_experiment_setup ('tier1', 'tier2', 'DECK', 'CS3', 'DAMIP',
    'GeoMIP', 'all'). The checkpoint path is auto-built as
    f"{checkpoint_dir}/inverse_{init_cond}_{group}_{tag}.pkl" for every
    group, including 'H-ext' - the original 3x/4a notebooks special-cased
    'H-ext' with the segment order swapped
    ("inverse_H-ext_<init_cond>_<tag>.pkl"), which was a naming bug, not an
    intentional convention; this function and every existing 'H-ext'
    checkpoint on disk have been normalized to the uniform template.

    No hyperparameter has a default here (aside from smoothness_weight/
    active_agents/mode) - every 3x/4a notebook tunes these per group, so
    callers must pass them explicitly rather than rely on a value that might
    silently differ from what was tuned for a given experiment.
    """
    group_emis_dicts = build_group_emis_dicts(setup["emis_dict_train_JAX"], setup["eval_sets"])
    if group not in group_emis_dicts:
        raise ValueError(f"Unknown group {group!r}; available: {sorted(group_emis_dicts)}")

    if active_agents is None:
        active_agents = setup["active_agents"]
    if mode is None:
        mode = setup["mode"]

    checkpoint_path = f"{checkpoint_dir}/inverse_{init_cond}_{group}_{tag}.pkl"

    return optimize_emissions_inverse(
        emis_dict=group_emis_dicts[group],
        params0=setup["params0"],
        num_updates=num_updates,
        step_size=step_size,
        momentum=momentum,
        nesterov=nesterov,
        K_inner=K_inner,
        lr_inner=lr_inner,
        wd_inner=wd_inner,
        active_agents=active_agents,
        init_cond=init_cond,
        T=T,
        filter_hist=filter_hist,
        smoothness_weight=smoothness_weight,
        mode=mode,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        resume_if_exists=resume_if_exists,
        preds_every=preds_every,
        batch_size=batch_size,
        key=key,
    )

# ==================================================================
# Part 4: MESM zonal (vector-target) emulator - init/train/eval variants
# of the Part 3 MLP that predict a full spatial (latitude) vector instead
# of a single GMST scalar
# ==================================================================

def init_mlp_params_vector(key: jax.random.PRNGKey, input_dim: int, hidden_sizes: list[int], output_dim: int) -> list[dict]:
    """
    Initialize parameters for an MLP with a vector output.

    Args:
        output_dim: Size of the output vector (e.g., number of spatial EOFs or bins).
    """
    params = []
    # Architecture: input -> hidden ... -> hidden -> output_vector
    layer_dims = [input_dim] + hidden_sizes + [output_dim]

    keys = jax.random.split(key, len(layer_dims) - 1)

    for i in range(len(layer_dims) - 1):
        in_d, out_d = layer_dims[i], layer_dims[i+1]

        # Xavier/Glorot initialization
        lim = jnp.sqrt(6.0 / (in_d + out_d))
        W = jax.random.uniform(keys[i], (in_d, out_d), minval=-lim, maxval=lim)
        b = jnp.zeros((out_d,))

        params.append({'W': W, 'b': b})

    return params

def mlp_forward_vector(params: list[dict], X: jnp.ndarray) -> jnp.ndarray:
    """
    Forward pass for a vector-output MLP.

    Returns:
        Array of shape (N, output_dim). No squeezing is applied.
    """
    X = X.astype(params[0]["W"].dtype)

    # Flatten input to (N, D) if it comes in as (T, D) or similar
    N = X.shape[0]
    activations = X.reshape(N, -1)

    # Hidden layers (tanh activation)
    for layer in params[:-1]:
        linear = activations @ layer['W'] + layer['b']
        activations = jnp.tanh(linear)

    # Final Output Layer (Linear)
    final_layer = params[-1]
    y = activations @ final_layer['W'] + final_layer['b']

    return y

def train_mlp_sgd_vector(
    params0: list[dict], Xtr: jnp.ndarray, ytr: jnp.ndarray, weights: jnp.ndarray | None = None,
    K: int = 400, lr: float = 5e-2, weight_decay: float = 1e-2
) -> tuple[list[dict], jnp.ndarray]:
    """
    Training loop specifically for vector outputs (Xtr, ytr).
    ytr shape should be (N_samples, output_dim).
    """
    # Ensure types match
    pdt = params0[0]["W"].dtype
    Xtr = Xtr.astype(pdt)
    ytr = ytr.astype(pdt)

    if weights is not None:
        weights = weights.astype(pdt)
        weights = weights.reshape(1, -1)
    else:
        weights = 1.0

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.add_decayed_weights(weight_decay),
        optax.sgd(lr)
    )
    opt_state = optimizer.init(params0)

    @jax.checkpoint
    def step(carry, _):
        params, opt_state = carry

        def loss_fn(p):
            yhat = mlp_forward_vector(p, Xtr)
            sq_err = weights * (yhat - ytr)**2
            return jnp.mean(sq_err)

        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Safety for gradients
        grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6), grads)

        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        return (params, opt_state), loss

    (paramsK, _), losses = jax.lax.scan(step, (params0, opt_state), xs=None, length=K)
    return paramsK, losses

def build_dataset_vector_targets(
    emis_dict: dict,
    targets_dict: dict,
    scenarios: list[str],
    historical_name: str = "historical",
    agents: tuple = AGENTS_DEFAULT,
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
    target_crop_future: int = 162,
    emis_offset_hist: int = 111
) -> list[tuple[jnp.ndarray, jnp.ndarray, str]]:
    """
    Constructs a dataset where targets are retrieved from `targets_dict` rather
    than being simulated internally.

    Args:
        emis_dict: Dictionary of emissions scenarios (inputs).
        targets_dict: Dictionary of target arrays {scenario: (T, output_dim)}.
        scenarios: List of scenario names to include.

    Returns:
        List of (X_feature, y_target, scenario_name) tuples.
    """
    # Extract historical emissions for feature construction context (full from 1750)
    years_hist, emis_hist_dict = (None, None)
    if historical_name in emis_dict:
        years_hist, emis_hist_dict = extract_years_and_emis(
            emis_dict[historical_name], agents=agents
        )

    dataset = []

    for scen in scenarios:
        # Skip if scenario is missing from either inputs or targets
        if scen not in emis_dict or scen not in targets_dict:
            continue

        y_target = jnp.asarray(targets_dict[scen], dtype=jnp.float32)

        if scen != historical_name and historical_name in emis_dict:
            # For future scenarios, remove the prepended historical period
            if y_target.shape[0] > target_crop_future:
                y_target = y_target[target_crop_future:]
            else:
                print(f"Warning: Target for {scen} is shorter than crop length {target_crop_future}.")


        # 2. Get Input Features (Emissions)
        yrs_cur, emis_cur_dict = extract_years_and_emis(
            emis_dict[scen], agents=agents
        )

        # Handle historical context for features
        needs_history = (
            (scen != historical_name) and
            (years_hist is not None) and
            (scen in scens_with_hist)
        )

        cur_hist_emis = emis_hist_dict
        if needs_history:
             _, cur_hist_emis = CS3_hist_modifier(scen, years_hist, emis_hist_dict)

        # Build Features (same as baseline)
        X = make_features_emissions_generic(
            emis_curr_dict=emis_cur_dict,
            emis_hist_dict=(cur_hist_emis if needs_history else None),
            agents=agents,
            ema_windows_years=ema_windows_years,
            dt_years=1.0,
            zero_fill_missing=True,
        )

        if scen == historical_name:
            # Historical Emissions start 1750, Targets start 1861.
            if X.shape[0] > emis_offset_hist:
                X = X[emis_offset_hist:]

        # 3. Align Lengths
        # If feature generation and target array differ slightly in length, trim to min.
        N = min(X.shape[0], y_target.shape[0])
        dataset.append((X[:N], y_target[:N], scen))

    return dataset

def prepare_data_vector(
    emis_dict_train: dict,
    targets_dict_train: dict,
    scenarios_train: list[str],
    historical_name: str = "historical",
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
    target_crop_future: int = 162,
    emis_offset_hist: int = 111,
    precomp_stats_X: tuple[jnp.ndarray, jnp.ndarray] | None = None
) -> tuple[list, tuple[jnp.ndarray, jnp.ndarray]]:
    """Builds and scales the training data for vector targets."""
    # Build raw
    train_raw = build_dataset_vector_targets(
        emis_dict_train, targets_dict_train, scenarios_train,
        historical_name=historical_name, ema_windows_years=ema_windows_years,
        target_crop_future=target_crop_future, emis_offset_hist=emis_offset_hist
    )

    if not train_raw:
        raise ValueError("Train dataset empty. Check scenario keys.")

    # Stack to fit scalers
    Xtr_all = jnp.concatenate([d[0] for d in train_raw], axis=0)

    if precomp_stats_X is not None:
        # Use provided baseline stats
        stats_X = precomp_stats_X
        Xtr_scaled_all = apply_scaler(Xtr_all, stats_X)
    else:
        # Fit new stats
        Xtr_scaled_all, stats_X = fit_scaler(Xtr_all)

    # Reshape back to list
    train_scaled = []
    idx = 0
    for (X, y, scen) in train_raw:
        n = X.shape[0]
        train_scaled.append((
            Xtr_scaled_all[idx : idx + n],
            jnp.asarray(y, dtype=jnp.float32),
            scen
        ))
        idx += n

    return train_scaled, stats_X

def evaluate_emulator_vector_over_multiple_tests(
    params: list[dict],
    stats_X: tuple[jnp.ndarray, jnp.ndarray],
    eval_emis_sets: dict,
    eval_targets_sets: dict,
    lat_coords: np.ndarray | None = None,
    historical_name: str = "historical",
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
    target_crop_future: int = 162,
    emis_offset_hist: int = 111
) -> tuple[dict, dict, dict]:
    """
    Evaluates vector emulator with both Zonal (per-dim) and Global (weighted) NRMSE.
    Returns (results, predictions, ground_truth), each keyed by eval-set name then
    scenario ('mean' also included per eval-set with weighted-average zonal/global NRMSE).
    """

    results = {}
    predictions = {}
    ground_truth = {}

    # --- 1. Setup Latitude Weights ---
    # Expecting predictions to be shape (T, N_lat).
    # Weights should be (N_lat,)
    if lat_coords is not None:
        # Convert to radians and take cosine
        # We enforce a tiny epsilon floor to prevent division by zero if exactly 90 deg is passed
        weights = np.cos(np.deg2rad(lat_coords))
        weights = np.maximum(weights, 1e-6)
        weights = weights / np.sum(weights) # Normalize so they sum to 1
    else:
        # If no lats provided, uniform weighting for global metric
        # We can't know dimension yet, will init inside loop or assume uniform
        weights = None

    def apply_stats_to_test_vector(raw_list, sX):
        out = []
        for (X, y, scen) in raw_list:
            Xs = apply_scaler(X, sX)
            out.append((Xs, y, scen))
        return out

    for set_name, emis_d in eval_emis_sets.items():
        targets_d = eval_targets_sets.get(set_name)
        if targets_d is None: continue

        # Build raw test data
        scens = list(emis_d.keys())
        raw_test = build_dataset_vector_targets(
            emis_d, targets_d, scens,
            historical_name=historical_name, ema_windows_years=ema_windows_years,
            target_crop_future=target_crop_future, emis_offset_hist=emis_offset_hist
        )

        scaled_test = apply_stats_to_test_vector(raw_test, stats_X)

        # Storage for this evaluation set
        res_set = {}
        pred_set, truth_set = {}, {}

        # Lists to aggregate means across scenarios
        errs_global_list, errs_zonal_list = [], []
        lens = []

        for (Xs, ytrue, scen) in scaled_test:
            if set_name not in ['Tier 1', 'All'] and scen == historical_name:
                continue

            # Predict
            yhat = mlp_forward_vector(params, Xs)

            # Store predictions
            pred_set[scen] = yhat
            truth_set[scen] = ytrue

            # --- METRIC CALCULATION ---

            # Init weights if not done (uniform fallback)
            dims = ytrue.shape[1]
            if weights is None:
                current_weights = np.ones(dims) / dims
            else:
                if len(weights) != dims:
                    raise ValueError(f"Lat coords length {len(weights)} != Output dim {dims}")
                current_weights = weights

            # 1. Zonal NRMSE (Vector: one value per latitude band)
            # RMSE per column
            mse_zonal = np.mean((yhat - ytrue)**2, axis=0)
            rmse_zonal = np.sqrt(mse_zonal)
            # Normalize by range of truth per column
            max_abs_zonal = np.max(np.abs(ytrue), axis=0)
            nrmse_zonal = rmse_zonal / (max_abs_zonal + 1e-8)

            # 2. Global NRMSE (Scalar: weighted average over space)
            # Weighted MSE at each timestep, then averaged over time
            # (T, D) -> (T,) -> Scalar
            diff_sq = (yhat - ytrue)**2
            weighted_diff_sq = diff_sq * current_weights[None, :] # Broadcast weights
            mse_global_t = np.sum(weighted_diff_sq, axis=1) # Sum weighted errors over space
            max_abs_global = np.max(np.abs(ytrue))
            rmse_global_t = np.sqrt(mse_global_t)            # Shape (T,)
            nrmse_global_t = rmse_global_t / (max_abs_global + 1e-8)
            rmse_global = np.sqrt(np.mean(mse_global_t))    # Mean over time

            # Normalize by global max abs
            max_abs_global = np.max(np.abs(ytrue))
            nrmse_global = rmse_global / (max_abs_global + 1e-8)

            # Store per-scenario results
            res_set[scen] = {
                'global': float(nrmse_global),
                'zonal': nrmse_zonal,
                'global_t': nrmse_global_t
            }

            errs_global_list.append(nrmse_global)
            errs_zonal_list.append(nrmse_zonal)
            lens.append(ytrue.shape[0])

        # Calculate Means across scenarios
        if lens:
            mean_global = np.average(errs_global_list, weights=lens)
            # Average zonal vectors (stack them first)
            mean_zonal = np.average(np.stack(errs_zonal_list), axis=0, weights=lens)

            res_set['mean'] = {
                'global': float(mean_global),
                'zonal': mean_zonal
            }

        results[set_name] = res_set
        predictions[set_name] = pred_set
        ground_truth[set_name] = truth_set

    return results, predictions, ground_truth

def generate_and_eval_emulator_vector(
    emis_dict_train: dict,
    targets_dict_train: dict,
    eval_emis_sets: dict,
    eval_targets_sets: dict,
    output_dim: int,
    lat_coords: np.ndarray | None = None,
    hidden_sizes: list[int] = [16],
    lr: float = 5e-2,
    weight_decay: float = 1e-2,
    K: int = 400,
    save_path: str | None = None,
    verbose: bool = False,
    key_seed: int = 0,
    historical_name: str = "historical",
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
    precomp_stats_X: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    params0: list[dict] | None = None
) -> tuple:
    """
    Notebook-facing wrapper (used by 4c_evaluate_MESM_emulator.ipynb): build vector-
    target train data (prepare_data_vector), train a vector-output MLP
    (train_mlp_sgd_vector), evaluate it (evaluate_emulator_vector_over_multiple_tests),
    and optionally pickle the results to save_path. Note: test_vector_emulator.py
    calls this function under the name `generate_and_eval_baseline_emulator_vector`,
    which doesn't exist - see Phase 4 test-suite notes.
    Returns (results, preds, truths, paramsK[, stats_X if precomp_stats_X was None]).
    """

    # 1. Prepare Training Data
    train_scens = list(emis_dict_train.keys())
    train_scaled, stats_X = prepare_data_vector(
        emis_dict_train, targets_dict_train, train_scens,
        historical_name=historical_name, ema_windows_years=ema_windows_years,
        precomp_stats_X=precomp_stats_X
    )

    Xtr = jnp.concatenate([d[0] for d in train_scaled], axis=0)
    ytr = jnp.concatenate([d[1] for d in train_scaled], axis=0)

    weights = None
    if lat_coords is not None:
        w = jnp.cos(jnp.deg2rad(lat_coords))
        weights = w / jnp.mean(w)

    # 2. Train
    if params0 is None:
        key = jax.random.PRNGKey(key_seed)
        params0 = init_mlp_params_vector(key, Xtr.shape[1], hidden_sizes, output_dim)

    paramsK, losses = jax.jit(train_mlp_sgd_vector, static_argnames=("K",))(
        params0, Xtr, ytr, weights=weights, K=K, lr=lr, weight_decay=weight_decay
    )

    # 3. Evaluate (Pass lat_coords here)
    results, preds, truths = evaluate_emulator_vector_over_multiple_tests(
        paramsK, stats_X,
        eval_emis_sets, eval_targets_sets,
        lat_coords=lat_coords,
        historical_name=historical_name,
        ema_windows_years=ema_windows_years
    )

    if save_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        save_dict = {
            "results": results,
            "preds": preds,
            "truth": truths,
            "params": paramsK,
            "stats": stats_X
        }

        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)

        if verbose:
            print(f"Results saved to {save_path}")

    if verbose:
        for eval_set in results:
            if 'mean' in results[eval_set]:
                val = results[eval_set]['mean']['global']
                print(f"\n{eval_set} Mean Global NRMSE: {val:.4f}")

    if precomp_stats_X is None:
        return results, preds, truths, paramsK, stats_X
    return results, preds, truths, paramsK

def generate_target_data(scenarios_dict: dict, data_dir: str = "./", opt: bool = False) -> tuple:
    """
    Loads pickled zonal temperature data for scenarios defined in scenarios_dict.

    Args:
        scenarios_dict (dict): Keys are set names (e.g., 'Tier 1'), values are lists of scenario names.
        data_dir (str): Directory where the .pkl files are stored.

    Returns:
        dict: A dictionary with the same keys as scenarios_dict, where values are
              dictionaries mapping {scenario_name: target_array}.
    """
    target_sets, all_targets = {}, {}

    for label, scenario_list in scenarios_dict.items():
        set_targets = {}
        for scen in scenario_list:
            if opt:
                file_path = os.path.join(data_dir, f"{label}/opt_{scen}_mean.pkl")
            else:
                file_path = os.path.join(data_dir, f"{label}/{scen}_mean.pkl")

            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    set_targets[scen] = data
            else:
                print(f"Warning: File not found for scenario '{scen}' at {file_path}")

        target_sets[label] = set_targets
        all_targets.update(set_targets)

    for key1 in target_sets:
        for key2 in target_sets[key1]:
            output_dim = target_sets[key1][key2].shape[1]
            break
        break

    lat0, latf = -88, 88
    lat_coords = np.linspace(lat0, latf, output_dim)

    individual_dicts = [target_sets[k] for k in scenarios_dict]
    return target_sets, *individual_dicts, output_dim, lat_coords
# ==================================================================
# Part 5: notebook-facing data-prep for 5a_paper_plots.ipynb figures
# Each function loads/aggregates exactly what one utils_plotting figure call
# needs, so 5a_paper_plots.ipynb itself only ever does "load data" then
# "plot data" - no data construction happens in the notebook directly.
# ==================================================================

def load_fig1_tier1_scenarios(agents: list[str] = ['CO2']) -> dict:
    """
    Tier-1 calibration emissions for Figure 1's tier1-scenarios panel
    (utils_plotting.plot_tier1). Returns {'years', 'tier1', 'group'}, one
    entry per Tier-1 scenario.
    """
    _, emis_dict_calib_JAX, _, _ = utils_FaIR_JAX.generate_calib_data(agents)
    group = ['historical', 'H-ext', 'M', 'ML', 'L', 'VLHO', 'VLLO-ext']
    years = [np.arange(1750, 2024), np.arange(2024, 2501), np.arange(2024, 2151),
              np.arange(2024, 2151), np.arange(2024, 2151), np.arange(2024, 2151),
              np.arange(2024, 2501)]
    tier1 = [emis_dict_calib_JAX[scen][0] for scen in group]
    return {"years": years, "tier1": tier1, "group": group}


def load_fig2_data(
    agents: list[str] = ['CO2'],
    path: str = 'checkpoints/co2/inverse_constant_H-ext_co2_only.pkl',
) -> dict:
    """
    Optimal CO2 emissions trajectory vs. its H-ext target (Figure 2) for
    utils_plotting.plot_stacked_results / plot_stacked_results_ppt.
    Returns {'target_emissions', 'years', 'opt_emissions', 'opt_temp', 'opt_results'}.
    """
    _, emis_dict_calib_JAX, _, _ = utils_FaIR_JAX.generate_calib_data(agents)
    opt_results = load_inverse_ckpt(path)

    opt_emissions = [series['CO2'] for series in opt_results['U_traj']]
    opt_temp = [series[-1] for series in opt_results['train_temp_traj']]
    target_emissions = [emis_dict_calib_JAX['H-ext'][0].copy()]
    years = np.arange(2024, 2501)

    return {
        "target_emissions": target_emissions,
        "years": years,
        "opt_emissions": opt_emissions,
        "opt_temp": opt_temp,
        "opt_results": opt_results,
    }


def load_fig3_single_forcing_data(agents: list[str] = ['co2', 'ch4', 'n2o', 'Sulfur', 'BC']) -> dict:
    """
    Per-agent single-forcing inverse vs. baseline NRMSE (Figure 3) for
    utils_plotting.plot_rmse_comparison_single.
    Returns {'results_inverse', 'results_baseline', 'labels'}.
    """
    results_inverse, results_baseline = [], []
    for a in agents:
        path_inverse = f'checkpoints/{a}/inverse_constant_tier1_{a}_only.pkl'
        path_baseline = f'checkpoints/{a}/baseline_{a}_only.pkl'
        results_inverse.append(load_inverse_ckpt(path_inverse))
        with open(path_baseline, "rb") as f:
            results_baseline.append(pickle.load(f)['Tier 1']['mean'])

    labels = ['(a) CO$_2$', '(b) CH$_4$', '(c) N$_2$O', '(d) Sulfur', '(e) BC']
    return {"results_inverse": results_inverse, "results_baseline": results_baseline, "labels": labels}

def regenerate_fig4_all_agents_cache(
    save_path: str = 'data/plotting/optimal_all_agents_subset.pkl',
) -> dict:
    """
    Recompute the all-agents optimal-emulator NRMSE summary used by the
    Figure-4 performance-summary bar chart and overwrite `save_path`'s cache.

    Not called by default from the notebook - load_fig5_data() reads the
    existing cache instead. Call this only to refresh
    data/plotting/optimal_all_agents_subset.pkl after new inverse-optimization
    checkpoints are produced upstream (Phase 3 scripts).
    """
    agents = ['CO2', 'CH4', 'N2O', 'Sulfur', 'BC']
    active_agents = ('CO2', 'CH4', 'N2O', 'Sulfur', 'BC')

    params0, _ = generate_init_params_and_train_data(
        agents, active_agents, test_scen='historical', hidden_sizes=[16], idx_demo=1, verbose=False
    )
    eval_sets, *_ = generate_eval_data(agents, CS3=True, DAMIP=False, GeoMIP=False)

    training_paths = [
        'checkpoints/multi/inverse_constant_tier1_all_agents_subset2.pkl',
        'checkpoints/multi/inverse_constant_tier2_all_agents_subset.pkl',
        'checkpoints/multi/inverse_constant_DECK_all_agents_subset.pkl',
        'checkpoints/multi/inverse_constant_CS3_all_agents_subset.pkl',
        'checkpoints/multi/inverse_constant_all_all_agents_subset.pkl',
    ]
    train_scenarios = ['Opt. Tier 1', 'Opt. Tier 2', 'Opt. DECK', 'Opt. CS3', 'Opt. All']
    optimal_results_all = evaluate_optimal_emulator(
        training_paths=training_paths,
        train_scenarios=train_scenarios,
        eval_sets=eval_sets,
        params0=params0,
        active_agents=active_agents,
        inactive_mode="zeros",
        historical_name="historical",
        key=jax.random.PRNGKey(0),
        K=400,
        lr=5e-2,
        weight_decay=1e-2,
    )

    with open(save_path, "wb") as f:
        pickle.dump(optimal_results_all, f)

    return optimal_results_all

def regenerate_fig4_co2_only_cache(
    checkpoint_dir: str = 'checkpoints/co2_retuned',
    baseline_save_path: str = 'data/plotting/baseline_co2_only_retuned.pkl',
    optimal_save_path: str = 'data/plotting/optimal_co2_only_retuned.pkl',
    K_inner: float = 400, lr_inner: float = 5e-2, wd_inner: float = 1e-2,
    baseline_K: int = 400, baseline_lr: float = 5e-2, baseline_weight_decay: float = 1e-2,
    seed: int = 0,
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
) -> tuple[dict, dict]:
    """
    Recompute the CO2-only baseline/optimal-emulator NRMSE summaries used by
    the Figure-4 performance-summary bar chart's "(a) CO2-only" panel, and
    write both to save_path (does NOT overwrite the original
    data/plotting/{baseline,optimal}_co2_only.pkl - use distinct paths so old
    and new results stay independently available, matching Stage 0c's
    checkpoints/co2_retuned/ convention).

    `K_inner`/`lr_inner`/`wd_inner` are the *inner-loop* hyperparameters used
    to train the fresh evaluation-time emulator on each checkpoint's final
    optimized emissions (evaluate_optimal_emulator's own K/lr/weight_decay) -
    these should match whatever inner-loop hyperparameters actually produced
    the checkpoints in `checkpoint_dir` (Stage 0b's best_config_unified.json
    for checkpoints/co2_retuned/), NOT the baseline's own hyperparameters.
    `baseline_K`/`baseline_lr`/`baseline_weight_decay` are the baseline
    emulator's own, separately-tunable hyperparameters (Stage 6e's
    best_baseline_config_K400.json for the retuned baseline).

    `ema_windows_years` is forwarded to both the baseline's setup (via
    run_inverse_experiment_setup) and evaluate_optimal_emulator's fresh
    retrain/eval, so both sides of the comparison use one consistent EMA
    convention. IMPORTANT (temporary, pending a real fix): the checkpoints
    in `checkpoint_dir` were themselves produced by optimize_emissions_inverse,
    which trains on ema_windows_years=(5,30,50) (its own default) but - due to
    a live bug - always *evaluates* its internal test/error signal on
    build_valid's hardcoded (5,30,100), regardless of what's passed to it. So
    (5,30,100) here (this function's own long-standing default) does NOT
    match what the checkpoints were actually optimized against, and produces
    an artificially inflated (much worse-looking) NRMSE for the optimized
    emulator. Passing ema_windows_years=(5,30,50) instead reproduces the
    Stage 0b/6e "gap survives" numbers (best_config_unified.json /
    REVISIONS.md's gap-persistence tables) almost exactly - confirmed
    directly, e.g. CO2-only Tier 1: 0.175 at (5,30,100) vs. 0.0465 at
    (5,30,50), vs. 0.0420-0.0455 from Stage 0b/6e's own internal scoring.
    Use (5,30,50) here until optimize_emissions_inverse's build_valid call is
    fixed to actually respect its own ema_windows_years parameter (then this
    default should be revisited too).

    Call this only to refresh the cache after new CO2-only inverse-
    optimization checkpoints are produced upstream (Stage 0c).
    """
    setup = run_inverse_experiment_setup(
        ['CO2'], ('CO2',), mode='FaIR', CS3=True, DAMIP=False, GeoMIP=False,
        idx_demo=None, seed=seed,
        baseline_K=baseline_K, baseline_lr=baseline_lr, baseline_weight_decay=baseline_weight_decay,
        ema_windows_years=ema_windows_years,
    )

    training_paths = [
        f'{checkpoint_dir}/inverse_constant_tier1_co2_only.pkl',
        f'{checkpoint_dir}/inverse_constant_tier2_co2_only.pkl',
        f'{checkpoint_dir}/inverse_constant_DECK_co2_only.pkl',
        f'{checkpoint_dir}/inverse_constant_CS3_co2_only.pkl',
        f'{checkpoint_dir}/inverse_constant_all_co2_only.pkl',
    ]
    train_scenarios = ['Opt. Tier 1', 'Opt. Tier 2', 'Opt. DECK', 'Opt. CS3', 'Opt. All']
    optimal_results_co2 = evaluate_optimal_emulator(
        training_paths=training_paths,
        train_scenarios=train_scenarios,
        eval_sets=setup["eval_sets"],
        params0=setup["params0"],
        active_agents=('CO2',),
        inactive_mode="zeros",
        historical_name="historical",
        key=jax.random.PRNGKey(seed),
        K=K_inner,
        lr=lr_inner,
        weight_decay=wd_inner,
        mode='FaIR',
        ema_windows_years=ema_windows_years,
    )

    with open(baseline_save_path, "wb") as f:
        pickle.dump(setup["baseline_results"], f)
    with open(optimal_save_path, "wb") as f:
        pickle.dump(optimal_results_co2, f)

    return setup["baseline_results"], optimal_results_co2

def regenerate_fig4_co2_only_cache_seed_sweep(
    seeds: list[int] = tuple(range(50)),
    checkpoint_dir: str = 'checkpoints/co2_retuned/seed_sweep',
    tag: str = 'co2_only',
    unified_config_path: str = 'data/SI_results/hp_retune/best_config_unified.json',
    baseline_config_path: str = 'data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json',
    out_path: str = 'data/SI_results/seed_uncertainty/fig4_seed_spread_co2_only.pkl',
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
) -> dict[int, dict]:
    """
    Multi-seed companion to regenerate_fig4_co2_only_cache: for each seed,
    rebuilds that seed's own (baseline_results, optimal_results) pair - the
    baseline via run_inverse_experiment_setup(seed=seed, ...) (the Stage 0
    "fairness property" that seeds params0 and the baseline emulator
    together, so the baseline varies across seeds exactly like the optimized
    emulator does, no separate baseline-sweep code needed), the optimal
    emulator via evaluate_optimal_emulator against that seed's own
    checkpoints_dir/inverse_constant_{group}_{tag}_seed{seed}.pkl checkpoints
    (written by scripts/0c_regenerate_checkpoints_co2.py's multi-seed mode).

    K_inner/lr_inner/wd_inner and baseline_K/lr/weight_decay are read from
    Stage 0b/6e's tuned config JSONs by default - the SAME configs
    0c_regenerate_checkpoints_co2.py used to produce the checkpoints being
    evaluated here (unlike scripts/6a_seed_uncertainty_sweep.py's older
    compute_fig4_eval_spread, which still reads 3a_inverse_CO2_only.py's
    stale, never-retuned per-group EXPERIMENTS dict - that function is left
    untouched/frozen since REVISIONS.md already marks it superseded).

    Unlike that older prototype (which reused ONE fixed shared baseline for
    every seed), this caches BOTH baseline and optimal per seed -
    {seed: {"baseline": ..., "optimal": ...}} - so the plotting layer can
    compute pct-improvement using each seed's own baseline, not a shared one.

    H-ext is excluded (Figure 2's single-scenario example, not part of
    Figure 4). Call this only after the multi-seed checkpoints exist on disk
    (Stage 0c's cluster rerun); raises FileNotFoundError otherwise.
    """
    unified_cfg = json.load(open(unified_config_path))["config"]
    baseline_cfg = json.load(open(baseline_config_path))["config"]

    groups = ['tier1', 'tier2', 'DECK', 'CS3', 'all']
    train_scenarios = ['Opt. Tier 1', 'Opt. Tier 2', 'Opt. DECK', 'Opt. CS3', 'Opt. All']

    all_results = {}
    for seed in seeds:
        setup = run_inverse_experiment_setup(
            ['CO2'], ('CO2',), mode='FaIR', CS3=True, DAMIP=False, GeoMIP=False,
            idx_demo=None, seed=seed,
            baseline_K=baseline_cfg["K"], baseline_lr=baseline_cfg["lr"],
            baseline_weight_decay=baseline_cfg["weight_decay"],
            ema_windows_years=ema_windows_years,
        )

        training_paths = [f'{checkpoint_dir}/inverse_constant_{g}_{tag}_seed{seed}.pkl' for g in groups]
        for p in training_paths:
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"{p} missing - run scripts/0c_regenerate_checkpoints_co2.py --seed {seed} first"
                )

        optimal_results = evaluate_optimal_emulator(
            training_paths=training_paths,
            train_scenarios=train_scenarios,
            eval_sets=setup["eval_sets"],
            params0=setup["params0"],
            active_agents=('CO2',),
            inactive_mode="zeros",
            historical_name="historical",
            key=jax.random.PRNGKey(seed),
            K=unified_cfg["K_inner"],
            lr=unified_cfg["lr_inner"],
            weight_decay=unified_cfg["wd_inner"],
            mode='FaIR',
            ema_windows_years=ema_windows_years,
        )

        all_results[seed] = {"baseline": setup["baseline_results"], "optimal": optimal_results}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)

    return all_results

def load_fig4_data(
    baseline_path_co2: str = 'data/plotting/baseline_co2_only.pkl',
    optimal_path_co2: str = 'data/plotting/optimal_co2_only.pkl',
    baseline_path_all: str = 'data/plotting/baseline_all_agents_subset.pkl',
    optimal_path_all: str = 'data/plotting/optimal_all_agents_subset.pkl',
) -> dict:
    """
    Load the cached baseline/optimal-emulator NRMSE summaries (Figure 4:
    performance summary across optimization priorities) for
    utils_plotting.plot_vertical_stacked_bars.
    """
    def _load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    return {
        "baseline_results_list": [_load(baseline_path_co2), _load(baseline_path_all)],
        "optimized_results_list": [_load(optimal_path_co2), _load(optimal_path_all)],
        "train_scenarios": ['Opt. Tier 1', 'Opt. Tier 2', 'Opt. DECK', 'Opt. CS3', 'Opt. All'],
        "test_scenarios": ['Tier 1', 'Tier 2', 'DECK', 'CS3'],
        "x_labels": ['Opt. Priority 1', 'Opt. Priority 2', 'Opt. DECK', 'Opt. CS3', 'Opt. All'],
        "leg_labels": ['Priority 1', 'Priority 2', 'DECK', 'CS3'],
        "weights": [7, 5, 2, 2],
        "figname": 'performance_summary',
    }

def load_fig5_multi_forcing_data(
    path_inverse: str = 'checkpoints/multi/inverse_constant_tier1_all_agents_subset2.pkl',
    path_baseline: str = 'checkpoints/multi/baseline_all_agents_subset.pkl',
) -> dict:
    """
    Multi-agent inverse vs. baseline NRMSE (Figure 5) for
    utils_plotting.plot_rmse_comparison_multi. Returns {'results', 'baseline_error'}.
    """
    results = load_inverse_ckpt(path_inverse)
    with open(path_baseline, "rb") as f:
        baseline_error = pickle.load(f)['Tier 1']['mean']
    return {"results": results, "baseline_error": baseline_error}

def regenerate_fig6_individual_effects_cache(save_dir: str = 'data/plotting') -> dict:
    """
    Recompute the per-agent baseline/optimal-emulator predictions used by the
    individual-effects figure (M-GHG / M-aer / G6sulfur scenarios) and
    overwrite their caches under `save_dir`.

    Not called by default from the notebook - load_fig6_data() reads the
    existing cache instead. Call this only to refresh the cache after new
    inverse-optimization checkpoints are produced upstream (Phase 3 scripts).
    """
    agents = ['CO2', 'CH4', 'N2O', 'Sulfur', 'BC']
    active_agents = ('CO2', 'CH4', 'N2O', 'Sulfur', 'BC')

    params0, _ = generate_init_params_and_train_data(
        agents, active_agents, test_scen='historical', hidden_sizes=[16], idx_demo=1, verbose=False
    )
    eval_sets_ind_effects, *_ = generate_eval_data(agents, CS3=True, DAMIP=True, GeoMIP=True)

    _, y_hat_baseline, y_true_ind_effects = generate_and_eval_baseline_emulator(
        eval_sets_ind_effects["Tier 1"], eval_sets_ind_effects, save_path=None,
        verbose=False, hidden_sizes=[16]
    )

    training_paths_ind_effects = [
        'checkpoints/multi/inverse_constant_tier1_all_agents_subset2.pkl',
        'checkpoints/multi/inverse_constant_DAMIP_all_agents.pkl',
        'checkpoints/multi/inverse_constant_GeoMIP_all_agents.pkl',
        'checkpoints/multi/inverse_constant_all_all_agents.pkl',
    ]
    train_scenarios_ind_effects = ['Opt. Tier 1', 'Opt. DAMIP', 'Opt. GeoMIP', 'Opt. All']
    _, y_hat_ind_effects = evaluate_optimal_emulator(
        training_paths=training_paths_ind_effects,
        train_scenarios=train_scenarios_ind_effects,
        eval_sets=eval_sets_ind_effects,
        params0=params0,
        active_agents=active_agents,
        inactive_mode="zeros",
        historical_name="historical",
        key=jax.random.PRNGKey(0),
        K=400,
        lr=5e-2,
        weight_decay=1e-2,
        ind_effects=True,
    )

    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/y_hat_baseline_ind_effects.pkl', "wb") as f:
        pickle.dump(y_hat_baseline, f)
    with open(f'{save_dir}/y_true_ind_effects.pkl', "wb") as f:
        pickle.dump(y_true_ind_effects, f)
    with open(f'{save_dir}/y_hat_ind_effects.pkl', "wb") as f:
        pickle.dump(y_hat_ind_effects, f)

    return {
        "y_true_ind_effects": y_true_ind_effects,
        "y_hat_baseline": y_hat_baseline,
        "y_hat_ind_effects": y_hat_ind_effects,
        "train_scenarios_ind_effects": train_scenarios_ind_effects,
    }


def load_fig6_data(save_dir: str = 'data/plotting') -> dict:
    """
    Load the cached per-agent baseline/optimal-emulator predictions for the
    individual-effects figure (utils_plotting.plot_individual_effects_summary).
    """
    def _load(name):
        with open(f'{save_dir}/{name}.pkl', 'rb') as f:
            return pickle.load(f)

    return {
        "y_true_ind_effects": _load('y_true_ind_effects'),
        "y_hat_baseline": _load('y_hat_baseline_ind_effects'),
        "y_hat_ind_effects": _load('y_hat_ind_effects'),
        "train_scenarios_ind_effects": ['Opt. Tier 1', 'Opt. DAMIP', 'Opt. GeoMIP', 'Opt. All'],
    }


def load_fig7_emic_data() -> dict:
    """
    Load MESM (EMIC) zonal-validation summary data for the Figure-7 emic
    summary bar chart (utils_plotting.plot_scenario_difference_bars2):
    baseline/optimal MESM NRMSE summaries, the optimized CO2 trajectories
    used to drive MESM, and the MESM ensemble-mean global temperature
    response for the optimized-emissions runs.

    Uses parallel=False in open_mfdataset (Phase 5 fix): this call used to
    use parallel=True and was flagged as an "intermittent" NetCDF4/dask
    failure, but scripts/4b_process_MESM_data.py's identical call turned out
    to fail deterministically with parallel=True in practice - same
    underlying dask-worker/netCDF4-file-handle issue, just not actually
    intermittent. parallel=False only changes read concurrency, not values.
    """
    with open('data/plotting/baseline_co2_only_MESM.pkl', 'rb') as f:
        baseline_MESM = pickle.load(f)

    opt_MESM = []
    for IC in ['constant', 'sine', 'both']:
        with open(f'data/plotting/optimal_co2_only_MESM_{IC}.pkl', 'rb') as f:
            opt_MESM.append(pickle.load(f))

    co2_data = []
    for IC in ['constant', 'sine']:
        with open(f'checkpoints/co2/inverse_{IC}_all_co2_only_MESM.pkl', 'rb') as f:
            res = pickle.load(f)
        co2_data.append(res['U_traj'][-1]['CO2'])

    lat_coords = np.linspace(-88, 88, 46)
    lat_weights = np.cos(np.deg2rad(lat_coords))
    lat_weights = np.maximum(lat_weights, 1e-6)
    lat_weights = lat_weights / np.sum(lat_weights)

    global_mean_temp = []
    for scen in ['opt_all', 'opt_all_sine']:
        path = f'data/MESM/emis_driven/zonal_data/optimized/ZONALANN.{scen}*.nc'
        ds = xr.open_mfdataset(path, combine='nested', concat_dim='member', parallel=False, coords='minimal')
        ensemble_mean = ds['DT2M'].mean(dim='member').compute().values
        global_mean_temp.append(np.average(ensemble_mean, weights=lat_weights, axis=1))

    scenario_keys = ['historical', 'H-ext', 'M', 'ML', 'L', 'VLLO-ext', 'VLHO',
                      'H-ext-OS', 'M-ext', 'ML-ext', 'L-ext', 'VLHO-ext',
                      '2xCO2', '1pctCO2', 'AA', 'CT']
    x_labels = [r'$\it{historical}$', r'$\it{H}$-$\it{ext}$', r'$\it{M}$', r'$\it{ML}$', r'$\it{L}$',
                 r'$\it{VLLO}$-$\it{ext}$', r'$\it{VLHO}$', r'$\it{H}$-$\it{ext}$-$\it{OS}$',
                 r'$\it{M}$-$\it{ext}$', r'$\it{ML}$-$\it{ext}$', r'$\it{L}$-$\it{ext}$',
                 r'$\it{VLHO}$-$\it{ext}$', r'$\it{abrupt}$-$\it{2xCO2}$', r'$\it{1pctCO2}$',
                 r'$\it{AA}$', r'$\it{CT}$']
    legend_labels = ['Const.', 'Sine', 'Both']
    separator_indices = [6, 11, 13]
    group_labels = ['Priority 1', 'Prioriity 2', 'DECK', 'CS3']

    return {
        "baseline_results": baseline_MESM,
        "optimized_results_list": opt_MESM,
        "scenario_keys": scenario_keys,
        "legend_labels": legend_labels,
        "co2_data": co2_data,
        "global_mean_temp": global_mean_temp,
        "x_labels": x_labels,
        "separator_indices": separator_indices,
        "group_labels": group_labels,
    }

# ==================================================================
# Part 5b: notebook-facing data-prep for supplementary_notebooks/SI_plots.ipynb
# Same pattern as the fig1-7 helpers above: each function loads/aggregates
# exactly what one utils_plotting.plot_comparison_results/
# plot_vertical_stacked_bars call needs, so SI_plots.ipynb only ever does
# "load data" then "plot data".
# ==================================================================

def load_SI_ic_sensitivity_data(baseline_path: str = 'checkpoints/co2/baseline_co2_only.pkl') -> dict:
    """
    kwargs for utils_plotting.plot_comparison_results: CO2-only
    initial-condition sensitivity sweep (constant/gaussian/sine).
    """
    with open(baseline_path, 'rb') as f:
        baseline_results = pickle.load(f)

    IC_list = ['constant', 'gaussian', 'sine']
    return {
        "result_paths": [f'data/SI_results/sensitivity_initial_condition/inverse_{IC}_co2_only.pkl' for IC in IC_list],
        "column_titles": ['(a) Constant', '(b) Gaussian', '(c) Sinusoid'],
        "baseline_errors": [baseline_results['All']['mean']] * len(IC_list),
        "active_agents": ("CO2",),
    }


def load_SI_architecture_sensitivity_data(IC: str = 'sine') -> dict:
    """
    kwargs for utils_plotting.plot_comparison_results: CO2-only MLP-hidden-
    layer-architecture sensitivity sweep.
    """
    arch_list = ['8', '16', '32', '16_16']
    baseline_errors = []
    for arch in arch_list:
        with open(f'data/SI_results/sensitivity_architecture/baseline_{arch}_co2_only.pkl', 'rb') as f:
            baseline_errors.append(pickle.load(f)['All']['mean'])

    return {
        "result_paths": [f'data/SI_results/sensitivity_architecture/inverse_{arch}_co2_only_{IC}_test.pkl' for arch in arch_list],
        "column_titles": ['(a) [8]', '(b) [16]', '(c) [32]', '(d) [16, 16]'],
        "baseline_errors": baseline_errors,
        "active_agents": ("CO2",),
        "save_path": f'SI_arch_{IC}',
    }


def load_SI_feature_sensitivity_data(IC: str = 'sine') -> dict:
    """
    kwargs for utils_plotting.plot_comparison_results: CO2-only EMA-feature-
    window (short/medium/long) sensitivity sweep.
    """
    feat_list = ['short', 'medium', 'long']
    baseline_errors = []
    for feat in feat_list:
        with open(f'data/SI_results/sensitivity_features/baseline_{feat}_co2_only.pkl', 'rb') as f:
            baseline_errors.append(pickle.load(f)['All']['mean'])

    return {
        "result_paths": [f'data/SI_results/sensitivity_features/inverse_{feat}_co2_only_{IC}.pkl' for feat in feat_list],
        "column_titles": ['(a) Short', '(b) Medium', '(c) Long'],
        "baseline_errors": baseline_errors,
        "active_agents": ("CO2",),
        "save_path": f'SI_feat_{IC}',
    }


def regenerate_SI_extended_results_cache(agents: list[str] = ['N2O', 'Sulfur', 'BC']) -> None:
    """
    Recompute the per-agent optimal-emulator NRMSE summary (used by the SI
    extended-results stacked-bar figure) for every agent in `agents` and
    overwrite data/SI_results/extended_results/optimal_<agent>_only.pkl.

    Not called by default from the notebook - load_SI_extended_results_data()
    reads the existing cache instead. Call this only to refresh the cache
    after new single-agent inverse-optimization checkpoints are produced.
    Only covers N2O/Sulfur/BC by default, matching the original notebook -
    the CO2/CH4 baseline+optimal caches this figure also reads were produced
    by a separate, untracked process outside this notebook, not this loop.
    """
    for agent in agents:
        agent_lower = agent if agent in ('Sulfur', 'BC') else agent.lower()
        active_agents = (agent,)

        params0, _ = generate_init_params_and_train_data(
            [agent], active_agents, test_scen='historical', hidden_sizes=[16], idx_demo=1, verbose=False
        )
        eval_sets, *_ = generate_eval_data([agent], CS3=True, DAMIP=False, GeoMIP=False)

        training_paths = [
            f'checkpoints/{agent_lower}/inverse_constant_tier1_{agent_lower}_only.pkl',
            f'checkpoints/{agent_lower}/inverse_constant_tier2_{agent_lower}_only.pkl',
            f'checkpoints/{agent_lower}/inverse_constant_DECK_{agent_lower}_only.pkl',
            f'checkpoints/{agent_lower}/inverse_constant_CS3_{agent_lower}_only.pkl',
            f'checkpoints/{agent_lower}/inverse_constant_all_{agent_lower}_only.pkl',
        ]
        train_scenarios = ['Opt. Tier 1', 'Opt. Tier 2', 'Opt. DECK', 'Opt. CS3', 'Opt. All']
        optimal_results = evaluate_optimal_emulator(
            training_paths=training_paths,
            train_scenarios=train_scenarios,
            eval_sets=eval_sets,
            params0=params0,
            active_agents=active_agents,
            inactive_mode="zeros",
            historical_name="historical",
            key=jax.random.PRNGKey(0),
            K=400,
            lr=5e-2,
            weight_decay=1e-2,
        )

        save_path = f'data/SI_results/extended_results/optimal_{agent_lower}_only.pkl'
        with open(save_path, "wb") as f:
            pickle.dump(optimal_results, f)


def load_SI_extended_results_data(agent_lower_list: list[str] = ['co2', 'ch4', 'n2o', 'Sulfur', 'BC']) -> dict:
    """
    kwargs for utils_plotting.plot_vertical_stacked_bars: SI extended-results
    summary across all 5 single-agent optimizations.

    Fixes a real pre-existing bug: the original notebook's call passed only 5
    positional args (baseline_results_list, optimal_results_list,
    train_scenarios, test_scenarios, weights), but
    plot_vertical_stacked_bars requires 7 positional-capable args
    (...train_scenarios, test_scenarios, x_labels, leg_labels, weights...) -
    weights was silently landing in the x_labels slot and the call would
    TypeError on the two missing required args (leg_labels, weights) if run
    against the current function signature. x_labels/leg_labels below use
    the same values as the analogous Figure 5 call
    (load_fig5_data/utils_plotting.plot_vertical_stacked_bars in
    utils_inverse.py), since both plot the same train_scenarios/
    test_scenarios/weights for the same kind of grouped-bar summary.
    """
    baseline_results_list, optimized_results_list = [], []
    for agent_lower in agent_lower_list:
        with open(f'data/SI_results/extended_results/baseline_{agent_lower}_only.pkl', 'rb') as f:
            baseline_results_list.append(pickle.load(f))
        with open(f'data/SI_results/extended_results/optimal_{agent_lower}_only.pkl', 'rb') as f:
            optimized_results_list.append(pickle.load(f))

    return {
        "baseline_results_list": baseline_results_list,
        "optimized_results_list": optimized_results_list,
        "train_scenarios": ['Opt. Tier 1', 'Opt. Tier 2', 'Opt. DECK', 'Opt. CS3', 'Opt. All'],
        "test_scenarios": ['Tier 1', 'Tier 2', 'DECK', 'CS3'],
        "x_labels": ['Opt. Priority 1', 'Opt. Priority 2', 'Opt. DECK', 'Opt. CS3', 'Opt. All'],
        "leg_labels": ['Priority 1', 'Priority 2', 'DECK', 'CS3'],
        "weights": [7, 5, 2, 2],
        "titles": None,
        "figname": 'SI_extended_results',
    }
