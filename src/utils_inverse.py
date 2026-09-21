# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5, Claude Sonnet 5, and Gemini 3.1 Pro.
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
import csv
from pathlib import Path
from paths import DATA_DIR

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

    # indices (H-1).. (H+T-2) — clamp for safety if H==0
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
    # want EMA at indices (H-1).. (H+T-2)

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
    emis_curr_dict: dict,                     # dict: {"CO2": (T_cur,), "CH4": (T_cur,),...}
    years_hist: jnp.ndarray | None = None,                    # (T_hist,) or None
    emis_hist_dict: dict | None = None,                # dict or None: {"CO2": (T_hist,), "CH4": (T_hist,),...}
    agents: tuple = AGENTS_DEFAULT,              # tuple/list: which agents to include and their row order
    mode: str = 'FaIR',
    dt: float = 0.1,
    params: dict | None = None,
    use_checkpoint: bool = False,
) -> jnp.ndarray:
    """
    Returns GMST over the current scenario's years using the new multi-agent simulator.

    Inputs:
      - emis_curr_dict values are in native units per agent:
          CO2: GtCO2/yr, CH4: MtCH4/yr (consistent with simulate_temp)
      - Missing agents in emis_*_dict are zero-filled (keeps column layout stable).
      - params: optional explicit SCM parameter dict (e.g.
        utils_FaIR_JAX.params_from_theta(theta, base_params)), forwarded to
        simulate_temp in place of its own mode-based FAIR_PARAMS/MESM_PARAMS
        lookup. Differentiable w.r.t. theta - this is what makes the function
        usable as a calibration forward-model, not just an evaluation one (see
        the deprecated alternate MESM calibration). None (default) preserves the
        existing mode-based behaviour exactly.
      - use_checkpoint: forwarded to simulate_temp's own gradient-checkpointing
        option (see its docstring). False (default) preserves existing cost;
        only used by a jax.grad caller over a long trajectory.
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
            params=params,
            dt=dt,
            use_checkpoint=use_checkpoint,
        )
        GMST_curr = out_all["GMST"][-Tcur:]
    else:
        # No history: stack current only (zero-fill missing)
        emissions_by_agent_c = _stack_emissions(agents, emis_curr_dict, Tcur)
        out_c = utils_FaIR_JAX.simulate_temp(
            years=yrs_c,
            emissions_by_agent=emissions_by_agent_c,
            mode=mode,
            params=params,
            dt=dt,
            use_checkpoint=use_checkpoint,
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
    """Feature width of the first (X,...) row found across train/test datasets, else 0."""
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
        List of dicts [{'W':.., 'b':..},...] including the final output layer.
    """
    params = []
    # The architecture flows from input -> hidden_1 ->... -> hidden_n -> output (scalar)
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

    `batch_size`: default None reproduces the prior behavior exactly.
    When set to an int, each of the K steps samples `batch_size` rows
    without replacement from Xtr/ytr using `key` (split fresh every step),
    making this stochastic gradient descent in the classical sense.
    `batch_size >= Xtr.shape[0]` falls back to full-batch.
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
    the resulting mean test NRMSE.
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


# ---------------------------------------------------------------------------
# Smoothness penalty: reference-normalized form
# ---------------------------------------------------------------------------
SIGMA_REF_SCENARIOMIP = {
    "CO2": 25.0477, "CH4": 185.0055, "N2O": 5.2361,
    "Sulfur": 29.9742, "BC": 1.8038,
}

# "legacy"     : sum_a sum_t (dU_a)^2, unnormalized, in each agent's native
#                units.
# "normalized" : (1/|A|) sum_a mean_t[(dU_a)^2] / sigma_ref_a^2. Dimensionless,
#                length-normalized and agent-count-normalized, so a single
#                shared weight applies comparable pressure across regimes. The
#                denominator is a CONSTANT, not a function of U, so the
#                optimizer cannot inflate it instead of smoothing the numerator.
PENALTY_FORMS = ("legacy", "normalized")


def smoothness_penalty_terms(U_eff: dict, active_agents: tuple, penalty_form: str):
    """The penalty as a JAX scalar, for the objective's use."""
    if penalty_form == "legacy":
        reg = 0.0
        for a in active_agents:
            reg = reg + jnp.sum(jnp.diff(U_eff[a]) ** 2)
        return reg
    if penalty_form == "normalized":
        reg = 0.0
        for a in active_agents:
            sig = SIGMA_REF_SCENARIOMIP[a]
            reg = reg + jnp.mean(jnp.diff(U_eff[a]) ** 2) / (sig ** 2)
        return reg / max(len(active_agents), 1)
    raise ValueError(f"penalty_form must be one of {PENALTY_FORMS}, got {penalty_form!r}")


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
    penalty_form: str = "legacy",
) -> callable:
    """
    Build the bilevel objective `objective(U_pytree) -> (loss, aux)`: given a candidate
    emissions trajectory U_pytree, (1) mask inactive agents, (2) build a single-scenario
    train set from U_pytree (build_train), (3) train a fresh MLP emulator on it for
    K_inner SGD steps (train_mlp_sgd), (4) evaluate that emulator's mean NRMSE on
    `test_dataset_all`, plus an optional first-difference smoothness
    penalty on U_pytree. The outer gradient (w.r.t. U_pytree) is obtained by
    differentiating through the entire inner training loop.
    aux = (paramsK, test_s, train_temp_raw) for logging/checkpointing.
    """
    def objective(U_pytree):
        U_eff = _apply_active_mask_to_emis(U_pytree, active_agents, inactive_mode)

        # First-difference penalty on U, penalizing jaggedness directly.
        # See PENALTY_FORMS for why the normalized variant exists; "legacy"
        # is the default so every existing checkpoint stays reproducible.
        reg_loss = 0.0
        if smoothness_weight > 0.0:
            reg_loss = smoothness_penalty_terms(U_eff, active_agents, penalty_form)

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

def load_inverse_ckpt_errors_only(path: str) -> dict:
    """Load just the 'errors' trajectory from a checkpoint saved by save_inverse_ckpt.
    """
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {"errors": jnp.asarray(raw["errors"], dtype=jnp.float32)}


# -----------------------------------------------------------------------------
# Recovering true NRMSE from a checkpoint's recorded objective
# -----------------------------------------------------------------------------
# optimize_emissions_inverse records the FULL objective in 'errors', not the
# error term (see make_inverse_objective_single_train):
#
#     errors[k] = nrmse[k] + smoothness_weight * sum_agents sum_t (dU)^2

def smoothness_penalty(U: dict, penalty_form: str = "legacy",
                      active_agents: tuple | None = None) -> float:
    """Numpy twin of smoothness_penalty_terms, for post-hoc NRMSE recovery.
    """
    agents = active_agents if active_agents is not None else tuple(U)
    if penalty_form == "normalized":
        terms = [float(np.mean(np.diff(np.asarray(U[a], dtype=np.float64)) ** 2)
                       / SIGMA_REF_SCENARIOMIP[a] ** 2) for a in agents]
        return float(sum(terms) / max(len(terms), 1))
    if penalty_form != "legacy":
        raise ValueError(f"penalty_form must be one of {PENALTY_FORMS}, got {penalty_form!r}")
    return _smoothness_penalty_legacy(U)


def _smoothness_penalty_legacy(U: dict) -> float:
    """Sum of squared first differences over every agent in one emissions iterate.
    """
    return float(sum(np.sum(np.diff(np.asarray(v, dtype=np.float64)) ** 2)
                     for v in U.values()))


def _nrmse_from_preds(preds_entry: list, eps: float = 1e-8) -> float:
    """Recompute avg_nrmse_over_tests from one saved preds_traj entry.
    """
    vals, weights = [], []
    for (_scen, yhat, ytrue) in preds_entry:
        yhat = np.asarray(yhat, dtype=np.float64)
        ytrue = np.asarray(ytrue, dtype=np.float64)
        rmse = np.sqrt(np.mean((yhat - ytrue) ** 2))
        vals.append(rmse / (np.max(np.abs(ytrue)) + eps))
        weights.append(len(ytrue))
    return float(np.average(vals, weights=weights))

_NRMSE_RECOMPUTE_RTOL = 1e-5


def _infer_preds_every(raw: dict) -> int:
    """How many outer steps separate consecutive preds_traj entries.
    """
    n_err, n_preds = len(raw["errors"]), len(raw.get("preds_traj") or [])
    if n_preds > 1 and (n_err - 1) % (n_preds - 1) == 0:
        return (n_err - 1) // (n_preds - 1)
    recorded = (raw.get("meta") or {}).get("preds_every")
    return int(recorded) if recorded else 50


def _penalty_form_of(raw: dict) -> str:
    """The penalty form a checkpoint was written under.
    """
    return str((raw.get("meta") or {}).get("penalty_form") or "legacy")


def _active_of(raw: dict) -> tuple | None:
    aa = (raw.get("meta") or {}).get("active_agents")
    return tuple(aa) if aa else None


def recover_smoothness_weight(raw: dict, preds_every: int | None = None) -> tuple[float, float]:
    """Solve for the smoothness_weight a checkpoint was produced with.

    Objective's NRMSE term can be recomputed directly and the penalty read off as the residual,

        w = (errors[k] - nrmse[k]) / smoothness_penalty(U_traj[k-1])
    """
    errors = np.asarray(raw["errors"], dtype=np.float64)
    U_traj, preds_traj = raw["U_traj"], raw["preds_traj"]
    form, active = _penalty_form_of(raw), _active_of(raw)
    if preds_every is None:
        preds_every = _infer_preds_every(raw)

    residuals, penalties, n_sampled = [], [], 0
    for j in range(1, len(preds_traj)):
        k = preds_every * j
        if k >= len(errors) or k - 1 >= len(U_traj):
            break
        n_sampled += 1
        penalty = smoothness_penalty(U_traj[k - 1], form, active)
        if penalty > 0.0:
            residuals.append(errors[k] - _nrmse_from_preds(preds_traj[j]))
            penalties.append(penalty)

    if n_sampled == 0:
        raise ValueError(
            f"could not sample any (errors, preds_traj) pair at preds_every="
            f"{preds_every} (len(errors)={len(errors)}, "
            f"len(preds_traj)={len(preds_traj)}) - cannot recover smoothness_weight"
        )
    if not residuals:
        return 0.0, 0.0

    residuals, penalties = np.array(residuals), np.array(penalties)

    if np.max(np.abs(residuals)) <= _NRMSE_RECOMPUTE_RTOL * np.max(np.abs(errors)):
        return 0.0, 0.0

    solutions = residuals / penalties
    return float(np.median(solutions)), float(np.ptp(solutions))


def recover_nrmse_trajectory(raw: dict, smoothness_weight: float | None = None,
                             preds_every: int | None = None) -> np.ndarray:
    """Return the true NRMSE trajectory for a checkpoint, penalty removed.

    `smoothness_weight=None` (the default) recovers it from the checkpoint
    itself via recover_smoothness_weight. Pass an explicit value only to
    override that - e.g. to check a checkpoint against a config file.
    """
    errors = np.asarray(raw["errors"], dtype=np.float64)
    if smoothness_weight is None:
        recorded = (raw.get("meta") or {}).get("smoothness_weight")
        smoothness_weight = (float(recorded) if recorded is not None
                             else recover_smoothness_weight(raw, preds_every=preds_every)[0])

    if smoothness_weight == 0.0:
        return errors

    U_traj = raw["U_traj"]
    penalties = np.array([smoothness_penalty(U_traj[max(k - 1, 0)],
                                             _penalty_form_of(raw), _active_of(raw))
                          for k in range(len(errors))])
    nrmse = errors - smoothness_weight * penalties
    if np.any(nrmse <= 0.0):
        raise ValueError(
            f"smoothness_weight={smoothness_weight:g} drives NRMSE non-positive "
            f"(min {nrmse.min():.6g}) - the weight does not match this checkpoint"
        )
    return nrmse


def load_inverse_ckpt_nrmse_only(path: str, smoothness_weight: float | None = None,
                                 preds_every: int | None = None) -> dict:
    """Memory-light loader returning the penalty-corrected NRMSE trajectory.
    """
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {"errors": jnp.asarray(
        recover_nrmse_trajectory(raw, smoothness_weight, preds_every), dtype=jnp.float32)}


def scale_by_coord_pytree(weights_pytree: dict) -> optax.GradientTransformation:
    """An optax transform that elementwise-multiplies gradients by a matching pytree of weights."""
    def _mul(g, w): return g * w
    def init_fn(_): return ()
    def update_fn(updates, state, params=None):
        return jax.tree_util.tree_map(_mul, updates, weights_pytree), state
    return optax.GradientTransformation(init_fn, update_fn)

def preds_by_scenario(params: list[dict], test_list: list) -> list[tuple[str, jnp.ndarray, jnp.ndarray]]:
    """Run the MLP on every (X, y, scenario) test row; returns [(scenario, y_hat, y_true),...]."""
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
    penalty_form: str = "legacy",
) -> dict:
    """
    Finds an emissions trajectory U (one (T,) series per agent, starting from `init_cond`)
    that minimizes the objective built by make_inverse_objective_single_train - i.e. the trajectory whose
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
        ema_windows_years=ema_windows_years,
        penalty_form=penalty_form,
    )

    # --- 1. Create a Pure Loss Function (No Strings) ---
    # JAX cannot JIT functions that return strings in `aux`.
    # We wrap the objective to strip the strings from 'test_s'.
    def loss_fn_pure(U):
        loss, (paramsK, test_s, train_temp_raw) = objective(U)
        # test_s is [(X, y, name),...]. Strip 'name' for JIT.
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

    meta = {
        "num_updates": num_updates, "step_size": step_size, "momentum": momentum,
        "nesterov": nesterov, "K_inner": K_inner, "lr_inner": lr_inner,
        "wd_inner": wd_inner, "batch_size": batch_size,
        "smoothness_weight": smoothness_weight, "penalty_form": penalty_form,
        "sigma_ref": (SIGMA_REF_SCENARIOMIP if penalty_form == "normalized" else None),
        "init_cond": init_cond
            if isinstance(init_cond, str) or init_cond is None else "array",
        "T": T, "filter_hist": filter_hist, "mode": mode,
        "ema_windows_years": ema_windows_years, "preds_every": preds_every,
        "active_agents": list(active_agents) if active_agents else None,
    }

    if resume_if_exists and checkpoint_path and os.path.isfile(checkpoint_path):
        print("Resuming from checkpoint...")
        ckpt = load_inverse_ckpt(checkpoint_path)
        stored_meta = ckpt.get("meta") or {}
        if stored_meta:
            bad = _resume_config_mismatches(stored_meta, meta)
            if bad:
                raise ValueError(
                    "refusing to resume: this run's configuration differs from the "
                    f"checkpoint at {checkpoint_path}.\n  "
                    + "\n  ".join(bad)
                    + "\nResuming would apply the new settings to the restored "
                      "optimizer state and splice two different objectives into one "
                      "error curve. Write to a new checkpoint_path, or pass "
                      "resume_if_exists=False to overwrite deliberately.")
        else:
            print("  (checkpoint predates meta; cannot verify config match)")
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
                "paramsK_k": _tree_to_numpy(paramsK_k), "meta": meta,
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
    train_scaled: list,                 # list[(X_s, y, scen),...]
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
    eval_sets: dict,              # {"H": emis_dict_test_H, "S": emis_dict_test_S,...}
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
          "test_scaled": [(Xs, y, scen),...]  # scaled test set used
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
    (see 07b_seed_uncertainty_sweep) to vary emulator-training randomness.
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
  (missing trailing comma).
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
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
    batch_size: int | None = None,
    u_index: int = -1,
) -> dict | tuple[dict, dict]:
    """
    For each optimize_emissions_inverse checkpoint in `training_paths` (final U in
    the trajectory), train a fresh MLP on the resulting optimal emissions and
    evaluate NRMSE against every eval_sets entry. Returns results_out

    `u_index` (default -1 = the final iterate, i.e. exactly the prior behaviour)
    selects which outer iterate to evaluate. Any other value turns this into an
    out-of-sample convergence probe: the checkpoint's own `errors` curve is the
    bilevel objective measured IN-SAMPLE on the group being optimized, so it
    falls by construction and cannot say whether generalization is still
    improving. Evaluating U_traj[k] for a series of k answers that, and is what
    decides whether running longer helps or just overfits the target group.

    Returns results_out[train_label][eval_set][scenario] = NRMSE, plus 'mean',
    and y_hat_all (raw predictions) if ind_effects=True.
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

        n_iters = len(raw["U_traj"])
        if not -n_iters <= u_index < n_iters:
            raise IndexError(
                f"u_index={u_index} out of range for {path} "
                f"(trajectory has {n_iters} iterates)")
        U_final_dict = _tree_to_jnp(raw["U_traj"][u_index])

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
        paramsK, _ = train_mlp_sgd(
            params0, Xtr, ytr, K=K, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, key=key,
        )

        results_out[train_label] = {}
        if ind_effects:
            y_hat_all[train_label] = {}
        for test_name, emis_dict_test in eval_sets.items():
            test_raw = build_dataset_from_runfair_dict(
                emis_dict_test, historical_name=historical_name, mode=mode,
                ema_windows_years=ema_windows_years
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

    # For analysis
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
    initialization
    .
    `K`/`lr`/`weight_decay` are the baseline's own training hyperparameters
    (independently tunable from the optimized emulator's - see the baseline search,
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
    split; every other group name is a copy of the
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

RESUME_INVARIANT_KEYS = (
    "step_size", "momentum", "nesterov", "K_inner", "lr_inner", "wd_inner",
    "batch_size", "smoothness_weight", "penalty_form", "init_cond", "T",
    "filter_hist", "mode", "active_agents", "ema_windows_years",
)


def _resume_config_mismatches(stored: dict, incoming: dict) -> list[str]:
    """Which invariant hyperparameters differ between a checkpoint and this call.
    """
    out = []
    for k in RESUME_INVARIANT_KEYS:
        if k not in stored:
            continue
        a, b = stored[k], incoming.get(k)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            same = list(a) == list(b)
        elif isinstance(a, dict) and isinstance(b, dict):
            same = (set(a) == set(b)
                    and all(np.isclose(float(a[j]), float(b[j])) for j in a))
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                and not isinstance(a, bool) and not isinstance(b, bool):
            same = bool(np.isclose(float(a), float(b)))
        else:
            same = a == b
        if not same:
            out.append(f"{k}: checkpoint has {a!r}, this run passes {b!r}")
    return out


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
    penalty_form: str = "legacy",
    active_agents: tuple[str, ...] | None = None,
    mode: str | None = None,
    batch_size: int | None = None,
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> dict:
    """
    Run one optimize_emissions_inverse experiment against `group` - either
    'H-ext' (the single-scenario target used by every notebook's "2a"
    section) or one of the named eval_sets groups built by
    run_inverse_experiment_setup ('tier1', 'tier2', 'DECK', 'CS3', 'DAMIP',
    'GeoMIP', 'all'). The checkpoint path is auto-built as
    f"{checkpoint_dir}/inverse_{init_cond}_{group}_{tag}.pkl", uniformly for
    every group including 'H-ext'.
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
        penalty_form=penalty_form,
        mode=mode,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        resume_if_exists=resume_if_exists,
        preds_every=preds_every,
        batch_size=batch_size,
        ema_windows_years=ema_windows_years,
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
    # Architecture: input -> hidden... -> hidden -> output_vector
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
    Notebook-facing wrapper: build vector-
    target train data (prepare_data_vector), train a vector-output MLP
    (train_mlp_sgd_vector), evaluate it (evaluate_emulator_vector_over_multiple_tests),
    and optionally pickle the results to save_path.
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
        data_dir (str): Directory where the.pkl files are stored.

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
# Part 5: notebook-facing data-prep for pipeline/08_plotting/paper_plots.ipynb figures
# ==================================================================

def load_fig1_tier1_scenarios(agents: list[str] = ['CO2']) -> dict:
    """
    Tier-1 calibration emissions for Figure 1's tier1-scenarios panel
    (utils_plotting.plot_fig01_tier1_scenarios). Returns {'years', 'tier1', 'group'}, one
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
    utils_plotting.plot_fig02_co2_example / plot_stacked_results_ppt.
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
    utils_plotting.plot_fig03_single_forcing.
    Returns {'results_inverse', 'results_baseline', 'labels'}.
    """
    results_inverse, results_baseline = [], []
    for a in agents:
        path_inverse = f'checkpoints/{a}/inverse_constant_tier1_{a}_only.pkl'
        path_baseline = f'checkpoints/{a}/baseline_{a}_only.pkl'
        ckpt = load_inverse_ckpt(path_inverse)
        ckpt["errors"] = jnp.asarray(recover_nrmse_trajectory(ckpt), dtype=jnp.float32)
        results_inverse.append(ckpt)
        with open(path_baseline, "rb") as f:
            results_baseline.append(pickle.load(f)['Tier 1']['mean'])

    labels = ['(a) CO$_2$', '(b) CH$_4$', '(c) N$_2$O', '(d) Sulfur', '(e) BC']
    return {"results_inverse": results_inverse, "results_baseline": results_baseline, "labels": labels}


_FIG3_AGENT_TAGS = {'co2': 'co2_only', 'ch4': 'ch4_only', 'n2o': 'n2o_only',
                    'Sulfur': 'Sulfur_only', 'BC': 'BC_only'}

_FIG3_AGENT_DIRS = {'Sulfur': 'checkpoints/Sulfur_smooth/seed_sweep'}


def load_fig3_single_forcing_data_seed_sweep(
    agents: list[str] = ['co2', 'ch4', 'n2o', 'Sulfur', 'BC'],
    seeds: list[int] = tuple(range(50)),
) -> dict:
    """
    Multi-seed companion to load_fig3_single_forcing_data: kwargs for
    utils_plotting.plot_fig03_single_forcing with seed_errors_list/
    seed_baseline_error_list populated - each agent's tier1-group `errors`
    trajectory and baseline score, loaded across every seed in `seeds` from
    checkpoints/{agent}_retuned/seed_sweep/ (written by
    pipeline/03_hyperparameters/03b_regenerate_checkpoints_co2.py for CO2, 03b_regenerate_checkpoints_agent.py --agent X for the other four).
    """
    seed_errors_list, seed_baseline_error_list = [], []
    for a in agents:
        tag = _FIG3_AGENT_TAGS[a]
        checkpoint_dir = _FIG3_AGENT_DIRS.get(a, f'checkpoints/{a}_retuned/seed_sweep')
        errs, bases = [], []
        for seed in seeds:
            ckpt_path = f'{checkpoint_dir}/inverse_constant_tier1_{tag}_seed{seed}.pkl'
            baseline_path = f'{checkpoint_dir}/baseline_{tag}_seed{seed}.pkl'
            if not Path(ckpt_path).exists() or not Path(baseline_path).exists():
                raise FileNotFoundError(
                    f"{ckpt_path} or {baseline_path} missing - run "
                    f"pipeline/03_hyperparameters/03b_regenerate_checkpoints_agent.py --agent {a} first "
                    f"(pipeline/03_hyperparameters/03b_regenerate_checkpoints_co2.py for co2)"
                )
            errs.append(load_inverse_ckpt_nrmse_only(ckpt_path))
            with open(baseline_path, "rb") as f:
                bases.append(pickle.load(f)['Tier 1']['mean'])
        seed_errors_list.append(errs)
        seed_baseline_error_list.append(bases)

    labels = ['(a) CO$_2$', '(b) CH$_4$', '(c) N$_2$O', '(d) Sulfur', '(e) BC']
    return {"seed_errors_list": seed_errors_list,
            "seed_baseline_error_list": seed_baseline_error_list,
            "labels": labels}


def regenerate_fig4_all_agents_cache(
    save_path: str = 'data/plotting/optimal_all_agents_subset.pkl',
) -> dict:
    """
    Recompute the all-agents optimal-emulator NRMSE summary used by the
    Figure-4 performance-summary bar chart and overwrite `save_path`'s cache.
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
    write both to save_path. 
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
    rebuilds that seed's own (baseline_results, optimal_results) pair 
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
                    f"{p} missing - run pipeline/03_hyperparameters/03b_regenerate_checkpoints_co2.py --seed {seed} first"
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


def regenerate_fig4_all_agents_cache_seed_sweep(
    seeds: list[int] = tuple(range(50)),
    checkpoint_dir: str = 'checkpoints/multi_fig4/seed_sweep',
    tag: str = 'multi_fig4',
    unified_config_path: str = 'data/SI_results/hp_retune/multi/best_config_unified.json',
    baseline_config_path: str = 'data/SI_results/baseline_hp/k400_search_multi/best_baseline_config_K400.json',
    out_path: str = 'data/SI_results/seed_uncertainty/fig4_seed_spread_all_agents.pkl',
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
) -> dict[int, dict]:
    """
    Multi-agent analog of regenerate_fig4_co2_only_cache_seed_sweep, for
    Figure 4's multi-agent panel 
    """
    unified_cfg = json.load(open(unified_config_path))["config"]
    baseline_cfg = json.load(open(baseline_config_path))["config"]

    agents = ['CO2', 'CH4', 'N2O', 'Sulfur', 'BC']
    active_agents = ('CO2', 'CH4', 'N2O', 'Sulfur', 'BC')
    groups = ['tier1', 'tier2', 'DECK', 'CS3', 'all']
    train_scenarios = ['Opt. Tier 1', 'Opt. Tier 2', 'Opt. DECK', 'Opt. CS3', 'Opt. All']

    all_results = {}
    for seed in seeds:
        setup = run_inverse_experiment_setup(
            agents, active_agents, mode='FaIR', CS3=True, DAMIP=False, GeoMIP=False,
            idx_demo=None, seed=seed,
            baseline_K=baseline_cfg["K"], baseline_lr=baseline_cfg["lr"],
            baseline_weight_decay=baseline_cfg["weight_decay"],
            ema_windows_years=ema_windows_years,
        )

        training_paths = [f'{checkpoint_dir}/inverse_constant_{g}_{tag}_seed{seed}.pkl' for g in groups]
        for p in training_paths:
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"{p} missing - run pipeline/03_hyperparameters/03b_regenerate_checkpoints_multi_fig4.py "
                    f"--seed {seed} first"
                )

        optimal_results = evaluate_optimal_emulator(
            training_paths=training_paths,
            train_scenarios=train_scenarios,
            eval_sets=setup["eval_sets"],
            params0=setup["params0"],
            agents=agents,
            active_agents=active_agents,
            inactive_mode="zeros",
            historical_name="historical",
            key=jax.random.PRNGKey(seed),
            K=unified_cfg["K_inner"],
            lr=unified_cfg["lr_inner"],
            weight_decay=unified_cfg["wd_inner"],
            mode='FaIR',
            ema_windows_years=ema_windows_years,
            batch_size=unified_cfg["batch_size"],
        )

        all_results[seed] = {"baseline": setup["baseline_results"], "optimal": optimal_results}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)

    return all_results

_AGENT_SEED_SWEEP_TAGS = {"CH4": "ch4_only", "N2O": "n2o_only", "Sulfur": "Sulfur_only", "BC": "BC_only"}


def regenerate_SI_extended_results_cache_seed_sweep(
    agent: str,
    seeds: list[int] = tuple(range(50)),
    checkpoint_dir: str | None = None,
    tag: str | None = None,
    unified_config_path: str | None = None,
    baseline_config_path: str = None,
    out_path: str | None = None,
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
) -> dict[int, dict]:
    """
    Per-agent (CH4/N2O/Sulfur/BC) analog of regenerate_fig4_co2_only_cache_seed_sweep,
    for SI Fig 6 (extended results, 5 panels CO2/CH4/N2O/Sulfur/BC)
    """
    if baseline_config_path is None:
        raise ValueError(
            "baseline_config_path must be given explicitly - the transfer check's transfer "
            "check (the baseline transfer check) decides, per agent, "
            "whether to reuse CO2's tuned config or that agent's own dense-search winner."
        )
    agent_lower = agent if agent in ("Sulfur", "BC") else agent.lower()
    if checkpoint_dir is None:
        checkpoint_dir = f"checkpoints/{agent_lower}_retuned/seed_sweep"
    if tag is None:
        tag = _AGENT_SEED_SWEEP_TAGS[agent]
    if unified_config_path is None:
        unified_config_path = f"data/SI_results/hp_retune/{agent}/best_config_unified.json"
    if out_path is None:
        out_path = f"data/SI_results/seed_uncertainty/SI_extended_seed_spread_{agent}.pkl"

    unified_cfg = json.load(open(unified_config_path))["config"]
    baseline_cfg = json.load(open(baseline_config_path))["config"]

    groups = ['tier1', 'tier2', 'DECK', 'CS3', 'all']
    train_scenarios = ['Opt. Tier 1', 'Opt. Tier 2', 'Opt. DECK', 'Opt. CS3', 'Opt. All']

    all_results = {}
    for seed in seeds:
        setup = run_inverse_experiment_setup(
            [agent], (agent,), mode='FaIR', CS3=True, DAMIP=False, GeoMIP=False,
            idx_demo=None, seed=seed,
            baseline_K=baseline_cfg["K"], baseline_lr=baseline_cfg["lr"],
            baseline_weight_decay=baseline_cfg["weight_decay"],
            ema_windows_years=ema_windows_years,
        )

        training_paths = [f'{checkpoint_dir}/inverse_constant_{g}_{tag}_seed{seed}.pkl' for g in groups]
        for p in training_paths:
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"{p} missing - run pipeline/03_hyperparameters/03b_regenerate_checkpoints_agent.py "
                    f"--agent {agent} --seed {seed} first"
                )

        optimal_results = evaluate_optimal_emulator(
            training_paths=training_paths,
            train_scenarios=train_scenarios,
            eval_sets=setup["eval_sets"],
            params0=setup["params0"],
            active_agents=(agent,),
            inactive_mode="zeros",
            historical_name="historical",
            key=jax.random.PRNGKey(seed),
            K=unified_cfg["K_inner"],
            lr=unified_cfg["lr_inner"],
            weight_decay=unified_cfg["wd_inner"],
            mode='FaIR',
            ema_windows_years=ema_windows_years,
            batch_size=unified_cfg["batch_size"],
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
    utils_plotting._plot_vertical_stacked_bars.
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
        "figname": 'fig04_scm_summary',
    }

def load_fig5_multi_forcing_data(
    path_inverse: str = 'checkpoints/multi_fig4_smooth/seed_sweep/inverse_constant_tier1_multi_fig4_seed0.pkl',
    path_baseline: str = 'checkpoints/multi_fig4_smooth/seed_sweep/baseline_multi_fig4_seed0.pkl',
) -> dict:
    """
    Multi-agent inverse vs. baseline NRMSE (Figure 5) for
    utils_plotting.plot_fig05_multi_forcing. Returns {'results', 'baseline_error'}.
    """
    results = load_inverse_ckpt(path_inverse)
    results["errors"] = jnp.asarray(recover_nrmse_trajectory(results), dtype=jnp.float32)
    with open(path_baseline, "rb") as f:
        baseline_error = pickle.load(f)['Tier 1']['mean']
    return {"results": results, "baseline_error": baseline_error}

def load_fig5_multi_forcing_data_seed_sweep(
    seeds: list[int] = tuple(range(50)),
    checkpoint_dir: str = 'checkpoints/multi_fig4_smooth/seed_sweep',
    tag: str = 'multi_fig4',
) -> dict:
    """Multi-seed companion to load_fig5_multi_forcing_data.
    """
    errs, bases, missing = [], [], []
    for seed in seeds:
        ckpt_path = f'{checkpoint_dir}/inverse_constant_tier1_{tag}_seed{seed}.pkl'
        baseline_path = f'{checkpoint_dir}/baseline_{tag}_seed{seed}.pkl'
        if not Path(ckpt_path).exists() or not Path(baseline_path).exists():
            missing.append(seed)
            continue
        errs.append(load_inverse_ckpt_nrmse_only(ckpt_path))
        with open(baseline_path, "rb") as f:
            bases.append(pickle.load(f)['Tier 1']['mean'])

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(seeds)} seeds missing from {checkpoint_dir} "
            f"(first few: {missing[:5]}) - regenerate with "
            f"pipeline/03_hyperparameters/03b_regenerate_checkpoints_multi_fig4.py + "
            f"the cluster regeneration job, or transfer them from the cluster"
        )
    return {"seed_errors": errs, "seed_baseline_errors": bases}


def regenerate_fig6_individual_effects_cache(
    save_dir: str = 'data/plotting',
    unified_config_path: str = 'data/SI_results/hp_retune/multi/best_config_unified.json',
    baseline_config_path: str = 'data/SI_results/baseline_hp/k400_search_multi/best_baseline_config_K400.json',
    tier1_checkpoint_dir: str = 'checkpoints/multi_fig4_smooth/seed_sweep',
    tier1_tag: str = 'multi_fig4',
    multi_checkpoint_dir: str = 'checkpoints/multi_retuned_smooth/seed_sweep',
    multi_tag: str = 'all_agents',
    init_cond: str = 'constant',
    seed: int = 0,
) -> dict:
    """
    Recompute the per-agent baseline/optimal-emulator predictions used by the
    individual-effects figure (M-GHG / M-aer / G6sulfur scenarios) and
    overwrite their caches under `save_dir`.
    """
    agents = ['CO2', 'CH4', 'N2O', 'Sulfur', 'BC']
    active_agents = ('CO2', 'CH4', 'N2O', 'Sulfur', 'BC')

    unified_cfg = json.load(open(unified_config_path))["config"]
    baseline_cfg = json.load(open(baseline_config_path))["config"]

    params0, _ = generate_init_params_and_train_data(
        agents, active_agents, test_scen='historical', hidden_sizes=[16], idx_demo=None,
        verbose=False, seed=seed,
    )
    eval_sets_ind_effects, *_ = generate_eval_data(agents, CS3=True, DAMIP=True, GeoMIP=True)

    _, y_hat_baseline, y_true_ind_effects = generate_and_eval_baseline_emulator(
        eval_sets_ind_effects["Tier 1"], eval_sets_ind_effects, save_path=None,
        verbose=False, hidden_sizes=[16], seed=seed,
        K=baseline_cfg["K"], lr=baseline_cfg["lr"], weight_decay=baseline_cfg["weight_decay"],
    )

    training_paths_ind_effects = [
        f'{tier1_checkpoint_dir}/inverse_constant_tier1_{tier1_tag}_seed{seed}.pkl',
        f'{multi_checkpoint_dir}/inverse_{init_cond}_DAMIP_{multi_tag}_seed{seed}.pkl',
        f'{multi_checkpoint_dir}/inverse_{init_cond}_GeoMIP_{multi_tag}_seed{seed}.pkl',
        f'{multi_checkpoint_dir}/inverse_{init_cond}_all_{multi_tag}_seed{seed}.pkl',
    ]
    for _p in training_paths_ind_effects:
        if not Path(_p).exists():
            raise FileNotFoundError(
                f"{_p} missing - run pipeline/03_hyperparameters/03b_regenerate_checkpoints_multi.py "
                f"--seed {seed} --init-cond {init_cond} first")
    train_scenarios_ind_effects = ['Opt. Tier 1', 'Opt. DAMIP', 'Opt. GeoMIP', 'Opt. All']
    _, y_hat_ind_effects = evaluate_optimal_emulator(
        training_paths=training_paths_ind_effects,
        train_scenarios=train_scenarios_ind_effects,
        eval_sets=eval_sets_ind_effects,
        params0=params0,
        active_agents=active_agents,
        inactive_mode="zeros",
        historical_name="historical",
        key=jax.random.PRNGKey(seed),
        K=unified_cfg["K_inner"],
        lr=unified_cfg["lr_inner"],
        weight_decay=unified_cfg["wd_inner"],
        batch_size=unified_cfg["batch_size"],
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


def regenerate_fig6_individual_effects_cache_seed_sweep(
    seeds: list[int] = tuple(range(50)),
    tier1_checkpoint_dir: str = 'checkpoints/multi_fig4_smooth/seed_sweep',
    tier1_tag: str = 'multi_fig4',
    multi_checkpoint_dir: str = 'checkpoints/multi_retuned_smooth/seed_sweep',
    multi_tag: str = 'all_agents',
    init_cond: str = 'constant',
    unified_config_path: str = 'data/SI_results/hp_retune/multi/best_config_unified.json',
    baseline_config_path: str = 'data/SI_results/baseline_hp/k400_search_multi/best_baseline_config_K400.json',
    out_path: str = 'data/SI_results/seed_uncertainty/fig6_seed_spread_ind_effects.pkl',
    ema_windows_years: tuple = (5.0, 30.0, 100.0),
) -> dict[int, dict]:
    """
    Multi-seed companion to regenerate_fig6_individual_effects_cache, giving
    Figure 6 the same median+IQR treatment Figures 3/4/5 already have.
    """
    unified_cfg = json.load(open(unified_config_path))["config"]
    baseline_cfg = json.load(open(baseline_config_path))["config"]

    agents = ['CO2', 'CH4', 'N2O', 'Sulfur', 'BC']
    active_agents = ('CO2', 'CH4', 'N2O', 'Sulfur', 'BC')
    train_scenarios = ['Opt. Tier 1', 'Opt. DAMIP', 'Opt. GeoMIP', 'Opt. All']

    all_results = {}
    for seed in seeds:
        setup = run_inverse_experiment_setup(
            agents, active_agents, mode='FaIR', CS3=True, DAMIP=True, GeoMIP=True,
            idx_demo=None, seed=seed,
            baseline_K=baseline_cfg["K"], baseline_lr=baseline_cfg["lr"],
            baseline_weight_decay=baseline_cfg["weight_decay"],
            ema_windows_years=ema_windows_years,
        )

        training_paths = [
            f'{tier1_checkpoint_dir}/inverse_constant_tier1_{tier1_tag}_seed{seed}.pkl',
            f'{multi_checkpoint_dir}/inverse_{init_cond}_DAMIP_{multi_tag}_seed{seed}.pkl',
            f'{multi_checkpoint_dir}/inverse_{init_cond}_GeoMIP_{multi_tag}_seed{seed}.pkl',
            f'{multi_checkpoint_dir}/inverse_{init_cond}_all_{multi_tag}_seed{seed}.pkl',
        ]
        for p in training_paths:
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"{p} missing - run pipeline/03_hyperparameters/03b_regenerate_checkpoints_multi.py "
                    f"--seed {seed} --init-cond {init_cond} first"
                )

        _, y_hat_ind_effects = evaluate_optimal_emulator(
            training_paths=training_paths,
            train_scenarios=train_scenarios,
            eval_sets=setup["eval_sets"],
            params0=setup["params0"],
            active_agents=active_agents,
            inactive_mode="zeros",
            historical_name="historical",
            key=jax.random.PRNGKey(seed),
            K=unified_cfg["K_inner"],
            lr=unified_cfg["lr_inner"],
            weight_decay=unified_cfg["wd_inner"],
            batch_size=unified_cfg["batch_size"],
            ind_effects=True,
        )

        all_results[seed] = {
            "y_true": setup["ground_truth_delT"],
            "y_hat_baseline": setup["baseline_pred_delT"],
            "y_hat": y_hat_ind_effects,
        }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)

    return all_results


def load_fig6_data(
    save_dir: str = 'data/plotting',
    seed_cache_path: str | None = 'data/SI_results/seed_uncertainty/fig6_seed_spread_ind_effects.pkl',
) -> dict:
    """
    Load the cached per-agent baseline/optimal-emulator predictions for the
    individual-effects figure (utils_plotting.plot_fig06_individual_effects).
    """
    def _load(name):
        with open(f'{save_dir}/{name}.pkl', 'rb') as f:
            return pickle.load(f)

    out = {
        "y_true_ind_effects": _load('y_true_ind_effects'),
        "y_hat_baseline": _load('y_hat_baseline_ind_effects'),
        "y_hat_ind_effects": _load('y_hat_ind_effects'),
        "train_scenarios_ind_effects": ['Opt. Tier 1', 'Opt. DAMIP', 'Opt. GeoMIP', 'Opt. All'],
    }
    if seed_cache_path is not None:
        if Path(seed_cache_path).is_file():
            with open(seed_cache_path, 'rb') as f:
                out["seed_cache"] = pickle.load(f)
            print(f"load_fig6_data: seed spread loaded ({len(out['seed_cache'])} seeds) "
                  f"-> median + IQR")
        else:
            print(f"load_fig6_data: WARNING - {seed_cache_path} not found; falling back to the "
                  f"SINGLE-SEED figure. Build it with "
                  f"pipeline/07_results/07m_build_fig6_cache.py before reporting this panel.")
    return out

FIG6_OOD_SCENARIOS = ["H-ext-VLaer", "ssp534-over", "esm-bell-2000PgC"]

FIG6_OOD_CONFIG_LABELS = {
    "all": "Opt. All",
    "damip": "Opt. DAMIP",
    "geomip": "Opt. GeoMIP",
    "tier1": "Opt. Tier 1",
}


def load_fig6_ood_data(
    ood_dir: str = "data/SI_results/fig6_ood",
    scenarios: list[str] = FIG6_OOD_SCENARIOS,
) -> dict:
    """
    Load Figure 6's out-of-objective evaluation data (the out-of-objective evaluation):
    """
    ood_dir = Path(ood_dir)
    out: dict = {}

    scenario_cache_path = ood_dir / "fig6_ood_scenarios.pkl"
    if scenario_cache_path.is_file():
        with open(scenario_cache_path, "rb") as f:
            cache = pickle.load(f)
        out["ood_scenarios"] = {
            tag: {"years": np.asarray(cache["scenarios"][tag]["years"]),
                  "y_scm": np.asarray(cache["scenarios"][tag]["y_scm"])}
            for tag in scenarios
        }
    else:
        print(f"load_fig6_ood_data: WARNING - {scenario_cache_path} not found; "
              f"column 2 (OOD trajectories) cannot be built.")

    ood_seed_traj: dict[str, dict[str, np.ndarray]] = {tag: {} for tag in scenarios}
    for config, label in FIG6_OOD_CONFIG_LABELS.items():
        suffix = "" if config == "all" else f"_{config}"
        sweep_path = ood_dir / f"fig6_ood_seed_sweep{suffix}.pkl"
        if not sweep_path.is_file():
            print(f"load_fig6_ood_data: WARNING - {sweep_path} not found; "
                  f"'{label}' will be missing from column 2. Run "
                  f"pipeline/07_results/07i_fig6_ood_evaluate.py --mode collect --config {config}.")
            continue
        with open(sweep_path, "rb") as f:
            sweep = pickle.load(f)
        seeds = sorted(sweep)
        for tag in scenarios:
            ood_seed_traj[tag][label] = np.stack(
                [np.asarray(sweep[s]["optimized"][tag]["yhat"]) for s in seeds])
            ood_seed_traj[tag].setdefault(
                "Baseline Em.",
                np.stack([np.asarray(sweep[s]["baseline"][tag]["yhat"]) for s in seeds]),
            )
    if any(ood_seed_traj[tag] for tag in scenarios):
        out["ood_seed_traj"] = ood_seed_traj

    r2_table_path = ood_dir / "fig6_ood_r2_table_fig6.csv"
    if r2_table_path.is_file():
        with open(r2_table_path, newline="") as f:
            out["r2_table"] = list(csv.DictReader(f))
    else:
        print(f"load_fig6_ood_data: WARNING - {r2_table_path} not found; "
              f"column 3 (R^2 forest plot) cannot be built. Run "
              f"pipeline/07_results/07i_fig6_ood_evaluate.py --mode merge.")

    return out


def build_MESM_baseline_eval_sets(eval_dir: str = "data/MESM/emis_driven/zonal_data_mean/") -> dict:
    """
    Assemble the Tier 1/Tier 2/DECK/CS3 emissions and MESM zonal-temperature
    target sets used to evaluate the MESM vector (zonal-output) emulator, plus
    the Tier 1 training data for the baseline side specifically.
    """
    agents = ["CO2"]
    scenarios_eval = {
        "Tier 1": ["historical", "H-ext", "L", "M", "ML", "VLHO", "VLLO-ext"],
        "Tier 2": ["H-ext-OS", "M-ext", "ML-ext", "L-ext", "VLHO-ext"],
        "DECK": ["1pctCO2", "2xCO2"],
        "CS3": ["AA", "CT", "historical"],
    }

    (eval_emis_sets, emis_dict_tier1_JAX, emis_dict_tier2_JAX, emis_dict_CS3_JAX, emis_dict_all_JAX) = (
        generate_eval_data(agents, DECK=False, CS3=True, DAMIP=False, GeoMIP=False)
    )

    (eval_targets_sets, targets_dict_tier1, targets_dict_tier2, targets_dict_DECK, targets_dict_CS3, output_dim, lat_coords) = (
        generate_target_data(scenarios_eval, data_dir=eval_dir)
    )

    emis_path = str(DATA_DIR / "MESM" / "emis_driven")
    emis_1pct_path = f"{emis_path}/1PRCO2/carbemiss.txt"
    emis_1pct = np.loadtxt(emis_1pct_path, usecols=(2,), skiprows=2)
    emis_mat_1pct = np.zeros((5, len(emis_1pct)))
    emis_mat_1pct[0, :] = emis_1pct

    emis_2xCO2_path = f"{emis_path}/2xCO2/implco2emiss.3100.25.txt"
    emis_2xCO2 = np.loadtxt(emis_2xCO2_path, usecols=(2,))
    emis_mat_2xCO2 = np.zeros((5, len(emis_2xCO2)))
    emis_mat_2xCO2[0, :] = emis_2xCO2

    eval_emis_sets["DECK"] = {"1pctCO2": emis_mat_1pct, "2xCO2": emis_mat_2xCO2}

    return {
        "eval_emis_sets": eval_emis_sets,
        "eval_targets_sets": eval_targets_sets,
        "emis_dict_tier1_JAX": emis_dict_tier1_JAX,
        "targets_dict_tier1": targets_dict_tier1,
        "output_dim": output_dim,
        "lat_coords": lat_coords,
    }


def build_MESM_opt_eval_sets(
    eval_emis_sets: dict,
    eval_targets_sets: dict,
    ic_list: list[str],
    group: str = "all",
    eval_dir: str = "data/MESM/emis_driven/zonal_data_mean/",
) -> dict:
    """
    Build the CO2-only-optimized training set for the MESM vector emulator's
    "optimized" side
    """
    emis_dict_opt = {}
    for IC in ic_list:
        opt_path = f"checkpoints/co2/inverse_{IC}_{group}_co2_only_MESM.pkl"
        with open(opt_path, "rb") as f:
            res = pickle.load(f)
        co2_array = res["U_traj"][-1]["CO2"]
        emis_dict_opt[group + "_" + IC] = np.zeros((5, len(co2_array)))
        emis_dict_opt[group + "_" + IC][0, :] = co2_array.copy()

    eval_emis_opt_sets = {"optimized": emis_dict_opt.copy()}
    scenarios_train = {"optimized": [group + "_" + IC for IC in ic_list]}

    eval_targets_opt_sets, targets_dict_opt, _, _ = generate_target_data(
        scenarios_train, data_dir=eval_dir, opt=True
    )

    for key in eval_targets_sets:
        eval_targets_opt_sets[key] = eval_targets_sets[key].copy()
        eval_emis_opt_sets[key] = eval_emis_sets[key].copy()

    return {
        "eval_emis_opt_sets": eval_emis_opt_sets,
        "eval_targets_opt_sets": eval_targets_opt_sets,
        "emis_dict_opt": emis_dict_opt,
        "targets_dict_opt": targets_dict_opt,
        "scen_key": "optimized",
    }

# ==================================================================
# Part 5b: notebook-facing data-prep for pipeline/08_plotting/SI_plots.ipynb
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


def load_SI_ic_sensitivity_data_seed_sweep(
    seeds: list[int] = tuple(range(50)),
    checkpoint_dir: str = 'data/SI_results/sensitivity_initial_condition/seed_sweep',
    baseline_dir: str = 'checkpoints/co2_retuned/seed_sweep',
) -> dict:
    """
    Multi-seed companion to load_SI_ic_sensitivity_data
    """
    IC_list = ['constant', 'gaussian', 'sine']
    seed_result_paths, seed_baseline_errors = [], []
    for IC in IC_list:
        ckpt_paths, base_errs = [], []
        for seed in seeds:
            ckpt_path = f'{checkpoint_dir}/inverse_{IC}_all_co2_only_seed{seed}.pkl'
            baseline_path = f'{baseline_dir}/baseline_co2_only_seed{seed}.pkl'
            if not Path(ckpt_path).exists() or not Path(baseline_path).exists():
                raise FileNotFoundError(
                    f"{ckpt_path} or {baseline_path} missing - run "
                    f"pipeline/06_sensitivity/06a_sensitivity_seed_sweep.py --sweep ic --seed {seed} first"
                )
            ckpt_paths.append(ckpt_path)
            with open(baseline_path, "rb") as f:
                base_errs.append(pickle.load(f)['All']['mean'])
        seed_result_paths.append(ckpt_paths)
        seed_baseline_errors.append(base_errs)

    return {
        "seed_result_paths": seed_result_paths,
        "seed_baseline_errors": seed_baseline_errors,
        "column_titles": ['(a) Constant', '(b) Gaussian', '(c) Sinusoid'],
        "active_agents": ("CO2",),
        "save_path": 'SI_IC',
    }


def load_SI_architecture_sensitivity_data_seed_sweep(
    IC: str = 'sine',
    seeds: list[int] = tuple(range(50)),
    checkpoint_dir: str = 'data/SI_results/sensitivity_architecture/seed_sweep',
) -> dict:
    """
    Multi-seed companion to load_SI_architecture_sensitivity_data
    """
    arch_list = ['8', '16', '32', '16_16']
    seed_result_paths, seed_baseline_errors = [], []
    for arch in arch_list:
        ckpt_paths, base_errs = [], []
        for seed in seeds:
            ckpt_path = f'{checkpoint_dir}/inverse_{IC}_all_co2_only_{arch}_seed{seed}.pkl'
            baseline_path = f'{checkpoint_dir}/baseline_co2_only_{arch}_seed{seed}.pkl'
            if not Path(ckpt_path).exists() or not Path(baseline_path).exists():
                raise FileNotFoundError(
                    f"{ckpt_path} or {baseline_path} missing - run "
                    f"pipeline/06_sensitivity/06a_sensitivity_seed_sweep.py --sweep architecture "
                    f"--condition {arch} --seed {seed} first"
                )
            ckpt_paths.append(ckpt_path)
            with open(baseline_path, "rb") as f:
                base_errs.append(pickle.load(f)['All']['mean'])
        seed_result_paths.append(ckpt_paths)
        seed_baseline_errors.append(base_errs)

    return {
        "seed_result_paths": seed_result_paths,
        "seed_baseline_errors": seed_baseline_errors,
        "column_titles": ['(a) [8]', '(b) [16]', '(c) [32]', '(d) [16, 16]'],
        "active_agents": ("CO2",),
        "save_path": f'SI_arch_{IC}',
    }


def load_SI_feature_sensitivity_data_seed_sweep(
    IC: str = 'sine',
    seeds: list[int] = tuple(range(50)),
    checkpoint_dir: str = 'data/SI_results/sensitivity_features/seed_sweep',
) -> dict:
    """
    Multi-seed companion to load_SI_feature_sensitivity_data
    """
    feat_list = ['short', 'medium', 'long']
    seed_result_paths, seed_baseline_errors = [], []
    for feat in feat_list:
        ckpt_paths, base_errs = [], []
        for seed in seeds:
            ckpt_path = f'{checkpoint_dir}/inverse_{IC}_all_co2_only_{feat}_seed{seed}.pkl'
            baseline_path = f'{checkpoint_dir}/baseline_co2_only_{feat}_seed{seed}.pkl'
            if not Path(ckpt_path).exists() or not Path(baseline_path).exists():
                raise FileNotFoundError(
                    f"{ckpt_path} or {baseline_path} missing - run "
                    f"pipeline/06_sensitivity/06a_sensitivity_seed_sweep.py --sweep features "
                    f"--condition {feat} --seed {seed} first"
                )
            ckpt_paths.append(ckpt_path)
            with open(baseline_path, "rb") as f:
                base_errs.append(pickle.load(f)['All']['mean'])
        seed_result_paths.append(ckpt_paths)
        seed_baseline_errors.append(base_errs)

    return {
        "seed_result_paths": seed_result_paths,
        "seed_baseline_errors": seed_baseline_errors,
        "column_titles": ['(a) Short', '(b) Medium', '(c) Long'],
        "active_agents": ("CO2",),
        "save_path": f'SI_feat_{IC}',
    }


def regenerate_SI_extended_results_cache(agents: list[str] = ['N2O', 'Sulfur', 'BC']) -> None:
    """
    Recompute the per-agent optimal-emulator NRMSE summary (used by the SI
    extended-results stacked-bar figure) for every agent in `agents` and
    overwrite data/SI_results/extended_results/optimal_<agent>_only.pkl.
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
    kwargs for utils_plotting.plot_SI_extended_results
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


def load_SI_extended_results_data_seed_sweep(
    agent_lower_list: list[str] = ['co2', 'ch4', 'n2o', 'Sulfur', 'BC'],
    co2_cache_path: str = 'data/SI_results/seed_uncertainty/fig4_seed_spread_co2_only.pkl',
    other_cache_path_template: str = 'data/SI_results/seed_uncertainty/SI_extended_seed_spread_{agent}.pkl',
) -> dict:
    """
    Multi-seed companion to load_SI_extended_results_data
    """
    _agent_map = {'co2': 'CO2', 'ch4': 'CH4', 'n2o': 'N2O', 'Sulfur': 'Sulfur', 'BC': 'BC'}

    _cache_overrides = {
        'Sulfur': 'data/SI_results/seed_uncertainty/SI_extended_seed_spread_Sulfur_smooth.pkl',
    }
    seed_baseline_results_list, seed_optimized_results_list = [], []
    for agent_lower in agent_lower_list:
        agent = _agent_map[agent_lower]
        cache_path = (co2_cache_path if agent == 'CO2'
                      else _cache_overrides.get(agent, other_cache_path_template.format(agent=agent)))
        if not Path(cache_path).exists():
            raise FileNotFoundError(
                f"{cache_path} missing for agent {agent} - run "
                f"regenerate_SI_extended_results_cache_seed_sweep (or, for CO2, "
                f"regenerate_fig4_co2_only_cache_seed_sweep) first"
            )
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        seeds_sorted = sorted(cache)
        seed_baseline_results_list.append([cache[s]["baseline"] for s in seeds_sorted])
        seed_optimized_results_list.append([cache[s]["optimal"] for s in seeds_sorted])

    n = len(agent_lower_list)
    return {
        "baseline_results_list": [None] * n,   # unused - every panel is in seed mode
        "optimized_results_list": [None] * n,  # unused - every panel is in seed mode
        "seed_baseline_results_list": seed_baseline_results_list,
        "seed_optimized_results_list": seed_optimized_results_list,
        "train_scenarios": ['Opt. Tier 1', 'Opt. Tier 2', 'Opt. DECK', 'Opt. CS3', 'Opt. All'],
        "test_scenarios": ['Tier 1', 'Tier 2', 'DECK', 'CS3'],
        "x_labels": ['Opt. Priority 1', 'Opt. Priority 2', 'Opt. DECK', 'Opt. CS3', 'Opt. All'],
        "leg_labels": ['Priority 1', 'Priority 2', 'DECK', 'CS3'],
        "weights": [7, 5, 2, 2],
        "titles": None,
        "figname": 'SI_extended_results',
    }


# ==================================================================
# SI: representative-seed emissions comparison (CO2-only + multi-forcing)
# ==================================================================

def _pick_representative_seeds(final_errors_by_seed: dict) -> dict:
    """Pick the seeds whose final NRMSE lands closest to the 25th percentile,
    the median, and the 75th percentile of the across-seed distribution
    """
    values = list(final_errors_by_seed.values())
    q25_val, median_val, q75_val = (float(v) for v in np.percentile(values, [25, 50, 75]))
    q25 = min(final_errors_by_seed, key=lambda s: abs(final_errors_by_seed[s] - q25_val))
    median = min(final_errors_by_seed, key=lambda s: abs(final_errors_by_seed[s] - median_val))
    q75 = min(final_errors_by_seed, key=lambda s: abs(final_errors_by_seed[s] - q75_val))
    return {"q25": q25, "median": median, "q75": q75}


def _load_final_emissions_state(checkpoint_dir: str, tag: str, seed: int, group: str = 'tier1') -> dict:
    """Load just the converged (final-iteration) emissions state U_traj[-1]
    for one seed/group checkpoint, keyed by agent (e.g. 'CO2', 'CH4',...).
    """
    path = f'{checkpoint_dir}/inverse_constant_{group}_{tag}_seed{seed}.pkl'
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return _tree_to_jnp(raw["U_traj"][-1])


def _final_errors_from_fig4_cache(cache_path: str, seeds: list[int], train_scenario: str) -> dict:
    """{seed: NRMSE} for one training column of a Figure-4 seed-spread cache
    (data/SI_results/seed_uncertainty/fig4_seed_spread_{co2_only,
    all_agents_smooth}.pkl, built by regenerate_fig4_{co2_only,all_agents}_
    cache_seed_sweep)
    """
    if not Path(cache_path).exists():
        raise FileNotFoundError(
            f"{cache_path} missing - run scripts/build_fig4_seed_spread_cache_"
            f"{'co2.py' if 'co2' in cache_path else 'multi.py --checkpoint-dir checkpoints/multi_fig4_smooth/seed_sweep --suffix _smooth'} first"
        )
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    missing = [s for s in seeds if s not in cache]
    if missing:
        raise FileNotFoundError(f"{cache_path} missing {len(missing)} of {len(seeds)} seeds (first few: {missing[:5]})")
    return {s: float(cache[s]['optimal'][train_scenario]['All']['mean']) for s in seeds}


def load_SI_seed_emissions_comparison_data(
    seeds: list[int] = tuple(range(50)),
    co2_checkpoint_dir: str = 'checkpoints/co2_retuned/seed_sweep',
    co2_tag: str = 'co2_only',
    co2_seed_spread_cache: str = 'data/SI_results/seed_uncertainty/fig4_seed_spread_co2_only.pkl',
    multi_checkpoint_dir: str = 'checkpoints/multi_fig4_smooth/seed_sweep',
    multi_tag: str = 'multi_fig4',
    multi_seed_spread_cache: str = 'data/SI_results/seed_uncertainty/fig4_seed_spread_all_agents_smooth.pkl',
    multi_agents: list[str] = ('CO2', 'CH4', 'N2O', 'Sulfur', 'BC'),
    train_scenario: str = 'Opt. All',
    group: str = 'all',
) -> dict:
    """
    Data for the SI seed-emissions-comparison figure
    (utils_plotting.plot_seed_emissions_comparison)

    Returns:
        {
          'co2_emissions': {'q25': arr, 'median': arr, 'q75': arr},
          'multi_emissions': {'CO2': {...}, 'CH4': {...}, 'N2O': {...},
                               'Sulfur': {...}, 'BC': {...}},
          'co2_seed_info': {'q25': (seed, nrmse), 'median': (...), 'q75': (...)},
          'multi_seed_info': {same shape as co2_seed_info},
        }
    """
    co2_final = _final_errors_from_fig4_cache(co2_seed_spread_cache, seeds, train_scenario)
    co2_pick = _pick_representative_seeds(co2_final)

    multi_final = _final_errors_from_fig4_cache(multi_seed_spread_cache, seeds, train_scenario)
    multi_pick = _pick_representative_seeds(multi_final)

    co2_emissions = {}
    for label, seed in co2_pick.items():
        state = _load_final_emissions_state(co2_checkpoint_dir, co2_tag, seed, group=group)
        co2_emissions[label] = np.asarray(state['CO2'])

    multi_emissions = {a: {} for a in multi_agents}
    for label, seed in multi_pick.items():
        state = _load_final_emissions_state(multi_checkpoint_dir, multi_tag, seed, group=group)
        for a in multi_agents:
            multi_emissions[a][label] = np.asarray(state[a])

    return {
        "co2_emissions": co2_emissions,
        "multi_emissions": multi_emissions,
        "co2_seed_info": {label: (seed, co2_final[seed]) for label, seed in co2_pick.items()},
        "multi_seed_info": {label: (seed, multi_final[seed]) for label, seed in multi_pick.items()},
    }


def load_SI_scm_mesm_fidelity_panels() -> list[dict]:
    """
    Panels for the supplement's SCM-vs-MESM fidelity figure, cached by
    `pipeline/07_results/07e_scm_mesm_fidelity.py --panels`.
    """
    path = 'data/SI_results/scm_mesm_fidelity/si_scm_mesm_panels.pkl'
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run "
            f"`python pipeline/07_results/07e_scm_mesm_fidelity.py --panels` first.")
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_SI_baseline_convergence_curves() -> np.ndarray:
    """
    (n_seeds, K) baseline training-loss curves behind the supplement's
    convergence figure, written by `pipeline/03_hyperparameters/03d_baseline_convergence.py`.
    """
    path = 'data/SI_results/baseline_hp/convergence_curves_K400.npy'
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run "
            f"`python pipeline/03_hyperparameters/03d_baseline_convergence.py` first.")
    return np.load(path)
