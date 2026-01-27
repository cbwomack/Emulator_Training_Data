# -------
# Imports
# -------
import utils_FaIR_JAX
import numpy as np
import matplotlib.pyplot as plt
import pickle

# JAX
import jax
import jax.numpy as jnp
from jax import lax
import optax
from jax import tree as _jtree

import os, pickle, numpy as np
from functools import partial

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

def _as_jnp(x, dtype=jnp.float32):
    return jnp.asarray(x, dtype=dtype)

def _prev_and_cumu_prev(E_curr, E_hist=None):
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

def _ema(x, alpha):
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

def _ema_prev(E_curr, E_hist, alpha):
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
    emis_curr_dict,              # dict: agent -> (T,) emissions
    emis_hist_dict=None,         # dict or None: agent -> (H,) emissions
    agents=AGENTS_DEFAULT,        # tuple/list of agents to include (order = column grouping)
    ema_windows_years=(5.0, 30.0, 100.0),
    dt_years=1.0,
    zero_fill_missing=True,
):
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
    yrs = _as_jnp(yrs).reshape(-1)
    if yrs.size <= 1:
        return jnp.array(1.0, dtype=jnp.float32)
    return jnp.median(jnp.diff(yrs))

def _make_contiguous_years(yrs_hist: jnp.ndarray, yrs_curr: jnp.ndarray):
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

def _stack_emissions(agents, emis_dict, T, dtype=jnp.float32):
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
    years_curr,                         # (T_cur,)
    emis_curr_dict,                     # dict: {"CO2": (T_cur,), "CH4": (T_cur,), ...}
    years_hist=None,                    # (T_hist,) or None
    emis_hist_dict=None,                # dict or None: {"CO2": (T_hist,), "CH4": (T_hist,), ...}
    agents=AGENTS_DEFAULT,              # tuple/list: which agents to include and their row order
    mode='FaIR',
    dt=0.1
):
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

def extract_years_and_emis(emis_entry_for_scenario, agents=AGENTS_DEFAULT):
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
    emis_dict,
    historical_name="historical",
    agents=AGENTS_DEFAULT,
    mode='FaIR'
):
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
            ema_windows_years=(5.0, 30.0, 100.0),
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

def fit_scaler(X: jnp.ndarray, eps: float = 1e-8):
    mu = jnp.mean(X, axis=0)
    sd = jnp.sqrt(jnp.var(X, axis=0) + eps)
    Xs = (X - mu) / sd
    return Xs, (mu, sd)

def apply_scaler(X: jnp.ndarray, stats):
    mu, sd = stats
    return (X - mu) / sd

def _infer_feat_dim(train_dataset, test_dataset):
    for ds in (train_dataset, test_dataset):
        for (X, *_rest) in ds:
            return int(X.shape[1])
    return 0

def split_and_scale(train_dataset, test_dataset):
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

def init_mlp_params(key, input_dim, hidden_sizes):
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

def mlp_forward(params, X):
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

def _mse(pred, y):
    pred = pred.astype(y.dtype)
    return jnp.mean((pred - y)**2)

_tree_map    = _jtree.map
def train_mlp_sgd(params0, Xtr, ytr, K=400, lr=5e-2, weight_decay=1e-2):
    pdt = params0[0]["W"].dtype
    Xtr = Xtr.astype(pdt); ytr = ytr.astype(pdt)

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
            yhat = mlp_forward(p, Xtr)
            return _mse(yhat, ytr)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = _tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6), grads)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        return (params, opt_state), loss

    (paramsK, _), losses = jax.lax.scan(step, (params0, opt_state), xs=None, length=K)
    return paramsK, losses

def eval_on_tests_mlp(params, test_scaled_list):
    if not test_scaled_list:
        return jnp.array(0.0, dtype=jnp.float32)

    errs = [_mse(mlp_forward(params, X), y) for (X, y, _) in test_scaled_list]

    return jnp.mean(jnp.stack(errs))

def plot_mlp_predictions(params, Xs, y, metric="NRMSE", title_prefix="MLP fit"):
    yhat = mlp_forward(params, Xs)
    loss_val = _nrmse(yhat, y)

    plt.figure(figsize=(8,3))
    plt.plot(np.asarray(y),    label="truth", alpha=0.8)
    plt.plot(np.asarray(yhat), label="pred",  alpha=0.8)
    plt.legend(); plt.xlabel("time step"); plt.ylabel("target")
    plt.title(f"{title_prefix} - {metric.upper()}: {loss_val:.4f}")
    plt.tight_layout(); plt.show()

def freeze_inactive_agents(emis_dict, active_agents=("CO2",), agents=AGENTS_DEFAULT):
    out = {}
    for a in agents:
        x = emis_dict[a]
        out[a] = x if (a in active_agents) else jax.lax.stop_gradient(x)
    return out

def _apply_active_mask_to_emis(U_pytree, active_agents, inactive_mode="zeros"):
    def squash(u, active):
        if active:
            return u
        if inactive_mode == "stop_grad_zeros":
            return jax.lax.stop_gradient(jnp.zeros_like(u))
        # default: pure zeros (still fine if we also mask grads/updates):
        return jnp.zeros_like(u)
    return {ag: squash(U_pytree[ag], ag in active_agents) for ag in U_pytree}

# Reuse the helper from earlier to accept dict OR (N_agents, T) arrays
def _normalize_emissions_input(U_in, agents, T, dtype=jnp.float32):
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
    U_in,                              # dict {agent:(T,)} OR array (N_agents,T)
    agents=AGENTS_DEFAULT,
    ema_windows_years=(5.0, 30.0, 100.0),
    dtype=jnp.float32,
    years_hist=None,
    emis_hist_dict=None,
    mode='FaIR'
):
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

def build_dataset_no_history(emis_dict, scenarios, agents=AGENTS_DEFAULT, mode='FaIR'):
    """
    Build features/targets without prepending historical emissions.
    Uses a sentinel historical_name that won't be found in the dict.
    """
    return build_dataset_from_runfair_dict(
        emis_dict, scenarios, historical_name="__NONE__", agents=agents, mode=mode
    )

def build_valid(
    emis_dict_valid,
    historical_name="historical",
    agents=AGENTS_DEFAULT,
    mode='FaIR'
):
    """
    Combined validation set:
      - Tier-1 with historical context (where available)
      - Tier-2 without adding a historical sample (skip_hist=True)
      - DECK subset with NO historical context
    """
    valid = build_dataset_from_runfair_dict(
        emis_dict_valid,
        historical_name=historical_name, agents=agents, mode=mode
    )
    return valid

def make_objective_over_emissions(
    scen_name_tr1,
    train_data, test_data,
    emis_dict,
    params0,
    historical_name="historical",
    dtype=jnp.float32,
    agents=AGENTS_DEFAULT,
    ema_windows_years=(5.0, 30.0, 100.0),
    active_agents=("CO2",),
    mode='FaIR'
):
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
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Constant initial emissions."""
    return jnp.ones(T, dtype=dtype) * emis_val

def init_ramp_emissions(
    T: int,
    final_val: float = 50.0,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Ramp initial emissions."""
    return jnp.linspace(0, int(final_val), T)

def init_gaussian_emissions(
    T: int,
    peak_emis: float = 50.0,
    mu: float = 350.0,
    sigma: float = 100.0,
    floor: float = 0.0,
    dtype=jnp.float32,
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
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Ramp initial emissions."""
    t = jnp.arange(T, dtype=dtype)
    return peak_emis * jnp.sin(2 * jnp.pi * t / period).astype(dtype)


def make_init_emissions_pytree(
    T: int,
    agents=AGENTS_DEFAULT,
    active_agents=None,
    init_cond=None,               # str (keyword) | array-like (matrix) | None
    inactive_mode: str = "zeros", # "zeros" or "stop_grad_zeros"
    dtype=jnp.float32,
):
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
        "CO2":    {"peak": 50.0,  "mu": 350, "sig_frac": 0.15, "period": 150},
        "CH4":    {"peak": 500.0, "mu": 450, "sig_frac": 0.25, "period": 150},
        "N2O":    {"peak": 10.0,  "mu": 350, "sig_frac": 0.15, "period": 150},
        "Sulfur": {"peak": 60.0,  "mu": 350, "sig_frac": 0.15, "period": 150},
        "BC":     {"peak": 10.0,  "mu": 350, "sig_frac": 0.15, "period": 150},
        # Fallback for unknown agents
        "DEFAULT": {"peak": 50.0, "mu": 350, "sig_frac": 0.15, "period": 150},
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
    x = jnp.linspace(0.0, 1.0, T)
    w = x ** power
    return min_scale + (1.0 - min_scale) * w  # (T,)

def make_time_weights_pytree(T: int, agents=AGENTS_DEFAULT, power=2.0, min_scale=0.15):
    w = make_time_weights(T, power=power, min_scale=min_scale)
    return {a: w for a in agents}

def avg_nrmse_over_tests(params, test_list, eps: float = 1e-8):
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
    params0,
    test_dataset_all,
    K_inner=400,
    lr_inner=5e-2,
    wd_inner=1e-2,
    agents=AGENTS_DEFAULT,
    active_agents=("CO2",),
    inactive_mode="zeros",
    smoothness_weight=0.0,
    mode='FaIR'
):
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
            ema_windows_years=(5.0, 30.0, 100.0),
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
            params0, Xtr, ytr, K=K_inner, lr=lr_inner, weight_decay=wd_inner
        )

        # 5) Average NRMSE across all scenarios, weighted by scenario length
        nrmse_val = avg_nrmse_over_tests(paramsK, test_s)
        loss = nrmse_val + (smoothness_weight * reg_loss)

        aux  = (paramsK, test_s, train_temp_raw)
        return loss, aux
    return objective


def _tree_to_numpy(tree):
    return jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x)) if isinstance(x, (jnp.ndarray, jax.Array)) else x, tree
    )

def _tree_to_jnp(tree):
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, tree
    )

def save_inverse_ckpt(path, state):
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

def load_inverse_ckpt(path):
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

def scale_by_coord_pytree(weights_pytree):
    def _mul(g, w): return g * w
    def init_fn(_): return ()
    def update_fn(updates, state, params=None):
        return jax.tree_util.tree_map(_mul, updates, weights_pytree), state
    return optax.GradientTransformation(init_fn, update_fn)

def preds_by_scenario(params, test_list):
    out = []
    for (Xte, yte, scen) in test_list:
        yhat = mlp_forward(params, Xte)
        out.append((scen, yhat, yte))
    return out

def make_agent_mask_pytree(T, agents, active_agents):
    one  = lambda: jnp.ones((T,), jnp.float32)
    zero = lambda: jnp.zeros((T,), jnp.float32)
    return {ag: (one() if ag in active_agents else zero()) for ag in agents}

def mul_tree(a, b):
    return jax.tree_util.tree_map(lambda x, m: x * m, a, b)

def optimize_emissions_inverse(
    emis_dict,
    params0,
    num_updates=500,
    step_size=1e3,
    momentum=0.9,
    nesterov=True,
    K_inner=500,
    lr_inner=5e-2,
    wd_inner=1e-2,
    historical_name="historical",
    agents=AGENTS_DEFAULT,
    active_agents=None,
    init_cond=None,
    inactive_mode="zeros",
    T: int = 750,
    n_nonneg_prefix: int | None = None,
    filter_hist: bool = False,
    smoothness_weight=0.0,
    mode='FaIR',
    checkpoint_path=None,
    checkpoint_every=50,
    resume_if_exists=True,
    preds_every=50,
):
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
        mode=mode
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
        mode=mode
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
        if not resume_if_exists and os.path.isfile(checkpoint_path):
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

def plot_inverse_results(
    results,
    baseline_error,                  # 1) dashed reference line
    active_agents=None,              # 4) only plot these agents’ emissions
    agent_units=None,                # optional: {'CO2':'GtCO₂/yr','CH4':'MtCH₄/yr'}
    max_lines=11,                    # 7) up to 12 emissions curves
    pred_scenario=None,
    baseline_preds=None,
):
    """
    Multipanel plot:
      (a) NRMSE vs update step
      (b) Optimal emissions profiles (one subplot per active agent)
      (c) Training temperature trajectory
      (d) Predictions vs Truth
    """
    errors = jnp.asarray(results["errors"])
    U_traj = results["U_traj"]
    preds_traj = results.get("preds_traj", [])

    # --- helpers to read U_traj --------------------------------
    def _agents_in_state(state):
        if isinstance(state, (tuple, list)) and len(state) == 2:
            return ("CO2", "CH4")
        if isinstance(state, dict):
            return tuple(state.keys())
        raise ValueError("Unrecognized U_traj state format.")

    def _get_series(state, agent):
        if isinstance(state, (tuple, list)):
            if agent == "CO2": return jnp.asarray(state[0]).reshape(-1)
            if agent == "CH4": return jnp.asarray(state[1]).reshape(-1)
            raise KeyError(f"Agent {agent} not found in tuple state.")
        return jnp.asarray(state[agent]).reshape(-1)

    # Figure out which agents exist in this run
    present_agents = _agents_in_state(U_traj[0])
    if active_agents is None:
        active_agents = present_agents
    else:
        active_agents = tuple(a for a in active_agents if a in present_agents)

    # Units
    if agent_units is None:
        agent_units = {"CO2": "Gt/yr", "CH4": "Mt/yr", "N2O": "Mt/yr", "Sulfur":"Mt/yr", "BC": "Mt/yr"}

    n_update_steps = int(results.get("updates_done", max(0, errors.shape[0] - 1)))

    # years length
    if len(active_agents) == 0:
        probe_agent = present_agents[0]
    else:
        probe_agent = active_agents[0]
    T = int(_get_series(U_traj[-1], probe_agent).shape[0])

    # --- DYNAMIC SUBPLOTS ---
    # Rows: 1 (NRMSE) + N (Agents) + 1 (TrainTemp) + 1 (Preds)
    n_agents = len(active_agents)
    n_rows = 3 + n_agents

    # Adjust figure height based on number of rows to maintain aspect ratio
    fig_height = 2.5 * n_rows
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=1,
        figsize=(9, fig_height),
        constrained_layout=True
    )

    # Handle single axis case (unlikely but safe)
    if n_rows == 1: axes = [axes]

    # Assign axes
    ax_err = axes[0]
    ax_emis_list = axes[1 : 1 + n_agents] # Slice for emissions
    ax_train = axes[-2]
    ax_pred = axes[-1]

    # Share x-axis between all emissions plots and the training temperature
    for ax in ax_emis_list:
        ax.sharex(ax_train)

    # ----- Panel (a): scaled RMSE vs update step --------------------------------
    x_err = jnp.arange(errors.shape[0])
    ax_err.semilogy(x_err, errors, label="NRMSE")
    ax_err.axhline(float(baseline_error), ls="--", c='r', lw=1.5, label=f"Baseline emulator (avg.)")
    ax_err.set_xlim(0, n_update_steps)
    ax_err.set_xlabel("Update step")
    ax_err.set_ylabel("NRMSE")
    ax_err.text(
        0.015, 0.93, "NRMSE vs. Update Step", transform=ax_err.transAxes,
        ha="left", va="top", fontsize=16, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
    )
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best")

    # ----- Panel (b): Optimal emissions (One subplot per agent) -----------------
    n_total = len(U_traj)
    if n_total <= max_lines:
        sel_steps = np.arange(n_total, dtype=int)
    else:
        sel_steps = np.unique(np.linspace(0, n_total - 1, num=max_lines, dtype=int))

    # Iterate over agents and their corresponding axes
    for idx, ag in enumerate(active_agents):
        ax_curr = ax_emis_list[idx]
        has_any = False

        for i in sel_steps:
            state = U_traj[i]
            alpha = 0.3 + 0.7 * (i / max(1, len(U_traj) - 1))
            series = _get_series(state, ag)
            ax_curr.plot(series, alpha=alpha, label=f"Step {i}")
            has_any = True

        ax_curr.set_xlim(0, T)
        unit = agent_units.get(ag, "units/yr")
        ax_curr.set_ylabel(f"{ag} ({unit})")

        ax_curr.text(
            0.015, 0.93, f"Optimal {ag}", transform=ax_curr.transAxes,
            ha="left", va="top", fontsize=16, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.5)
        )

        ax_curr.grid(True, alpha=0.3)
        # Turn off x-tick labels for all emissions plots (shared with bottom)
        ax_curr.tick_params(axis="x", labelbottom=False)

        if has_any and idx == 0:
            # Only put legend on the first emissions plot to avoid clutter
            ax_curr.legend(ncol=3, fontsize=8, loc="best")

    # ----- Panel (c): Training temperature trajectory -------------------------
    train_temp_traj = results.get("train_temp_traj", [])
    if len(train_temp_traj) > 0:
        M_all = len(train_temp_traj)
        if M_all <= max_lines:
            sel_train = np.arange(M_all, dtype=int)
        else:
            sel_train = np.unique(np.linspace(0, M_all - 1, num=max_lines, dtype=int))

        for k, i in enumerate(sel_train):
            temp_list = train_temp_traj[i]
            if temp_list is None or len(temp_list) == 0: continue
            y_train = np.asarray(temp_list[0])
            if y_train.ndim > 1: y_plot = y_train[:, 0]
            else: y_plot = y_train

            alpha = 0.3 + 0.7 * (k / max(1, len(sel_train) - 1))
            ax_train.plot(y_plot, alpha=alpha, label=f"Step {i}")

        ax_train.set_xlabel("Year")
        ax_train.set_ylabel(r"$\overline{\Delta T}(t)$ ($^\circ$C)")
        ax_train.grid(True, alpha=0.3)
        ax_train.set_xlim(0, T)
        ax_train.text(
            0.015, 0.93, r"$\overline{\Delta T}(t)$ from Optimal Emissions", transform=ax_train.transAxes,
            ha="left", va="top", fontsize=16, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.5)
        )

    # ----- Panel (d): Predictions vs Truth ------------------------------------
    if pred_scenario is None:
        target_scen = preds_traj[0][0][0] if len(preds_traj) > 0 else "Unknown"
    else:
        target_scen = pred_scenario

    def _find_scen_idx(step_list, name):
        for j, (sc, _, _) in enumerate(step_list):
            if sc == name: return j
        return None

    N_all = len(preds_traj)
    if N_all > 0:
        if N_all <= max_lines:
            sel_pred = np.arange(N_all, dtype=int)
        else:
            sel_pred = np.unique(np.linspace(0, N_all - 1, num=max_lines, dtype=int))

        last_ytrue = None
        for k, i in enumerate(sel_pred):
            step_list = preds_traj[i]
            j = _find_scen_idx(step_list, target_scen)
            if j is None: continue
            _, yhat, ytrue = step_list[j]
            yhat, ytrue = jnp.asarray(yhat), jnp.asarray(ytrue)

            alpha = 0.3 + 0.7 * (k / max(1, len(sel_pred) - 1))
            ax_pred.plot(yhat, alpha=alpha, label=(f"Step {i}"))
            last_ytrue = ytrue

        if last_ytrue is not None:
            ax_pred.plot(last_ytrue, ls="--", c="C3", label=f"truth: {target_scen}")
            if baseline_preds is not None:
                ax_pred.plot(baseline_preds, ls="-.", c="C2", label=f"Baseline Emulator")
            ax_pred.set_xlim(0, int(last_ytrue.shape[0]) - 1)

    ax_pred.text(
        0.015, 0.93, f"Predictions vs Truth ({target_scen})", transform=ax_pred.transAxes,
        ha="left", va="top", fontsize=16, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.5)
    )
    ax_pred.set_xlabel("Year")
    ax_pred.set_ylabel(r"$\overline{\Delta T}(t)$ ($^\circ$C)")
    ax_pred.grid(True, alpha=0.3)
    ax_pred.legend(ncol=3, loc="best", fontsize=8)

    return

def prepare_baseline_data(
    emis_dict_train,
    emis_dict_test,
    historical_name="historical",
    mode='FaIR'
):
    """Build baseline train/test datasets with the same feature construction as inverse."""
    train_data = build_dataset_from_runfair_dict(emis_dict_train, historical_name=historical_name, mode=mode)
    test_data  = build_dataset_from_runfair_dict(emis_dict_test, historical_name=historical_name, mode=mode)
    # scale (fit on train, apply to test) — same as inverse
    train_s, test_s, stats = split_and_scale(train_data, test_data)
    return train_s, test_s, stats

def train_baseline_emulator(
    train_scaled,                 # list[(X_s, y, scen), ...]
    key,
    hidden_sizes=[16],
    K=400,
    lr=5e-2,
    weight_decay=1e-2,
    dtype=jnp.float32
):
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
    params,
    dataset,          # list[(X, y, scen)]
    stats             # (mu, sd) from fit on the baseline train set
):
    # Apply the SAME scaler used in training
    test_scaled = []
    for (X, y, scen) in dataset:
        Xs = apply_scaler(jnp.asarray(X, dtype=jnp.float32), stats)
        test_scaled.append((Xs, jnp.asarray(y, dtype=jnp.float32), scen))
    return avg_nrmse_over_tests(params, test_scaled)

def _nrmse(yhat: jnp.ndarray, ytrue: jnp.ndarray, eps: float = 1e-8):
    max_abs = jnp.maximum(jnp.max(jnp.abs(ytrue)), eps)
    return jnp.sqrt(jnp.mean((yhat - ytrue) ** 2)) / max_abs

# --- helper: apply train stats to a test dataset list[(X, y, scen)] ---------
def _apply_stats_to_test(test_dataset, stats):
    out = []
    for (X, y, scen) in test_dataset:
        Xs = apply_scaler(jnp.asarray(X, jnp.float32), stats)
        out.append((Xs, jnp.asarray(y, jnp.float32), scen))
    return out

# --- main wrapper ------------------------------------------------------------
def evaluate_baseline_over_multiple_tests(
    emis_dict_train,
    eval_sets: dict,              # {"H": emis_dict_test_H, "S": emis_dict_test_S, ...}
    historical_name="historical",
    key=jax.random.PRNGKey(0),
    hidden_sizes=[16],
    K=400,
    lr=5e-2,
    weight_decay=1e-2,
    mode='FaIR'
):
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
            emis_dict_test, historical_name=historical_name, mode=mode
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

def create_baseline(train_s, test_s, hidden_sizes=[16], idx_demo=None, verbose=False):

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

    key    = jax.random.PRNGKey(0)
    input_dim = Xtr.shape[1]

    params0 = init_mlp_params(key, input_dim=input_dim, hidden_sizes=hidden_sizes)

    train_mlp_sgd_jit = jax.jit(train_mlp_sgd, static_argnames=("K",))
    paramsK, losses = train_mlp_sgd_jit(params0, Xtr, ytr, K=400, lr=5e-2, weight_decay=1e-2)

    if idx_demo is not None:
        (Xs_demo, y_demo, scen_demo) = test_s[idx_demo]
        plot_mlp_predictions(paramsK, Xs_demo, y_demo, metric="RMSE", title_prefix=f"{scen_demo}")

    rmse_list = eval_test_nrmse_by_scenario(paramsK, test_s)
    avg_rmse  = jnp.mean(jnp.stack([r for (_, r) in rmse_list]))

    if verbose:
        # Print results
        for name, r in rmse_list:
            print(f"{name:30s}  RMSE = {float(r):.6f}")
        print(f"\nAverage test RMSE across {len(rmse_list)} scenarios: {float(avg_rmse):.6f}")

    return params0

def test_get_grad(scen, params0, emis_dict_JAX, train_data, test_data, agents=AGENTS_DEFAULT, active_agents=('CO2'), mode='FaIR'):

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
    params0: dict=None,
    agents=AGENTS_DEFAULT,
    active_agents=None,
    inactive_mode="zeros",
    historical_name="historical",
    key=jax.random.PRNGKey(0),
    K=400,
    lr=5e-2,
    weight_decay=1e-2,
    mode='FaIR',
    ind_effects=False
):
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
            ema_windows_years=(5.0, 30.0, 100.0),
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
                emis_dict_test, historical_name=historical_name
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

def CS3_hist_modifier(scen_name, years_hist, emis_hist_dict):
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

def generate_init_params_and_train_data(agents, active_agents, test_scen, hidden_sizes=[16], idx_demo=None, verbose=False, mode='FaIR'):

    emis_dict_train_FaIR, emis_dict_test_FaIR, emis_dict_train_JAX, emis_dict_test_JAX, delT_dict_train_FaIR, delT_dict_test_FaIR, delT_dict_train_JAX, delT_dict_test_JAX = utils_FaIR_JAX.generate_train_test(agents, mode=mode)

    train_data = build_dataset_from_runfair_dict(emis_dict_train_JAX, mode=mode)
    test_data = build_dataset_from_runfair_dict(emis_dict_test_JAX, mode=mode)
    train_s, test_s, stats = split_and_scale(train_data, test_data)

    params0 = create_baseline(train_s, test_s, hidden_sizes=hidden_sizes, idx_demo=idx_demo, verbose=verbose)
    test_get_grad(test_scen, params0, emis_dict_train_JAX, train_data, test_data, active_agents=active_agents, mode=mode)

    return params0, emis_dict_train_JAX

def generate_eval_data(agents, CS3=False, DAMIP=False, GeoMIP=False):

    eval_data = utils_FaIR_JAX.generate_JAX_data(agents, CS3=CS3, DAMIP=DAMIP, GeoMIP=GeoMIP)

    keys = ["Tier 1", "Tier 2", "DECK"]
    if CS3: keys.append("CS3")
    if DAMIP: keys.append("DAMIP")
    if GeoMIP: keys.append("GeoMIP")

    # TEMPORARY!
    if len(agents) == 5:
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

def generate_and_eval_baseline_emulator(emis_dict_train, eval_sets, save_path=None, verbose=False, hidden_sizes=[16], mode='FaIR'):

    paramsK_base, baseline_results, baseline_pred_delT, ground_truth_delT = evaluate_baseline_over_multiple_tests(
    emis_dict_train=emis_dict_train,
    eval_sets=eval_sets,
    historical_name="historical",
    key=jax.random.PRNGKey(0),
    hidden_sizes=hidden_sizes,
    K=400,
    lr=5e-2,
    weight_decay=1e-2,
    mode=mode
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

def plot_baseline_pred_delT(baseline_results, baseline_pred_delT, ground_truth_delT):
    test_set = 'All'
    N_scens = len(baseline_results[test_set])
    rows = int(np.ceil(N_scens / 3))
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5*rows), constrained_layout=True)
    axes = axes.ravel()

    for i, scen in enumerate(baseline_results[test_set]):
        if scen == 'mean':
            continue
        ax = axes[i]
        ax.plot(ground_truth_delT[test_set][scen],  label="truth", alpha=0.9)
        ax.plot(baseline_pred_delT[test_set][scen],  label="prediction", ls="--", alpha=0.9)
        ax.set_title(f"{scen} - NRMSE={baseline_results[test_set][scen]:.3f}")
        ax.set_xlabel("time step")
        ax.set_ylabel("GMST (K)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    return