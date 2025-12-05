# Imports
import random
from tokenize import group
import numpy as np
import matplotlib.pyplot as plt

import run_fair

# Scitkit-learn
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Tensorflow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW, SGD

import seaborn as sns
from cmcrameri import cm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## Setup plots
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "sans-serif",
  "font.sans-serif": ["Helvetica Light"],
})

def plot_yhats_grid(error_dict, yhat_dict, ytrue_dict):
  train_scenarios = sorted(error_dict.keys())
  test_scenarios = sorted({test for train in train_scenarios for test in error_dict[train].keys()})
  n_test = len(test_scenarios)
  cols = int(np.ceil(np.sqrt(n_test))); rows = int(np.ceil(n_test / cols))
  fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), constrained_layout=True, sharex=True, sharey=True)

  axes_flat = np.atleast_1d(axes).flatten()

  for i, test in enumerate(test_scenarios):
    ax = axes_flat[i]
    for j, train in enumerate(train_scenarios):
      if train == 'noise':
        ls = '--'
      else:
        ls = '-'
      ax.plot(yhat_dict[train][test], lw=1, alpha=0.8, ls=ls, label=train, c=cm.batlowKS(j))  # all yhats for this test
    ax.plot(ytrue_dict[test], lw=2, alpha=0.8, label='True', c='k')
    ax.legend(loc='best', fontsize=8, ncols=2)

    ax.set_title(f"Test {test}")
    ax.set_ylim([-5,10])

    row = i // cols
    col = i % cols

    if col == 0:
      ax.set_ylabel("yhat")
    if row == rows - 1:
      ax.set_xlabel('t')

  for k in range(n_test, rows*cols):
    axes_flat[k].axis('off')  # hide empties

  return

def rmse(y_true, y_pred):
  y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
  return float(np.sqrt(np.mean(np.power(np.subtract(y_true, y_pred), 2))))

def nrmse(y_true, y_pred):
  denom = max(np.max(np.abs(y_true)), 1e-12)  # normalize by largest magnitude in y_test
  return rmse(y_true/denom, y_pred/denom)

def plot_error_heatmap(error_dict, train_scenarios, test_scenarios):
  # Build matrix with rows=test, cols=train
  E = np.full((len(test_scenarios), len(train_scenarios)), np.nan)
  for r, te in enumerate(test_scenarios):
    for c, tr in enumerate(train_scenarios):
      E[r, c] = error_dict.get(tr, {}).get(te, np.nan)

  avg_error_row = np.mean(E, axis=0)
  E_all = np.vstack([E, avg_error_row])
  yticklabels = test_scenarios.copy() + ['Avg.']

  # Plot
  fig, ax = plt.subplots(figsize=(1.2*len(train_scenarios)+2, 1.0*len(test_scenarios)+2), constrained_layout=True)
  sns.heatmap(E_all, ax=ax, vmin=0, vmax=1, cmap=cm.lajolla_r,
              xticklabels=train_scenarios, yticklabels=yticklabels,
              annot=True, fmt=".2g", cbar_kws={"label": "RMSE"})
  ax.set_xlabel("Train scenario")
  ax.set_ylabel("Test scenario")
  ax.set_title("Error heatmap")
  return

def project(v, u):
  """
  Calculates the projection of vector v onto vector u.
  proj_u(v) = (<v,u>/<u,u>) * u
  """
  # Avoid division by zero if u is the zero vector
  u_dot_u = np.dot(u, u)
  if u_dot_u == 0:
    return np.zeros_like(v)
  return (np.dot(v, u) / u_dot_u) * u

def generate_uncorrelated_signals(n, signal_length=1000):
  """
  Generates n uncorrelated white noise signals of a specified length.

  Args:
    n (int): The number of signals to generate.
    signal_length (int): The number of samples in each signal.

  Returns:
    tuple: A tuple containing:
      - v_signals (np.ndarray): The original correlated white noise signals.
      - e_signals (np.ndarray): The final uncorrelated (orthonormal) signals.
  """
  if n <= 0:
    raise ValueError("Number of signals (n) must be a positive integer.")
  if signal_length <= 0:
    raise ValueError("Signal length must be a positive integer.")

  # 1. Generate n white noise signals (v_n)
  # Each signal is a row in the matrix
  v_signals = np.random.randn(n, signal_length)
  u_signals = np.zeros_like(v_signals)

  # 2. Apply the Gram-Schmidt process to create uncorrelated signals (u_n)
  for i in range(n):
    v_i = v_signals[i]
    u_i = v_i
    # Subtract projections onto previous u signals
    for j in range(i):
      u_j = u_signals[j]
      u_i -= project(v_i, u_j)

    # Normalize and save
    u_i -= np.mean(u_i)
    u_i /= np.std(u_i)

    u_signals[i] = u_i

  return u_signals

def verify(u_signals):
  """
  Verifies that the signals are uncorrelated and have a norm of 1
  """
  n_signals = u_signals.shape[0]

  # Check correlation (inner product of different signals should be 0)
  print("\nVerifying Correlation (dot products of e_i, e_j for i!=j should be close to 0.0):")
  # Only show the upper triangle of the correlation matrix for brevity
  for i in range(n_signals):
    for j in range(i + 1, n_signals):
      corr = np.dot(u_signals[i], u_signals[j])
      print(f"  <e_{i+1}, e_{j+1}>: {corr: .6f}")





# ---------------------
# Version 2 of Features
# ---------------------

def make_features_from_scenario(
    scenario_emissions,
    historical_emissions=None,
    agents=['CO2'],
    ema_windows_years=(5.0, 30.0),
    dt_years=1.0
):
  """
  Constructs causal features for an LSTM/MLP using NumPy, replicating the 
  logic of 'make_features_emissions_generic' (JAX version).

  For each agent, it creates 4 features:
  1. E_prev: Previous year's emissions.
  2. EMA_short_prev: Short-term EMA (approx 5y) up to t-1.
  3. EMA_long_prev: Long-term EMA (approx 30y) up to t-1.
  4. CumE_prev: Cumulative emissions up to t-1.

  Args:
      scenario_emissions (dict): Dict of {agent: [values...]} for the future/test period.
      historical_emissions (dict): Dict of {agent: [values...]} for the history.
      agents (list): List of agent names to process.
      ema_windows_years (tuple): (short_window, long_window) for EMA calculation.
      dt_years (float): Time step size (default 1.0).

  Returns:
      np.ndarray: Feature matrix of shape (N_scenario_steps, 4 * len(agents)).
  """
  if not agents or not scenario_emissions:
    return np.zeros((0, 0))

  # 1. Determine N (length of the scenario/simulation period)
  try:
    N = len(scenario_emissions[agents[0]])
  except KeyError:
    raise ValueError(f"Agent '{agents[0]}' not found in scenario emissions.")

  # 2. Pre-compute Alpha values for EMA
  # Formula: alpha = 1 - exp(-dt / window)
  w_short, w_long = ema_windows_years
  alpha_short = 1.0 - np.exp(-dt_years / w_short)
  alpha_long = 1.0 - np.exp(-dt_years / w_long)

  all_agent_features = []

  for agent in agents:
    # --- Fetch Data ---
    e_scenario = np.asarray(scenario_emissions.get(agent, []), float).ravel()
    if len(e_scenario) != N:
      raise ValueError(f"Time series for agent '{agent}' has mismatched length.")

    e_hist = np.array([])
    if historical_emissions and agent in historical_emissions:
      e_hist = np.asarray(historical_emissions[agent], float).ravel()

    # --- Concatenate History + Scenario ---
    # We compute features on the full timeline to ensure continuity of EMAs
    e_combined = np.concatenate([e_hist, e_scenario])

    # --- Helper: Causal Shift ---
    # We want features at time 't' to depend only on data up to 't-1'.
    # Strategy: Compute statistics on the full array, then shift right by 1.
    # 1. Compute metrics (Cumulative, EMA) on e_combined
    # 2. Prepend a 0.0 (the state before any data exists)
    # 3. Slice out the portion corresponding to the scenario start.

    # A. Cumulative Emissions
    cumu_combined = np.cumsum(e_combined)

    # B. Exponential Moving Averages (EMAs)
    ema_short_combined = _numpy_ema(e_combined, alpha_short)
    ema_long_combined = _numpy_ema(e_combined, alpha_long)

    # --- Align Features (Shift by 1 for causality) ---
    # Create the "Previous" arrays by prepending 0 and dropping the last element.
    # If e_combined is [e0, e1, e2], prev is [0, e0, e1].

    # 1. Previous Emissions
    e_prev_full = np.concatenate(([0.0], e_combined[:-1]))

    # 2. Previous Cumulative
    cumu_prev_full = np.concatenate(([0.0], cumu_combined[:-1]))

    # 3. Previous EMAs
    ema_short_prev_full = np.concatenate(([0.0], ema_short_combined[:-1]))
    ema_long_prev_full = np.concatenate(([0.0], ema_long_combined[:-1]))

    # --- Slice to Scenario Period ---
    # The scenario starts after len(e_hist) samples.
    start_idx = len(e_hist)

    # Slice the relevant rows corresponding to the scenario timesteps
    agent_feats = np.column_stack([
        e_prev_full[start_idx : start_idx + N],          # Feature 1: Lag 1
        ema_short_prev_full[start_idx : start_idx + N],  # Feature 2: EMA Short
        ema_long_prev_full[start_idx : start_idx + N],   # Feature 3: EMA Long
        cumu_prev_full[start_idx : start_idx + N]        # Feature 4: Cumulative
    ])

    all_agent_features.append(agent_feats)

  # Stack all agents horizontally
  final_features = np.column_stack(all_agent_features)
  return final_features

def _numpy_ema(x, alpha):
  """
  Calculates the Exponential Moving Average (EMA) using a pure Python loop.
  Matches JAX logic: y_t = (1-alpha)*y_{t-1} + alpha*x_t.
  Initialization: y[0] = alpha * x[0].
  """
  y = np.zeros_like(x)
  if len(x) == 0:
    return y

  curr = alpha * x[0]
  y[0] = curr

  # Simple loop (performant enough for typical climate time series lengths)
  for i in range(1, len(x)):
    curr = (1.0 - alpha) * curr + alpha * x[i]
    y[i] = curr

  return y

def evaluate_trainsets_vs_tests_with_history(
  train_scenarios,
  train_series,
  test_scenarios,
  test_series,
  agents=['CO2'],
  training_groups=None,
  random_state=0,
  historical_scenario_name='historical'
):
  """
  Evaluates model performance by training on specified groups and testing on others,
  correctly handling a separate historical emissions period.
  """
  if training_groups is None:
    training_groups = {name: [name] for name in train_scenarios}

  # 1. Consolidate all data and extract the historical series
  train_data_map = {name: data for name, data in zip(train_scenarios, train_series)}
  test_data_map = {name: data for name, data in zip(test_scenarios, test_series)}

  historical_series_train, historical_series_test = None, None
  if historical_scenario_name in train_data_map:
    historical_series_train = train_data_map.pop(historical_scenario_name)
  historical_emissions_train = historical_series_train[0] if historical_series_train else None

  if historical_scenario_name in test_data_map:
    historical_series_test = test_data_map.pop(historical_scenario_name)
  historical_emissions_test = historical_series_test[0] if historical_series_test else None

  # 2. Pre-compute features for all scenarios (train and test)
  features_map_train, features_map_test = {}, {}

  # Populate train features
  for name, series_data in train_data_map.items():
    emissions_dict, y, needs_history = series_data
    hist_to_use = historical_emissions_train if needs_history else None

    X = make_features_from_scenario(
      emissions_dict,
      historical_emissions=hist_to_use,
      agents=agents,
    )
    y_adjusted = np.asarray(y, float)[:X.shape[0]]
    features_map_train[name] = (X, y_adjusted)

  # Add historical emissions to train features
  if historical_emissions_train:
    emissions_dict, y, needs_history = historical_series_train
    X_hist = make_features_from_scenario(
      emissions_dict,
      historical_emissions=None,  # History has no preceding data
      agents=agents,
    )
    y_hist = np.asarray(y, float)[:X_hist.shape[0]]
    features_map_train[historical_scenario_name] = (X_hist, y_hist)

  # Populate test features
  for name, series_data in test_data_map.items():
    emissions_dict, y, needs_history = series_data
    hist_to_use = historical_emissions_test if needs_history else None

    X = make_features_from_scenario(
      emissions_dict,
      historical_emissions=hist_to_use,
      agents=agents,
    )
    y_adjusted = np.asarray(y, float)[:X.shape[0]]
    features_map_test[name] = (X, y_adjusted)

  # Add historical emissions to test features
  if historical_emissions_test:
    emissions_dict, y, needs_history = historical_series_test
    X_hist = make_features_from_scenario(
      emissions_dict,
      historical_emissions=None,  # History has no preceding data
      agents=agents,
    )
    y_hist = np.asarray(y, float)[:X_hist.shape[0]]
    features_map_test[historical_scenario_name] = (X_hist, y_hist)

  # 3. Run the main training and evaluation loop
  error_dict, yhat_dict = {}, {}
  for group_name, scenario_names_in_group in training_groups.items():
    all_Xtr, all_ytr = [], []
    for scenario_name in scenario_names_in_group:
      if scenario_name in features_map_train:
        X, y = features_map_train[scenario_name]
        all_Xtr.append(X)
        all_ytr.append(y)
      else:
        raise ValueError(f'Scenario {scenario_name} for training not found.')

    if not all_Xtr:
      print(f"Warning: No training data found for group '{group_name}'. Skipping.")
      continue

    Xtr_combined = np.vstack(all_Xtr)
    ytr_combined = np.concatenate(all_ytr)

    model = MLPRegressor(hidden_layer_sizes=(16,16,16,16), random_state=random_state, max_iter=500)
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr_combined)
    model.fit(Xtr_scaled, ytr_combined)

    error_dict[group_name], yhat_dict[group_name] = {}, {}
    for test_name in test_scenarios:
      if test_name in features_map_test:
        Xte, yte = features_map_test[test_name]
        Xte_scaled = scaler.transform(Xte)
        yhat = model.predict(Xte_scaled)
        error_dict[group_name][test_name] = rmse(yte, yhat)
        yhat_dict[group_name][test_name] = yhat

  return error_dict, yhat_dict

def evaluate_LSTM(
  train_scenarios,
  train_series,
  test_scenarios,
  test_series,
  agents=['CO2'],
  training_groups=None,
  random_state=0,
  historical_scenario_name='historical',
  lstm_size=16,
  dense_size=32,
  dropout_rate=0.1,
  l2_penalty=0.001
):
  """
  Evaluates an LSTM model's performance by training on specified groups and
  testing on others, handling a separate historical emissions period.
  This function is a direct replacement for the MLP version.
  """
  tf.random.set_seed(random_state)
  np.random.seed(random_state)

  if training_groups is None:
    training_groups = {name: [name] for name in train_scenarios}

  # 1. Consolidate all data and extract the historical series
  train_data_map = {name: data for name, data in zip(train_scenarios, train_series)}
  test_data_map = {name: data for name, data in zip(test_scenarios, test_series)}

  historical_series_train, historical_series_test = None, None
  if historical_scenario_name in train_data_map:
    historical_series_train = train_data_map.pop(historical_scenario_name)
  historical_emissions_train = historical_series_train[0] if historical_series_train else None

  if historical_scenario_name in test_data_map:
    historical_series_test = test_data_map.pop(historical_scenario_name)
  historical_emissions_test = historical_series_test[0] if historical_series_test else None

  # 2. Pre-compute features for all scenarios (train and test)
  features_map_train, features_map_test = {}, {}

  # Populate train features
  for name, series_data in train_data_map.items():
    emissions_dict, y, needs_history = series_data
    hist_to_use = historical_emissions_train if needs_history else None
    X = make_features_from_scenario(emissions_dict, historical_emissions=hist_to_use, agents=agents)
    y_adjusted = np.asarray(y, float)[:X.shape[0]]
    features_map_train[name] = (X, y_adjusted)

  # Add historical emissions to train features
  if historical_emissions_train:
    emissions_dict, y, needs_history = historical_series_train
    X_hist = make_features_from_scenario(emissions_dict, historical_emissions=None, agents=agents)
    y_hist = np.asarray(y, float)[:X_hist.shape[0]]
    features_map_train[historical_scenario_name] = (X_hist, y_hist)

  # Populate test features
  for name, series_data in test_data_map.items():
    emissions_dict, y, needs_history = series_data
    hist_to_use = historical_emissions_test if needs_history else None
    X = make_features_from_scenario(emissions_dict, historical_emissions=hist_to_use, agents=agents)
    y_adjusted = np.asarray(y, float)[:X.shape[0]]
    features_map_test[name] = (X, y_adjusted)

  # Add historical emissions to test features
  if historical_emissions_test:
    emissions_dict, y, needs_history = historical_series_test
    X_hist = make_features_from_scenario(emissions_dict, historical_emissions=None, agents=agents)
    y_hist = np.asarray(y, float)[:X_hist.shape[0]]
    features_map_test[historical_scenario_name] = (X_hist, y_hist)

  # 3. Run the main training and evaluation loop
  error_dict, yhat_dict = {}, {}
  for group_name, scenario_names_in_group in training_groups.items():
    all_Xtr, all_ytr = [], []
    for scenario_name in scenario_names_in_group:
      if scenario_name in features_map_train:
        X, y = features_map_train[scenario_name]
        all_Xtr.append(X)
        all_ytr.append(y)
      else:
        raise ValueError(f'Scenario {scenario_name} for training not found.')

    if not all_Xtr:
      print(f"Warning: No training data found for group '{group_name}'. Skipping.")
      continue

    Xtr_combined = np.vstack(all_Xtr)
    ytr_combined = np.concatenate(all_ytr)

    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr_combined)

    # Reshape data for LSTM: [samples, timesteps, features]
    # Here, we treat each year's feature set as a sequence of length 1.
    #batch_size = 32
    batch_size = Xtr_scaled.shape[0]
    n_features = Xtr_scaled.shape[1]
    Xtr_reshaped = Xtr_scaled.reshape((Xtr_scaled.shape[0], 1, n_features))

    # Define the LSTM model (includes dropout and weight decay)
    model = Sequential([
      Input(shape=(1, n_features)),
      LSTM(lstm_size,
           kernel_regularizer=l2(l2_penalty)),
      Dense(1,
            kernel_regularizer=l2(l2_penalty))
    ])
    """
    model = Sequential([
      Input(shape=(1, n_features)),
      LSTM(lstm_size,
           kernel_regularizer=l2(l2_penalty)),
      Dropout(dropout_rate),
      Dense(dense_size,
            activation='relu',
            kernel_regularizer=l2(l2_penalty)),
      Dropout(dropout_rate),
      Dense(1,
            kernel_regularizer=l2(l2_penalty))
    ])"""

    X_tensor = tf.convert_to_tensor(Xtr_reshaped, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(ytr_combined, dtype=tf.float32)
    opt = SGD(learning_rate=5e-2)

    @tf.function(jit_compile=True)
    def train_loop(X, y, steps):
      for i in tf.range(steps):
        with tf.GradientTape() as tape:
          y_pred = model(X, training=True)
          loss = tf.reduce_mean(tf.square(y - y_pred))
          if model.losses:
            loss += tf.add_n(model.losses)

        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
      return loss

    train_loop(X_tensor, y_tensor, steps=400)

    #model.compile(optimizer=AdamW(weight_decay=l2_penalty), loss='mean_squared_error')
    #model.compile(optimizer=SGD(learning_rate=5e-2), loss='mean_squared_error')
    """
    total_gradient_steps = 400
    training_epochs = 100
    steps_per_epoch = total_gradient_steps // training_epochs

    train_dataset = tf.data.Dataset.from_tensor_slices((Xtr_reshaped, ytr_combined))
    train_dataset = train_dataset.shuffle(buffer_size=Xtr_reshaped.shape[0])
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.repeat()

    training_epochs = 400

    model.fit(train_dataset,
              epochs=training_epochs,
              batch_size=batch_size,
              #steps_per_epoch=steps_per_epoch,
              shuffle=False,
              verbose=0)

    """

    error_dict[group_name], yhat_dict[group_name] = {}, {}
    for test_name in test_scenarios:
      if test_name in features_map_test:
        Xte, yte = features_map_test[test_name]
        if Xte.shape[0] == 0: continue # Skip empty test sets

        Xte_scaled = scaler.transform(Xte)
        # Reshape test data for LSTM prediction
        Xte_reshaped = Xte_scaled.reshape((Xte_scaled.shape[0], 1, n_features))

        yhat = model.predict(Xte_reshaped, verbose=0).flatten() # flatten to make it 1D

        error_dict[group_name][test_name] = rmse(yte, yhat)
        yhat_dict[group_name][test_name] = yhat

  return error_dict, yhat_dict

