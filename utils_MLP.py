# Imports
import random
from tokenize import group
import numpy as np
import matplotlib.pyplot as plt

import run_fair

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import seaborn as sns
from cmcrameri import cm

## Setup plots
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "sans-serif",
  "font.sans-serif": ["Helvetica Light"],
})

def _ema(e, tau):
  # exponential moving average
  a = 1.0 - np.exp(-1.0 / float(tau))
  s = 0.0
  out = np.empty_like(e)
  for i, v in enumerate(e):
    s += a * (v - s)
    out[i] = s
  return out

def make_features(emissions_dict, agents, n_lags=10):
  all_agent_features = []
  # Determine the length of the time series from the first agent

  N = len(emissions_dict[agents[0]]) if agents else 0
  if N == 0:
    # Handle case with no agents if necessary
    return np.zeros((0, 0))

  for agent in agents:
    e = np.asarray(emissions_dict[agent], float).ravel()
    if len(e) != N:
      raise ValueError(f"Time series for agent '{agent}' has mismatched length.")

    # Create lag features for the current agent
    X_lags = np.zeros((N, n_lags + 1))
    for k in range(n_lags + 1):
      X_lags[k:, k] = e[:N - k]

    # Create cumulative emissions feature for the current agent
    cumu = np.cumsum(e)

    agent_feature_block = np.column_stack([X_lags, cumu])
    all_agent_features.append(agent_feature_block)

  # Horizontally stack the feature blocks from all agents
  base = np.column_stack(all_agent_features)

  return base

def _new_model(random_state=0):
  return make_pipeline(
      StandardScaler(),
      MLPRegressor(hidden_layer_sizes=(8, 8), activation="relu",
                    max_iter=5000, early_stopping=True, random_state=random_state, shuffle=False)
  )

def evaluate_trainsets_vs_tests(
  train_scenarios,
  train_series,
  test_scenarios,
  test_series,
  agents=['CO2'],
  training_groups=None,
  n_lags=10,
  random_state=0
):

  def features(s):  # build X,y for a series
    emissions_dict, y = s
    X = make_features(emissions_dict, agents, n_lags=n_lags)
    y = np.asarray(y, float)[:X.shape[0]]
    return X, y

  if training_groups is None:
    training_groups = {name: [name] for name in train_scenarios}

  # Map from scenario names to data for fast lookup
  scenario_data_map = {name: data for name, data in zip(train_scenarios, train_series)}
  error_dict, yhat_dict = {}, {}

  for group_name, scenario_names_in_group in training_groups.items():

    # 1. Aggregate data for all scenarios in the current group
    all_Xtr, all_ytr = [], []
    for scenario_name in scenario_names_in_group:
      if scenario_name in scenario_data_map:
        series_data = scenario_data_map[scenario_name]
        X, y = features(series_data)
        all_Xtr.append(X)
        all_ytr.append(y)
      else:
        raise ValueError(f'Warning: Scenario {scenario_name} not found.')

    # 2. Concatenate into a single training set for this group
    Xtr_combined = np.vstack(all_Xtr)
    ytr_combined = np.concatenate(all_ytr)

    # 3. Create and fit one model on the combined group data
    model = _new_model(random_state=random_state)
    model.fit(Xtr_combined, ytr_combined)

    # 4. Use the group name as the key for the results
    error_dict[group_name], yhat_dict[group_name] = {}, {}
    for j, test in enumerate(test_scenarios):
      Xte, yte = features(test_series[j])
      yhat = model.predict(Xte)
      error_dict[group_name][test] = nrmse(yte, yhat)  # normalized by max of y_test
      yhat_dict[group_name][test] = yhat

  return error_dict, yhat_dict

def plot_yhats_grid(error_dict, yhat_dict, ytrue_dict):
  train_scenarios = sorted(error_dict.keys())
  test_scenarios = sorted({test for train in train_scenarios for test in error_dict[train].keys()})
  n_test = len(test_scenarios)
  cols = int(np.ceil(np.sqrt(n_test))); rows = int(np.ceil(n_test / cols))
  fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), constrained_layout=True, sharex=True, sharey=True)

  for i, test in enumerate(test_scenarios):
    ax = axes[i // cols][i % cols]
    for j, train in enumerate(train_scenarios):
      if train == 'noise':
        ls = '--'
      else:
        ls = '-'
      ax.plot(yhat_dict[train][test], lw=1, alpha=0.8, ls=ls, label=train, c=cm.batlowKS(j))  # all yhats for this test
    ax.plot(ytrue_dict[test], lw=2, alpha=0.8, label='True', c='k')
    ax.legend(loc='upper left', fontsize=8)

    ax.set_title(f"Test {test}")
    #ax.set_ylim([-0.5,9])

    if i % 4 == 0:
      ax.set_ylabel("yhat")
    if i >= cols * (rows - 1):
      ax.set_xlabel('t')

  for k in range(n_test, rows*cols):
    axes[k // cols][k % cols].axis('off')  # hide empties

  return

def predict_series(model, emissions, extras=None, n_lags=10):
  return model.predict(make_features(emissions, extras, n_lags))

def rmse(y_true, y_pred):
  y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
  return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def nrmse(y_true, y_pred):
  denom = max(np.max(np.abs(y_true)), 1e-12)  # normalize by largest magnitude in y_test
  return rmse(y_true/denom, y_pred/denom)

def plot_error_heatmap(error_dict, train_scenarios, test_scenarios):
  # Build matrix with rows=test, cols=train
  E = np.full((len(test_scenarios), len(train_scenarios)), np.nan)
  for r, te in enumerate(test_scenarios):
    for c, tr in enumerate(train_scenarios):
      E[r, c] = error_dict.get(tr, {}).get(te, np.nan)

  avg_error_row = np.median(E, axis=0)
  E_all = np.vstack([E, avg_error_row])
  yticklabels = test_scenarios.copy() + ['Avg.']

  # Plot
  fig, ax = plt.subplots(figsize=(1.2*len(train_scenarios)+2, 1.0*len(test_scenarios)+2), constrained_layout=True)
  sns.heatmap(E_all, ax=ax, vmin=0, vmax=1, cmap=cm.lajolla_r,
              xticklabels=train_scenarios, yticklabels=yticklabels,
              annot=True, fmt=".2g", cbar_kws={"label": "NRMSE / global max"})
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