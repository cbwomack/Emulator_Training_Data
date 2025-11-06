"""
utils_optimization.py

Helper file for performing hyperparameter optimization for a climate emulator.
Includes:
- Signal generation functions for synthetic data.
- Parameter space builders.
- A wrapper class for the black-box evaluation function.
- A pipeline for:
  1. Latin Hypercube Sampling (Exploration)
  2. Bayesian Optimization (Refinement)
"""

# Imports
## Standard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

## Math/Optimization
from scipy import signal
from scipy.stats import qmc
from bayes_opt import BayesianOptimization

## Utils
import utils_ML
import run_fair

## System
import warnings
from pathlib import Path

# --------------------------------------
# Section 1: Signal Generation Functions
# --------------------------------------

def zero_mean(t, offset):
  """A mean trend with an offset."""
  return np.zeros_like(t) + offset

def poly5(t, c0, c1, c2, c3, c4, c5):
  """
  5th-order polynomial in (normalized) time with 6 coefficients.
  Time is normalized to [0, 1] internally to keep coefficients well-scaled.
  y(t) = c0 + c1*u + c2*u^2 + c3*u^3 + c4*u^4 + c5*u^5,  where u = (t - t0)/(tN - t0)
  """
  if len(t) > 1:
    t0, tN = t[0], t[-1]
    denom = (tN - t0) if (tN - t0) != 0 else 1.0
    u = (t - t0) / denom
  else:
    u = np.zeros_like(t, dtype=float)

  # Horner's method
  return (((((c5*u + c4)*u + c3)*u + c2)*u + c1)*u + c0)

def sinusoidal_cycle(t, amplitude, frequency):
  """A long-period sinusoidal cycle."""
  return amplitude * np.sin(2 * np.pi * frequency * t)

def delayed_sinusoid(t, amplitude, frequency, t_start):
  """
  A sinusoid that 'begins' at t_start: zero before, sine after.
  Equivalent to sinusoidal_cycle shifted by t_start.
  """
  tau = t - t_start
  return np.where(t >= t_start, amplitude * np.sin(2 * np.pi * frequency * tau), 0.0)

def sine_offset(t, amplitude, frequency, offset):
  """A long-period sinusoidal cycle."""
  return amplitude * np.sin(2 * np.pi * frequency * t) + offset

def sine_sweep(t, amplitude, f0, f1, n_steps):
  """A sinusoidal sweep across multiple frequencies"""
  return amplitude * signal.chirp(t, f0, n_steps, f1)

def multi_sine(t, A_slow, f_slow,
               A_fast, f_fast):
  """
  A unified signal combining a slow sine and a fast sine.
  """
  slow_sine_component = A_slow * np.sin(2 * np.pi * f_slow * t)
  fast_sine_component = A_fast * np.sin(2 * np.pi * f_fast * t)

  return slow_sine_component + fast_sine_component

def multi_trend(t, A_slow, f_slow,
                       A_fast, f_fast,
                       A_step, k, t_center):
  """
  A unified signal combining a slow sine, a fast sine, and a smooth step (tanh).
  This creates a complex, continuous signal for optimization.
  """
  slow_sine_component = A_slow * np.sin(2 * np.pi * f_slow * t)
  fast_sine_component = A_fast * np.sin(2 * np.pi * f_fast * t)
  step_component = A_step * np.tanh(k * (t - t_center))

  return slow_sine_component + fast_sine_component + step_component

def ornstein_uhlenbeck(t, theta, sigma, mu):
  """
  Generates an Ornstein-Uhlenbeck process.
  This process models mean-reverting behavior, where the process is
  continuously pulled towards a long-term mean.

  Args:
    t (np.ndarray): Time array.
    theta (float): Drift term, rate of reversion to the mean.
    sigma (float): Standard deviation of the noise (volatility).
    mu (float): The long-term mean of the process.

  Returns:
    np.ndarray: The generated Ornstein-Uhlenbeck process.
  """
  n_steps = len(t)
  # Assume constant time steps
  dt = t[1] - t[0] if n_steps > 1 else 1
  x = np.zeros(n_steps)
  # Start the process at its long-term mean
  x[0] = mu

  # Iterate through time to generate the process
  for i in range(1, n_steps):
    drift = theta * (mu - x[i-1]) * dt
    noise = sigma * np.sqrt(dt) * np.random.randn()
    x[i] = x[i-1] + drift + noise
  return x

def random_telegraph_signal(t, lambda_rate, level_min, level_max):
  """
  Generates a Random Telegraph Signal.
  The signal holds at a random constant value for a random duration,
  then jumps to a new random value.

  Args:
    t (np.ndarray): Time array.
    lambda_rate (float): Rate parameter for the exponential distribution
                        governing jump times (average jumps per year).
    level_min (float): The minimum value the signal can jump to.
    level_max (float): The maximum value the signal can jump to.

  Returns:
    np.ndarray: The generated Random Telegraph Signal.
  """
  n_steps = len(t)
  out = np.zeros(n_steps)
  current_time = 0
  current_step_idx = 0

  while current_step_idx < n_steps:
    # Duration of the hold is drawn from an exponential distribution
    # The average duration is 1 / lambda_rate
    hold_duration = np.random.exponential(1.0 / lambda_rate)

    # The level of the signal is drawn from a uniform distribution
    hold_level = np.random.uniform(level_min, level_max)

    # Determine the end index for this hold period
    end_time = current_time + hold_duration
    end_step_idx = np.searchsorted(t, end_time, side='right')

    # Fill the signal array with the constant value for this duration
    out[current_step_idx:end_step_idx] = hold_level

    # Update our position
    current_step_idx = end_step_idx
    current_time = end_time

  return out

# --------------------------------------------
# Section 2: Configuration and Parameter Space
# --------------------------------------------

def get_mean_functions(n_steps, eps=1e-7):
  """
  Returns the dictionary of mean functions, dynamically setting
  time-dependent parameters like t_center.
  """
  return {
      "const": {"function": zero_mean,
                "params": {
                  "m_param_offset": {
                        "CO2": (0.0, 100.0),
                        "CH4": (0.0, 500.0),
                        "default": (0.0, 100.0)
                        }}},
      "poly5": {
          "function": poly5,
          "params": {
              # CO2-focused ranges; tune as needed for your magnitude/units
              "m_param_c0": {
                "CO2": (0.0, 250.0),
                "default": (0.0, 250.0) },
              "m_param_c1": {
                "CO2": (-100.0, 300.0),
                "default": (-100.0, 300.0) },
              "m_param_c2": {
                "CO2": (-100.0, 600.0),
                "default": (-100.0, 600.0) },
              "m_param_c3": {
                "CO2": (-100.0, 600.0),
                "default": (-100.0, 600.0) },
              "m_param_c4": {
                "CO2": (-100.0, 600.0),
                "default": (-100.0, 600.0) },
              "m_param_c5": {
                "CO2": (-100.0, 600.0),
                "default": (-100.0, 600.0) }
          }
      },
      "sinusoid": {
          "function": sinusoidal_cycle,
          "params": {"m_param_amplitude": {
                      "CO2": (5.0, 250.0),
                      "CH4": (100.0, 800.0),
                      "default": (5.0, 250.0)
                      },
                      "m_param_frequency": {
                        "default": (0.0005, 0.01) # 2,000 to 100 years
                        }}
      },
      "delayed_sinusoid": {
          "function": delayed_sinusoid,
          "params": {
              "m_param_amplitude": {
                  "CO2": (5.0, 250.0),
                  "CH4": (100.0, 800.0),
                  "Sulfur": (10, 100),
                  "default": (5.0, 250.0)
              },
              "m_param_frequency": {
                  "default": (0.0005, 0.01)  # 2,000 to 100 years
              },
              "m_param_t_start": {
                  # choose different start windows per agent as needed
                  "CO2": (0.0, 0.5 * n_steps),
                  "CH4": (0.0, 0.5 * n_steps),
                  "Sulfur": (0.0, 0.5 * n_steps),
                  "default": (0.0, 0.5 * n_steps)
              }}
      },
      "sine_offset": {
          "function": sine_offset,
          "params": {"m_param_amplitude": {
                      "CO2": (5.0, 250.0),
                      "CH4": (100.0, 800.0),
                      "default": (5.0, 250.0)
                      },
                      "m_param_frequency": {
                        "default": (0.0005, 0.01) # 2,000 to 100 years
                        },
                      "m_param_offset": {
                        "CO2": (0.0, 100.0),
                        "CH4": (0.0, 500.0),
                        "default": (0.0, 100.0)
                      }}
      },
      "sweep":{
          # Pass n_steps to the partial function or wrapper
          "function": lambda t, **kwargs: sine_sweep(t, n_steps=n_steps, **kwargs),
          "params": {"m_param_amplitude": {"default": (10.0, 75.0)},
                      "m_param_f0": {"default": (0.0005, 0.001 - eps)}, # 2,000 to 1,000 years
                      "m_param_f1": {"default": (0.001 + eps, 0.01)}}   # 1,000 to 100 years
      },
      "multi_sine":{
          "function": multi_sine,
          "params": {
              "m_param_A_slow": {
                  "CO2": (0.0, 250.0),
                  "CH4": (0.0, 1000.0),
                  "default": (0.0, 250.0)
                  },
              "m_param_f_slow": {"default": (0.0005, 0.001)},
              "m_param_A_fast": {
                  "CO2": (0.0, 100.0),
                  "CH4": (0.0, 500.0),
                  "default": (0.0, 250.0)
                  },
              "m_param_f_fast": {"default": (0.001, 0.01)}
          }
      },
      "multi_trend":{
          "function": multi_trend,
          "params": {
              "m_param_A_slow": {
                "CO2": (0.0, 250),
                "CH4": (0.0, 800),
                "default": (0.0, 200.0)},
              "m_param_f_slow": {"default": (0.0005, 0.001)},
              "m_param_A_fast": {
                  "CO2": (0.0, 100.0),
                  "CH4": (0.0, 400.0),
                  "default": (0.0, 250.0)
                  },
              "m_param_f_fast": {"default": (0.001, 0.01)},
              "m_param_A_step": {"default": (0.0, 500.0)},
              "m_param_k": {"default": (0.02, 0.4)},
              "m_param_t_center": {"default": (0.05 * n_steps, 0.9 * n_steps)}
          },
      },
      "ornstein_uhlenbeck": {
          "function": ornstein_uhlenbeck,
          "params": {
              "m_param_theta": {"default": (0.001, 1.0)},
              "m_param_sigma": {
                  "CO2": (1.0, 50.0),
                  "CH4": (1.0, 200.0),
                  "default": (1.0, 100.0)
                  },
              "m_param_mu": {
                  "CO2": (0.0, 100.0),
                  "CH4": (0.0, 200.0),
                  "default": (0.0, 100.0)
                  }
          }
      },
      "telegraph": {
          "function": random_telegraph_signal,
          "params": {
              "m_param_lambda_rate": {"default": (0.01, 0.2)},
              "m_param_level_min": {
                  "CO2": (0.0, 25.0),
                  "CH4": (0.0, 100.0),
                  "default": (0.0, 50.0)
                  },
              "m_param_level_max": {
                  "CO2": (25.0, 100.0),
                  "CH4": (100.0, 400.0),
                  "default": (50.0, 200.0)
                  }
          }
      }
  }

def build_pbounds(CONFIG, AGENTS, n_steps):
  """
  Builds the parameter bounds dictionary for the optimizer
  based on the experiment configuration.
  """
  MEAN_FUNCTIONS = get_mean_functions(n_steps)
  chosen_config = MEAN_FUNCTIONS[CONFIG["CHOSEN_MEAN_FUNCTION"]]
  mean_function_to_run = chosen_config["function"]

  pbounds = {}
  for agent in AGENTS:
    # Add noise parameters for this agent
    if CONFIG["USE_WHITE_NOISE"] and CONFIG['CHOSEN_MEAN_FUNCTION'] != 'ornstein_uhlenbeck':
      for param, bounds in CONFIG["NOISE_PARAM_SPACE"].items():
        pbounds[f'{param}_{agent}'] = bounds
    # Add mean function parameters for this agent
    for param, bounds_dict in chosen_config["params"].items():
      bounds = bounds_dict.get(agent, bounds_dict["default"])
      pbounds[f'{param}_{agent}'] = bounds

  return pbounds, mean_function_to_run

# ---------------------
# Section 3: Evaluation
# ---------------------

def evaluate_lstm_ensemble(n_ensemble=1, **kwargs):
  """
  Wraps the LSTM evaluation to average over an ensemble.
  """
  if n_ensemble <= 1:
    return utils_ML.evaluate_LSTM(**kwargs)

  all_error_dicts = []
  for i in range(n_ensemble):
    kwargs['random_state'] = i
    error_dict, _ = utils_ML.evaluate_LSTM(**kwargs)
    all_error_dicts.append(error_dict)

  avg_error_dict = {}
  if not all_error_dicts:
    return {}, None

  template = all_error_dicts[0]
  for group_name in template:
    avg_error_dict[group_name] = {}
    for test_name in template[group_name]:
      errors_for_scenario = [d[group_name][test_name] for d in all_error_dicts]
      avg_error_dict[group_name][test_name] = np.mean(errors_for_scenario)

  return avg_error_dict, None

def evaluate_parameter_set(params, CONFIG, mean_func, agents,
                           n_steps, FS, start, stop,
                           scenarios_valid, series_valid, loss_metric):
  """
  Evaluates a single parameter set for all agents.
  """
  rmse_scores = []
  for _ in range(CONFIG["N_INNER_SAMPLES"]):
    uncorrelated_noise = utils_ML.generate_uncorrelated_signals(n=len(agents), signal_length=n_steps)
    U = np.zeros((len(agents), n_steps), dtype=float)

    for i, agent in enumerate(agents):
      mean_func_params = {
        k.replace('m_param_', ''): v
        for k, v in params.items()
        if k.startswith(f'm_param_') and k.endswith(f'_{agent}')
      }
      mean_func_params_clean = {k.replace(f'_{agent}', ''): v for k, v in mean_func_params.items()}
      m_t = mean_func(np.arange(n_steps), **mean_func_params_clean)

      if not CONFIG['USE_WHITE_NOISE'] or CONFIG['CHOSEN_MEAN_FUNCTION'] == 'ornstein_uhlenbeck':
        u_t = m_t
      else:
        sigma = params[f'sigma_{agent}']
        f_low = params[f'f_low_{agent}']
        f_high = params[f'f_high_{agent}']
        base_noise = uncorrelated_noise[i]
        nyquist = 0.5 * FS
        b, a = signal.butter(4, [f_low/nyquist, f_high/nyquist], btype='band')
        w_t = signal.filtfilt(b, a, base_noise)
        w_t = (w_t - np.mean(w_t)) / np.std(w_t)
        u_t = m_t + sigma * w_t
      U[i] = u_t

    #U_decorr = zca_decorrelate_rows(U, eps=1e-8, keep_rms=True)
    emissions_dict = {agent: U[i] for i, agent in enumerate(agents)}

    f_train = run_fair.build_fair(start, stop)
    run_fair.set_emissions(f_train, np.arange(start, stop), run_fair.ensure_all_agents(emissions_dict, n_steps))
    f_train.run(progress=False)

    if np.any(f_train.concentration.sel(specie='CO2') < 100) or \
      np.any(f_train.concentration.sel(specie='CH4') < 100):
      penalty = 1e9
      current_mean = np.mean(rmse_scores) if rmse_scores else 0
      return current_mean + penalty

    _, _, training_temp, _ = run_fair.run_fair(f_train, np.arange(start, stop))
    train_scenarios = ["bo_training_run"]
    train_series = [(emissions_dict, training_temp, False)]

    error_dict, _ = evaluate_lstm_ensemble(
      n_ensemble=CONFIG["N_ENSEMBLE_MODELS"],
      train_scenarios=train_scenarios,
      train_series=train_series,
      test_scenarios=scenarios_valid,
      test_series=series_valid,
      agents=agents,
      n_lags=0, lstm_size=16, dense_size=32,
      dropout_rate=0.2, l2_penalty=0.001
    )
    scores_for_this_run = [err for err in error_dict["bo_training_run"].values()]

    if loss_metric == 'max':
      loss_score = np.max(scores_for_this_run)
    elif loss_metric == 'mean':
      loss_score = np.mean(scores_for_this_run)
    elif loss_metric == 'median':
      loss_score = np.median(scores_for_this_run)
    else:
      raise ValueError(f'Error, loss {loss_metric} not recognized.')

    rmse_scores.append(loss_score)

  return np.mean(rmse_scores)

class OptimizationWrapper:
  """
  A class to wrap the complex evaluation function, holding all static
  dependencies so it can be called by the optimizer with only `**params`.
  """
  def __init__(self, CONFIG, mean_func, agents, n_steps, FS,
               start, stop, scenarios_valid, series_valid, loss_metric):
    self.CONFIG = CONFIG
    self.mean_func = mean_func
    self.agents = agents
    self.n_steps = n_steps
    self.FS = FS
    self.start = start
    self.stop = stop
    self.scenarios_valid = scenarios_valid
    self.series_valid = series_valid
    self.loss_metric = loss_metric

    # Suppress warnings during optimization runs
    warnings.simplefilter('ignore')

  def __call__(self, **params):
    """
    This method is called by the Bayesian Optimizer.
    It must return a single score to be maximized.
    """
    # Enforce constraints
    if self.CONFIG["USE_WHITE_NOISE"] and self.CONFIG['CHOSEN_MEAN_FUNCTION'] != 'ornstein_uhlenbeck':
      for agent in self.agents:
        if params[f'f_low_{agent}'] > params[f'f_high_{agent}']:
          # Return a very bad score (maximization)
          return -np.inf

    # Run the full evaluation
    rmse = evaluate_parameter_set(
      params, self.CONFIG, self.mean_func, self.agents,
      self.n_steps, self.FS, self.start, self.stop,
      self.scenarios_valid, self.series_valid, self.loss_metric
    )

    # We return -rmse because the optimizer maximizes
    return -rmse

# --------------------------------
# Section 4: Optimization Pipeline
# --------------------------------

def generate_experiment_filename(mean_func_name, use_white_noise, n_inner, n_ensemble, agents, validation_name, loss):
  """
  Generates a unique, descriptive base filename for log and result files.

  Example:
  mean-sinusoid_noise-on_inner-10_ensemble-1_agents-CO2+CH4_valid-scenarioMIP
  """

  # 1. Mean Function
  mean_str = f"mean-{mean_func_name}"

  # 2. White Noise
  noise_str = "noise-on" if use_white_noise else "noise-off"

  # 3. Inner Samples
  inner_str = f"inner-{n_inner}"

  # 4. Ensemble Members
  ensemble_str = f"ensemble-{n_ensemble}"

  # 5. Agents
  # Sort agents to ensure consistent naming
  # (e.g., ['CO2', 'CH4'] is same as ['CH4', 'CO2'])
  sorted_agents = sorted(agents)
  # Use '+' as a sub-delimiter for lists
  agents_str = f"agents-{'+'.join(sorted_agents)}"

  # 6. Validation Dataset
  valid_str = f"valid-{validation_name}"

  # 7. Loss function
  loss_str = f"loss-{loss}"

  # Assemble the base name
  base_name = f"{mean_str}_{noise_str}_{inner_str}_{ensemble_str}_{agents_str}_{valid_str}_{loss_str}"

  return base_name

def run_lhs_phase(black_box_func, pbounds, n_samples, csv_path):
  """
  Runs the initial Latin Hypercube Sampling phase.
  If csv_path exists, it loads results. Otherwise, it runs the
  samples, saves the results, and returns them.
  """
  csv_file = Path(csv_path)
  param_names = list(pbounds.keys())

  try:
    # --- 1. Check for existing results ---
    print(f"Checking for existing LHS results at {csv_file}...")
    results_df = pd.read_csv(csv_file)
    n_existing = len(results_df)
    print(f"Found and loaded {len(results_df)} existing LHS results.")

  except FileNotFoundError:
    # --- 2. No file found, run new LHS ---
    print(f"No results file found. Running {n_samples} new LHS evaluations...")
    # Create an empty DataFrame with the correct columns
    results_df = pd.DataFrame(columns=param_names + ['target'])
    n_existing = 0

  n_to_run = n_samples - n_existing
  if n_to_run <= 0:
    print(f"Already have {n_existing} samples, meeting or exceeding target of {n_samples}. No new runs needed.")
    # Make sure to return only the requested number of samples
    results_df = results_df.head(n_samples)

  else:
    print(f"Target is {n_samples}, found {n_existing}. Running {n_to_run} new LHS evaluations...")
    sampler = qmc.LatinHypercube(d=len(param_names), seed=42)
    unit_samples_all = sampler.random(n=n_samples)

    # Get lower and upper bounds from pbounds
    l_bounds = [pbounds[k][0] for k in param_names]
    u_bounds = [pbounds[k][1] for k in param_names]

    # Scale samples from [0, 1] to the parameter space
    scaled_samples_all = qmc.scale(unit_samples_all, l_bounds, u_bounds)

    # Slice to get only the new samples we need to run
    new_samples_to_run = scaled_samples_all[n_existing:n_samples]

    new_results_data = []
    for i, sample in enumerate(new_samples_to_run):
      # Convert array back to parameter dictionary
      params_dict = {name: val for name, val in zip(param_names, sample)}

      print(f"Running LHS sample {n_existing + i + 1}/{n_samples}...")
      target_score = black_box_func(**params_dict)

      # Store results
      row = params_dict.copy()
      row['target'] = target_score
      new_results_data.append(row)

    if new_results_data:
      new_results_df = pd.DataFrame(new_results_data)
      results_df = pd.concat([results_df, new_results_df], ignore_index=True)

      print(f"LHS phase complete. Saving all {len(results_df)} results to {csv_file}...")
      # Save the combined DataFrame
      results_df.to_csv(csv_file, index=False)

  # --- 4. Prepare results for BO warm-start ---
  param_list = [row.to_dict() for _, row in results_df[param_names].iterrows()]
  target_list = results_df['target'].tolist()

  return param_list, target_list

def setup_bo_optimizer(black_box_func, pbounds, lhs_results, bo_state_file, random_state=42):
  """
  Initializes the BayesianOptimization object.
  - "Warms up" the optimizer with LHS results.
  - Loads previous BO runs from the JSON log if it exists.
  - Subscribes the JSON logger for future runs.
  """
  param_list, target_list = lhs_results

  # --- 1. Initialize Optimizer ---
  optimizer = BayesianOptimization(
    f=black_box_func,
    pbounds=pbounds,
    random_state=random_state,
    verbose=1
  )

  # --- 2. Setup JSON Logger for saving and resuming ---
  state_file = Path(bo_state_file)
  if state_file.exists():
    print(f"Loading existing optimizer state from: {state_file}")
    optimizer.load_state(str(state_file))
    print(f"Optimizer state loaded. Found {len(optimizer.res)} previous points.")
  else:
    # --- 3. No state found, register LHS points ---
    print(f"No state file found. Registering {len(param_list)} LHS points...")
    for params, target in zip(param_list, target_list):
      try:
        # register() will check if the point already exists
        optimizer.register(params=params, target=target)
      except KeyError:
        # This can happen if a parameter is slightly different due to
        # float precision from CSV. We can ignore it.
        pass

    n_lhs = len(param_list)
    n_bo_loaded = len(optimizer.res) - n_lhs
    print(f"Optimizer setup complete. Loaded {n_lhs} LHS points and {n_bo_loaded} previous BO points.")

  return optimizer

def run_bo_iterations(optimizer, n_iter, bo_state_file):
  """
  Runs the Bayesian Optimization loop for a set number of new iterations
  and saves the updated state.
  """
  if n_iter <= 0:
    print("No new iterations requested. Skipping BO run.")
    return

  print(f"Running {n_iter} new Bayesian Optimization iterations...")

  # Use maximize() to run the loop. It will use the optimizer's
  # internal function (self._f) and register points automatically.
  optimizer.maximize(
    init_points=0,  # We already loaded or registered.
    n_iter=n_iter
  )

  # --- Save the updated state ---
  # We save the state *after* the optimization run is complete.
  print(f"\nSaving updated optimizer state to: {bo_state_file}")
  optimizer.save_state(str(bo_state_file))

  print("\nOptimization Complete!")
  print("--- Best Result ---")
  print(optimizer.max)

# ---------------------
# Section 5: Evaluation
# ---------------------

def zca_decorrelate_rows(U, eps=1e-8, keep_rms=True):
  """
  U: shape (n_agents, T). Rows are agent signals u_a(t).
  Returns U_decorr with zero-mean rows and lag-0 cross-cov ≈ 0.
  """
  Uc = U - U.mean(axis=1, keepdims=True)                       # center rows
  C = (Uc @ Uc.T) / (Uc.shape[1] - 1)                          # row covariance (n_agents x n_agents)
  # Eigen-decomp
  eigvals, eigvecs = np.linalg.eigh(C)
  D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + eps))
  W = eigvecs @ D_inv_sqrt @ eigvecs.T                          # ZCA whitening matrix
  Uw = W @ Uc                                                   # decorrelated (unit-ish variance)
  if keep_rms:
    # restore each row's original RMS (using centered RMS)
    rms_src = np.sqrt((Uc**2).mean(axis=1, keepdims=True) + eps)
    rms_tgt = np.sqrt((Uw**2).mean(axis=1, keepdims=True) + eps)
    scale = rms_src / rms_tgt
    Uw = Uw * scale
  return Uw

def plot_optimal_signals(
  best_params, mean_func, agents,
  n_steps, FS, start, stop, CONFIG
):
  """
  Generates and plots one representative instance of the optimized
  emissions and temperature time series.
  """

  # Generate one set of noise
  base_noise = utils_ML.generate_uncorrelated_signals(len(agents), n_steps)
  years = np.arange(start, stop)
  U = np.zeros((len(agents), n_steps), dtype=float)

  # Reconstruct the signals using the best parameters
  for i, agent in enumerate(agents):
    mean_func_params = {
          k.replace('m_param_', ''): v
          for k, v in best_params.items()
          if k.startswith(f'm_param_') and k.endswith(f'_{agent}')
      }
    mean_func_params_clean = {k.replace(f'_{agent}', ''): v for k, v in mean_func_params.items()}
    m_t = mean_func(np.arange(n_steps), **mean_func_params_clean)

    if not CONFIG['USE_WHITE_NOISE'] or CONFIG['CHOSEN_MEAN_FUNCTION'] == 'ornstein_uhlenbeck':
      u_t = m_t
    else:
      sigma = best_params[f'sigma_{agent}']
      f_low = best_params[f'f_low_{agent}']
      f_high = best_params[f'f_high_{agent}']
      nyquist = 0.5 * FS

      # Calculate filter coefficients once
      b, a = signal.butter(4, [f_low/nyquist, f_high/nyquist], btype='band')

      w_t_filtered = signal.filtfilt(b, a, base_noise[i])
      w_t_filtered = (w_t_filtered - np.mean(w_t_filtered)) / np.std(w_t_filtered)
      u_t = m_t + sigma * w_t_filtered

    U[i] = u_t

  #U_decorr = zca_decorrelate_rows(U, eps=1e-8, keep_rms=True)
  emissions_dict = {agent: U[i] for i, agent in enumerate(agents)}

  # --- Plotting ---
  fig, axes = plt.subplots(len(agents) + 1, 1, figsize=(12, 6 * (len(agents) + 1)), sharex=True)

  # Plot emissions
  for i, agent in enumerate(agents):
    ax = axes[i]
    ax.plot(years, emissions_dict[agent], color='dodgerblue')
    ax.set_title(f'Forcing Signal for {agent}')
    ax.set_ylabel('Emissions')
    ax.grid(True, alpha=0.4)
    ax.set_xlim([1750, 2500])

  # --- Run FaIR and plot temperature ---
  f_train = run_fair.build_fair(start, stop)
  run_fair.set_emissions(f_train, years, run_fair.ensure_all_agents(emissions_dict, n_steps))
  f_train.run(progress=False)
  _, _, training_temp, _ = run_fair.run_fair(f_train, years)

  ax_temp = axes[-1]
  ax_temp.plot(years, training_temp, color='red')
  ax_temp.set_title('Resulting Temperature Anomaly')
  ax_temp.set_ylabel('Temperature (°C)')
  ax_temp.set_xlabel('Year')
  ax_temp.grid(True, alpha=0.4)

  plt.tight_layout(rect=[0, 0, 1, 0.96])
  plt.show()

  return emissions_dict, training_temp

def full_evaluation(
  emissions_dict_opt, delT_opt,
  scenarios_tier1, series_tier1,
  test_scenarios, test_series,
  agents
):
  """
  Trains one LSTM model on the provided optimal data and evaluates it
  against all specified test scenarios.

  (Refactored from user's 'get_detailed_rmse_breakdown' and assumptions)
  """
  train_group_bracket = ['H-ext', 'VLLO-ext']
  train_group_all_cmip7 = scenarios_tier1.copy()
  training_groups = {
    'Bracket': train_group_bracket,
    'All_CMIP7': train_group_all_cmip7,
    'best_run': ['best_run']
  }
  # Format as (emissions_dict, temp_series, is_historical)
  train_scenarios = scenarios_tier1 + ['best_run']
  train_series = series_tier1 + [(emissions_dict_opt, delT_opt, False)]

  # --- Train and evaluate the model ---
  # This calls the function from your utils_ML.py file
  error_dict, yhat_dict = utils_ML.evaluate_LSTM(
    train_scenarios=train_scenarios,
    train_series=train_series,
    test_scenarios=test_scenarios,
    test_series=test_series,
    agents=agents,
    training_groups=training_groups,
    n_lags=0,
    lstm_size=16,
    dense_size=32,
    dropout_rate=0.2,
    l2_penalty=0.01
  )

  print("Training and evaluation complete.")
  return error_dict, yhat_dict, list(training_groups.keys())

def plot_emulator_performance_grid(yhat_dict, scenarios_train, scenarios_test, series_test, start):
  """
  Plots a grid of subplots comparing the emulator's predictions (yhat)
  against the true temperature series for each test scenario.
  """

  model_styles = {
    'best_run': {'color': 'red', 'linestyle': '-', 'label': 'Emulator (Optimized)'},
    'Bracket': {'color': 'dodgerblue', 'linestyle': '--', 'label': 'Emulator (Bracket)'},
    'All_CMIP7': {'color': 'orange', 'linestyle': ':', 'label': 'Emulator (All_CMIP7)'},
  }
  true_style = {'color': 'k', 'linestyle': '-', 'label': 'FaIR (True)', 'linewidth': 2}

  # Filter out scenarios for which we don't have predictions
  scenarios_to_plot = scenarios_test.copy()
  series_map = {name: data for name, data in zip(scenarios_test, series_test)}

  n_plots = len(scenarios_to_plot)
  if n_plots == 0:
    print("No predictions found to plot.")
    return

  # --- Create the subplot grid ---
  # Aim for a roughly square grid
  n_cols = math.ceil(math.sqrt(n_plots))
  n_rows = math.ceil(n_plots / n_cols)

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4),
                            squeeze=False, constrained_layout=True, sharey=True)

  axes_flat = axes.flatten()

  for i, scenario_name in enumerate(scenarios_to_plot):
    ax = axes_flat[i]

    # Get true temperature (y_true)
    # Data format is (emissions_dict, temp_series, is_historical)
    _, y_true, _ = series_map[scenario_name]
    n_steps_true = len(y_true)
    if scenario_name == 'historical' or 'CO2' in scenario_name or 'CH4' in scenario_name or 'Sulfur' in scenario_name:
      start = 1750
    else:
      start = 2025
    years_true = np.arange(start, start + n_steps_true)
    ax.plot(years_true, y_true, **true_style)
    ax.set_xlim([years_true[0], years_true[-1]])

    title_lines = f"Test: {scenario_name}"

    for train_name in scenarios_train:
      # Get predicted temperature (y_pred)
      y_pred = yhat_dict[train_name][scenario_name]
      n_steps_pred = len(y_pred)
      years_pred = np.arange(start, start + n_steps_pred)

      # Get style
      style = model_styles.get(train_name, {})

      # Plot
      ax.plot(years_pred, y_pred, **style)

    ax.set_title(title_lines)
    if i > n_cols * (n_rows - 1):
      ax.set_xlabel('Year', fontsize=16)
    if i % n_cols == 0:
      ax.set_ylabel('Temperature (°C)', fontsize=16)
    ax.grid(True, alpha=0.4)

  handles, labels = axes_flat[0].get_legend_handles_labels()
  axes_flat[0].legend(handles, labels, fontsize=16)

  # Hide any unused subplots
  for i in range(n_plots, len(axes_flat)):
    axes_flat[i].axis('off')