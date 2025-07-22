# Required imports
## Basic imports
import numpy as np
import xarray as xr
import pickle

## Math imports
from scipy.interpolate import interp1d

## Optimization
import jax
import jax.numpy as jnp
from jax.nn import softplus, sigmoid


def weighted_global_mean(ds):
  weights = np.cos(np.deg2rad(ds.lat))
  ds_glob = ds.weighted(weights).mean(("lon", "lat"))

  return ds_glob

def interpolate(input, dt):
  # Interpolate if input is a vector
  if len(input.shape) == 1:
    Nt = len(input)
    Nt_interp = int(Nt/dt)
    t_vec = np.linspace(0, Nt - 1, Nt)
    t_vec_interp = np.linspace(0, Nt - 1, Nt_interp)
    interp_func = interp1d(t_vec, input, kind='linear')
    input_interp = interp_func(t_vec_interp)

  # Interpolate if input is a matrix
  else:
    Nb, Nt = input.shape
    Nt_interp = int(Nt/dt)
    input_interp = np.empty((Nb, Nt_interp))
    t_vec = np.linspace(0, Nt - 1, Nt)
    t_vec_interp = np.linspace(0, Nt - 1, Nt_interp)
    for i in range(Nb):
      interp_func = interp1d(t_vec, input[i,:], kind='linear')
      input_interp[i,:] = interp_func(t_vec_interp)

  return input_interp, Nt_interp

###################################
## Optimization Helper Functions ##
###################################

def apply_constraints(params):
  soft_plus_params = jnp.empty(len(params))
  soft_plus_params = soft_plus_params.at[0].set(softplus(params[0]))    # C1,     (0, infty)
  soft_plus_params = soft_plus_params.at[1].set(softplus(params[1]))    # C2,     (0, infty)
  soft_plus_params = soft_plus_params.at[2].set(0.25*sigmoid(params[0])) # delta,  (0, 0.5)
  soft_plus_params = soft_plus_params.at[3].set(softplus(params[3]))    # lam,    (0, infty)
  soft_plus_params = soft_plus_params.at[4].set(sigmoid(params[4]))     # D,      (0, 1)

  return soft_plus_params

def soft_quad_constraint(x, lo, hi):
  # Grows quadratically when x leaves the interval
  return jnp.square(jax.nn.relu(lo - x)) + jnp.square(jax.nn.relu(x - hi))

##############################
## General Helper Functions ##
##############################

def save_results(metrics, name):
  """
  Save a metrics dictionary to disk as a pickle file.

  Parameters
  ----------
  metrics : dict
    Data to be written.
  name : str
    Filename (without ".pkl").

  Returns
  -------
  None
  """

  with open(f'{name}.pkl', 'wb') as file:
    pickle.dump(metrics, file)
  return

def open_results(name):
  """
  Load a pickled object from disk.

  Parameters
  ----------
  name : str
    Filename (without “.pkl”).

  Returns
  -------
  dict
    Data retrieved from disk.
  """

  with open(f'{name}.pkl', 'rb') as file:
    metric = pickle.load(file)
  return metric