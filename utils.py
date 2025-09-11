# Required imports
## Basic imports
import numpy as np
import xarray as xr
import pickle

## Math
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.linalg import toeplitz
from scipy.sparse.linalg import spsolve_triangular
from scipy.optimize import minimize

## Optimization
import jax
import jax.numpy as jnp
from jax.nn import softplus, sigmoid

## Plotting
import matplotlib.pyplot as plt
import seaborn as sns


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

def toeplitz_rank(u, L):
  m, N  = u.shape
  Hcols = N - L + 1
  H = np.vstack([u[:, i:i+Hcols] for i in range(L)])
  return np.linalg.matrix_rank(H)

def compute_Fisher_individual(u, t, k, sigma):
  # u = forcing vector
  # t = current lag
  # k = number of parameters
  # sigma = variance
  N_pad = k - 1 # zeros needed on the left
  u_pad = np.concatenate((np.zeros(N_pad), u))
  idx = t + N_pad # position of u_t in padded series
  lag = u_pad[idx - (k - 1):idx + 1][::-1]

  return np.outer(lag, lag) / sigma**2

def compute_Fisher_full(u, sigma):
  k = len(u)
  F_full = np.zeros((k, k))
  for t in range(k):
    F_full += compute_Fisher_individual(u, t, k, sigma)

  return F_full

def compute_A_optimality(F):
  # Want to minimize trace of the inverse of the
  # information matrix. Equivalent to minimizing
  # the average variance of the estimates of the
  # regression coefficients.

  return np.trace(np.linalg.inv(F))

def compute_D_optimality(F):
  # Want to maximize determinant of the
  # information matrix. Equivalent to maximizing
  # differential Shannon information content
  # of the parameter estimates.

  return np.linalg.slogdet(F)

def compute_E_optimality(F):
  # Want to maximize the minimum eigenvalue of the
  # information matrix. Aims to make the least certain
  # parameter estimate as precise as possible

  eigenvals, eigenvecs = np.linalg.eig(F)
  return np.min(eigenvals)

def get_optimality_conditions(Fisher, sigma, verbose=True):
  A = compute_A_optimality(Fisher)
  D = compute_D_optimality(Fisher)
  E = compute_E_optimality(Fisher)

  if verbose:
    print(f'\tA optimality - minimize {A}')
    print(f'\tD optimality - maximize {D}')
    print(f'\tE optimality - maximize {E}')

  return A, D, E

#####################
## Three Box Model ##
#####################

class three_box_ebm:
  def __init__(self, lam, C, D, T0, dt):
    self.C    = C                     # Heat capacity [J kg-1 K-1]
    self.D    = D                     # Diffusivity [W m-2 K-1]
    self.dt   = dt                    # Time step [seconds]
    self.lam  = self.make_lam(lam, D) # Feedback parameter [W m-2 K-1]
    self.T0   = T0                    # Initial temperature [K]

  def make_lam(self, lam_diag, D=0.0):
    lam = np.asarray(lam_diag, dtype=float)
    lam = np.diag(lam)

    if D != 0.0:
      lam += np.diag([D, 2*D, D])
      lam[0, 1] = lam[1, 0] = -D
      lam[1, 2] = lam[2, 1] = -D

    return lam

  def box_iter(self, F, Nt_year):
    B = np.eye(len(self.C))
    F_interp, Nt_interp = interpolate(F, self.dt)
    T_out = np.empty((3, Nt_interp))
    T_out[:, 0] = self.T0

    for n in range(1, Nt_interp):
      Ft_interp = F_interp[:,n - 1]

      RHS = 1 / self.C * (-self.lam @ T_out[:, n - 1] + B @ Ft_interp)
      T_out[:, n] = T_out[:, n - 1] + self.dt * RHS

    return T_out

  def plot_box(self, T_out):
    plt.plot(T_out.T)
    return

def get_regularization(w, F):
  """
  Estimate ridge regularization hyper-parameters by
  minimising a data-misfit cost.

  Parameters
  ----------
  w : ndarray
    Target response, shape (n_space, n_time).
  F : ndarray
    Forcing time series, shape (1, n_time) or (n_time,).

  Returns
  -------
  ndarray
    Optimized hyper-parameters [sig2, lam2].
  """

  # Random initial guess for the two hyper-parameters
  init_params = np.random.rand(2)

  # Toeplitz representation of forcing for convolution
  F_toep = sparse.csr_matrix(toeplitz(F[0,:], np.zeros_like(F[0,:])))

  # Optimization of the hyper-parameter cost function
  res = minimize(fit_opt_hyper,
                 init_params,
                 args=(w, F_toep),
                 method='L-BFGS-B')

  return res.x

def fit_opt_hyper(params, w, F_toep):
  """
  Negative log-evidence objective for tuning regularisation
  hyper-parameters (sig^2, lam^2).

  Parameters
  ----------
  params : array-like
    Two-element vector [sig2, lam2] to be optimised.
  w : ndarray
    Response data, shape (n_space, n_time).
  F_toep : sparse matrix
    Toeplitz forcing operator (n_time x n_time).

  Returns
  -------
  float
    Value of the objective to minimise.
  """

  # Unpack hyper-parameters
  sig2, lam2 = params

  # Build covariance matrix
  n_x, n_t = w.shape
  Sig = sig2*np.eye(n_t) + lam2*(F_toep @ F_toep.T)

  # Log-determinant of sigma; invalid if sign <= 0
  sign, logdet_Sig = np.linalg.slogdet(Sig)
  if sign <= 0:
    return np.inf

  # Quadratic form
  Sig_inv_w = np.linalg.solve(Sig, w.T)
  quadratic_term = np.sum(w * Sig_inv_w.T)

  # Log-evidence (marginal likelihood) and its negative
  log_evidence_value = -0.5 * (n_x * logdet_Sig + quadratic_term)
  return -log_evidence_value



def plot_single_heatmap(error_metrics: dict,
                        train_scenarios: list[str],
                        test_scenarios: list[str],
                        vmax: float,
                        cmap: str = "Reds",
                        long_title: str = '',
                        add_xlabel: bool = True,
                        add_ylabel: bool = True,
                        add_cbar: bool = False) -> None:

  # Instantiate data array
  fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
  data = np.empty((len(test_scenarios), len(train_scenarios)))

  for i, scen_test in enumerate(test_scenarios):
    for j, scen_train in enumerate(train_scenarios):
      if scen_train == scen_test:
        data[i,j] = np.nan
        continue

      try:
        value = np.mean(error_metrics[scen_train][scen_test])
      except KeyError:
        value = np.nan
      data[i,j] = value

  # Plot the heatmap using the provided axis and vmax
  sns.heatmap(
    data,
    ax=ax,
    cmap=cmap,
    vmin=0,
    vmax=vmax,
    linewidth=0.5,
    annot=True,
    fmt=".3g",
    cbar=add_cbar,
    cbar_kws={"label": r"NRMSE [\%]"} if add_cbar else None
  )

  # Configure labels and title for the subplot
  #ax.set_title(long_title)
  #tick_labels = ['Abr.','Hi. Em.', 'Plat.', 'Over.']

  #if add_xlabel:
  #  ax.set_xticklabels(tick_labels, rotation=45, ha="right")
  #else:
  #  # Hide x-axis labels if not needed (for top row plots)
  #  ax.set_xlabel("")
  #  ax.set_xticklabels([])

  #if add_ylabel:
  #  ax.set_yticklabels(tick_labels, rotation=45)
  #else:
    # Hide y-axis labels if not needed (for right column plots)
  #  ax.set_ylabel("")
  #  ax.set_yticklabels([])

  return

def calc_RMSE(w_true, w_est):
  return np.sqrt(np.mean((w_true - w_est)**2, axis=1))

def calc_NRMSE(w_true, w_est):
  return calc_RMSE(w_true, w_est)/np.abs(np.mean(w_true, axis=1))*100