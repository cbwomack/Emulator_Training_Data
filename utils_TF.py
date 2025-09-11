# Required imports
## Basic imports
import numpy as np

## Math
from scipy import signal
from scipy.fft import ifft, rfft, rfftfreq

## Plotting
import matplotlib.pyplot as plt
from cmcrameri import cm

## Setup plots
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "sans-serif",
  "font.sans-serif": ["Helvetica Light"],
})

###########################
## System Identification ##
###########################

def identify_system(input, output, fs, nperseg, one_sided=False):
  # System Identification using Welch's Method

  # One-Sided Approach ---
  if one_sided:
    f, Pxx = signal.welch(input, fs=fs, nperseg=nperseg, return_onesided=True)
    _, Pxy = signal.csd(input, output, fs=fs, nperseg=nperseg, return_onesided=True)
    H_est = Pxy / Pxx

    # Reconstruct the full symmetric spectrum before IFFT
    H_full = np.zeros(nperseg, dtype=complex)
    H_full[:nperseg // 2 + 1] = H_est
    H_full[nperseg // 2 + 1:] = np.conj(H_est[1:-1][::-1])
    H_est = H_full

  # Two-Sided Approach
  else:
    f, Pxx = signal.welch(input, fs=fs, nperseg=nperseg, return_onesided=False)
    _, Pxy = signal.csd(input, output, fs=fs, nperseg=nperseg, return_onesided=False)
    H_est = Pxy / Pxx

  # Convert Estimated Transfer Functions to Time Domain (IRF)
  h_est = ifft(H_est).real

  return H_est, h_est, f

#################################
## System Identification Plots ##
#################################

def plot_freq(H_est, f, one_sided=False, plot_true=False, H_true=None, f_true=None):

  # Frequency Domain Plots
  fig, ax = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

  # Plot Magnitude and Phase
  if plot_true:
    ax[0].loglog(f_true, np.abs(H_true), c=cm.batlowS(0), ls='-', lw=3, label='True Magnitude')
    ax[1].semilogx(f_true, np.angle(H_true, deg=True), c=cm.batlowS(0), ls='-', lw=3, label='True Phase')

  if one_sided:
    ax[0].loglog(f, np.abs(H_est), c=cm.batlowS(4), ls='--', label='Estimated (One-Sided)')
    ax[1].semilogx(f, np.angle(H_est, deg=True), c=cm.batlowS(4), ls='--', label='Estimated (One-Sided)')
  else:
    ax[0].loglog(np.fft.fftshift(f), np.fft.fftshift(np.abs(H_est)), c=cm.batlowS(4), ls='--', lw=2, label='Estimated (Two-Sided)')
    ax[1].semilogx(np.fft.fftshift(f), np.fft.fftshift(np.angle(H_est, deg=True)), c=cm.batlowS(4), ls='--', lw=2, label='Estimated (Two-Sided)')

  ax[0].set_title("Transfer Function Magnitude")
  ax[0].set_ylabel("|H(f)| [K / (W m$^{-2}$)]")
  ax[0].grid(True, which='both')
  ax[0].legend()

  ax[1].set_title("Transfer Function Phase")
  ax[1].set_xlabel("Frequency [Hz]")
  ax[1].set_ylabel("Phase [degrees]")
  ax[1].grid(True, which='both')
  ax[1].legend()

  return

def plot_time(h_est, one_sided=False, plot_true=False, h_true=None):

  fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)

  # Impulse Response Plot
  if plot_true:
    ax.plot(h_true, c=cm.batlowS(0), ls='-', lw=3, label='True Impulse Response')

  if one_sided:
    ax.plot(h_est, c=cm.batlowS(4), ls='--', label='Estimated IRF (One-Sided)')
  else:
    ax.plot(h_est, c=cm.batlowS(4), ls='--', label='Estimated IRF (Two-Sided)')

  ax.set_title("Impulse Response Function")
  ax.set_xlabel("Time [Years]")
  ax.set_ylabel("h(t) [K / (W m-2)]")
  ax.grid(True)
  ax.legend()

  return