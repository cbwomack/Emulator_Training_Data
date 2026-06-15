# Experiments for identifying the optimal choice of climate emulator training data

In this repository, you'll find a series of experiments that leverage a differentiable simple Climate Model (SCM) to compute the optimal set of training data for a neural network emulator. This is the companion code to "Optimal scenario design for climate emulation: How to train your emulator" (soon to be submitted).

# Usage
## Jupyter notebooks and code
All code for the differentiable SCM, and emulator training and testing is written in python, specifically within the Jupyter notebooks included in this repo. A large portion of this code relies on [JAX](https://docs.jax.dev/en/latest/quickstart.html) for optimization and automatic differentiation (autodiff), along with [cmcrameri](https://www.fabiocrameri.ch/colourmaps/) for plotting using accessible color maps.

Notebooks are organized as:
1. Generate scenario output data (temperature) using FaIR. This is used in the optimization in step 2.
2. (a) Calibrate the differentiable simple climate model based on FaIR. (b) Create the sulfur injection profile required for GeoMIP; uses the calibrated SCM with autodiff to compute this. (c) Calibrates a second version of the differentiable SCM to match the MIT Earth System Model (MESM) global mean temperature response.
3. Uses the differentiable SCM to optimize training emissions data for a simple neural network emulator. (a - e) are single-forcing experiments, while (f - g) are multi-forcing experiments.
4. (a) Repeats the optimization for the MESM-calibrated SCM; this is ultimately not used in the final work, but left for completeness. (b) Process the MESM zonal temperature data to make the format compatible with the emulator framework. (c) Evaluate the performance of the zonal MESM emulator.
5. Create all plots for the manuscript.

There are also notebooks for the supplemental material that start with the prefix "SI". The "utils" files contain various helper functions that are necessary for experiments, plotting, etc.

## Data
Data required to run FaIR is uploaded here under /data/FaIR/, while MESM output data will be uploaded to Zenodo for publication as the files are larger than recommended to upload on github.
