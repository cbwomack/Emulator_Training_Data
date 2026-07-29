# Experiments for identifying the optimal choice of climate emulator training data

In this repository, you'll find a series of experiments that leverage a differentiable simple climate model (SCM) to compute the optimal set of training data for a neural-network climate emulator. This is the companion code to "Optimal scenario design for climate emulation: How to train your emulator" (soon to be submitted).

# Usage
## Scripts and Notebooks
All code for the differentiable SCM, and emulator training and testing, is written in Python, split between `.py` scripts and Jupyter notebooks. A large portion of this code relies on [JAX](https://docs.jax.dev/en/latest/quickstart.html) for optimization and automatic differentiation (autodiff), along with [cmcrameri](https://www.fabiocrameri.ch/colourmaps/) for plotting using accessible color maps.

Every notebook that involves real compute (FaIR data generation, SCM calibration, or the bilevel inverse-optimization used to pick training emissions) follows a two-piece pattern: a `.py` script under `scripts/` that runs and checkpoints the actual computation, so it can be scheduled as a batch job and resumed if interrupted, and a companion notebook of the same name that loads the results and produces the corresponding figures.

Notebooks and scripts are organized as:
1. `1_generate_FaIR_data`: Generates scenario emissions/temperature data using FaIR (ScenarioMIP tier1/tier2, DECK, GeoMIP, and CS3 "Global Change Outlook 2025" scenarios). This is used throughout the rest of the pipeline.
2. `2a_calibrate_FaIR_JAX` / `2b_optimize_GeoMIP` / `2c_calibrate_MESM`: (a) Calibrates the differentiable SCM against FaIR. (b) Solves for the sulfur-injection emissions profile required for the GeoMIP scenarios, using the calibrated SCM's autodiff. (c) Calibrates a second, MESM-targeted version of the SCM in two sequential steps - carbon cycle (emissions -> concentrations) first, then climate sensitivity (concentrations -> temperature), continuing from the carbon-cycle result.
3. `3a_inverse_CO2_only` / `3b_inverse_all_agents`: Uses the differentiable SCM's autodiff to optimize training emissions for a neural-network emulator - single-agent (CO2-only) and multi-agent versions, respectively. The single-agent CH4/N2O/Sulfur/BC variants and an all-agents-subset variant live in `supplementary_notebooks/` (`SIa`-`SIe`).
4. `4a_inverse_CO2_only_MESM` / `4b_process_MESM_data` / `4c_evaluate_MESM_emulator`: (a) Repeats the CO2-only inverse optimization for the MESM-calibrated SCM. (b) Aggregates MESM's raw zonal-temperature ensemble output into the format the emulator framework expects. (c) Evaluates the resulting zonal-output emulator's performance against MESM.
5. `5a_paper_plots`: Produces every figure in the manuscript.

`supplementary_notebooks/` holds the material for the supplement, prefixed `SI`, plus `gen_MESM_data` (writes the emissions input files MESM itself is driven by) and `SI_plots` (supplemental figures). `utils_FaIR_JAX.py`, `utils_inverse.py`, and `utils_plotting.py` hold the helper functions shared by every notebook and script above - JAX SCM physics/calibration, inverse-optimization and dataset construction, and plotting, respectively - each organized into section banners matching the pipeline parts described above. `run_fair.py` is the FaIR-driving counterpart to `utils_FaIR_JAX.py`.

## Tests
`tests/` holds a pytest suite over the pipeline's most load-bearing, non-plotting code: Tier 1 covers the pure JAX SCM physics core (`utils_FaIR_JAX.py`), Tier 2 the most-reused notebook-facing functions (`utils_inverse.py`), and Tier 3 the vector (zonal-output) emulator path used by Part 4. Run with `pytest tests/` from the repo root.

## Environment
This repo was developed against the conda environment specified in `environment.yml`. <a href="https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment">See this tutorial for instructions on loading an environment from a .yml.</a>

## Data
```
data/
├── CS3_outlook25/     # Raw "Global Change Outlook 2025" AA/CT scenario files (.fordata, .nc)
├── FaIR/               # FaIR's own reference calibration/config files, required to run FaIR itself
├── FaIR_IO/
│   ├── emissions/      # Cached ScenarioMIP tier1/tier2 emissions, keyed by scenario set + agents
│   └── delT/            # Cached per-scenario FaIR-simulated GMST anomaly, same keying
├── GeoMIP/              # Solved sulfur-injection emissions profile for the GeoMIP scenarios
├── saved_emissions/     # Cached GeoMIP/CS3 emissions dicts
├── JAX_calibration/*    # JAX SCM calibration checkpoints (FaIR- and MESM-targeted)
├── MESM/*                # Raw MIT Earth System Model ensemble output (concentration- and emissions-driven)
├── SI_results/*          # Supplementary sensitivity-sweep results (initial condition / architecture / features)
└── plotting/*             # Cached data-aggregation pickles consumed directly by 5a_paper_plots.ipynb
```
`*` = gitignored - regenerated on demand by the corresponding `scripts/*.py` entry point (`FaIR_IO/`'s two subfolders are also just caches, but are committed since generating them requires real FaIR runs worth sharing rather than re-running per clone). `checkpoints/` (inverse-optimization checkpoints) and `Figures/` (rendered figures) are gitignored the same way, regenerated by running the relevant script/notebook rather than committed.
