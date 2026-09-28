# Optimal scenario design for climate emulation

Companion code to "Optimal scenario design for climate emulation" to answer the question, "What is the optimal set of training data for a climate emulator?"

We use a differentiable Simple Climate Model (SCM) to compute the optimal set of training data for a neural-network climate emulator of that SCM. The SCM is written in [JAX](https://docs.jax.dev/en/latest/quickstart.html) and based on FaIRv2.0.0 [(Leach et al., 2020)](https://gmd.copernicus.org/articles/14/3007/2021/). The model being written in JAX makes it automatically differentiable, i.e., we can calculate gradients through the entire chain from data generation/simulation through to emulator training and evaluation. We use this to calculate the sensitivity of the emulator evaluation loss with respect to the input training emissions for the emulator. We take this gradient and optimize the training data itself via gradient descent, maximizing emulator performance on the evaluation dataset. See Figure 1 of the [preprint manuscript](https://arxiv.org/abs/2606.19302) for a graphical overview of the optimization process.

Our results indicate that the optimization process yields a new set of emissions scenarios that are extremely distinct from any standard climate scenarios (e.g., ScenarioMIP). Training an emulator with these optimized scenarios gives widespread extrapolative performance improvements, suggesting that the optimized emissions pathways may isolate more physically salient features than standard scenarios. You can check out the full results [here]((https://arxiv.org/abs/2606.19302))!

## Setup

```bash
uv pip install -r requirements.txt
uv pip install -e .
```

Scripts are written to be run from the repository root, e.g.,

```bash
python pipeline/04_optimization/04a_inverse_CO2_only.py
```

A conda environment is also provided in `environment.yml`.

## Getting starting

Both notebooks at the top level are self-contained worked examples of the method. Each one trains a baseline emulator from [ScenarioMIP-CMIP7](https://gmd.copernicus.org/articles/19/2627/2026/), then loads the optimized training emissions produced by the corresponding pipeline script and compares the two.

| Notebook | What it optimizes |
|---|---|
| `example_CO2_only.ipynb` | CO2-only emissions trajectory. |
| `example_multi_forcing.ipynb` | Combined CO2, CH4, N2O, sulfur and black carbon emissions trajectory. |

Both follow the same four steps:

1. **Set up.** Choose the forcing agents and emulator architecture, and generate the initial parameters and the evaluation scenario sets.
2. **Train the baseline emulator** on the ScenarioMIP-CMIP7 Priority 1 scenarios, using independently tuned hyperparameters.
3. **Load the optimized emissions.** The notebooks read cached optimal scenario data. To regenerate one, run `pipeline/04_optimization/04a_inverse_CO2_only.py` (or `04b_inverse_all_agents.py`) and point the cell at the resulting file.
4. **Compare** the optimized emulator against the baseline on held-out scenarios.

To explore the code functionality, you can change the following primary levers: `agents` (which agents enter the temperature simulation), `active_agents` (which ones optimize over), `hidden_sizes` (emulator architecture), and `save_path` (set it to cache the trained baseline). In the pipeline scripts, `num_updates` sets the number of outer-loop iterations and `step_size` the size of each update to the emissions trajectory. The multi-agent case needs a per-agent `step_size`, because the optimization is sensitive to the relative scaling between agents.

## Repository layout

```
src/            JAX SCM physics and calibration (utils_FaIR_JAX.py), inverse
                optimization and dataset construction (utils_inverse.py),
                plotting (utils_plotting.py), FaIR driver (run_fair.py),
                repo-root paths (paths.py)

pipeline/
  01_scenario_data/     FaIR scenario generation               (Supplement S5)
  02_scm_calibration/   SCM calibration, FaIR- and MESM-target (Supplement S2)
  03_hyperparameters/   Hyperparameter search and checkpoints  (Supplement S3)
  04_optimization/      Training-data optimization             (Supplement S1)
  05_mesm/              Extension to MESM                      (Supplement S4)
  06_sensitivity/       Sensitivity analyses                   (Supplement S6)
  07_results/           Evaluation, caches and tables          (Supplement S7)
  08_plotting/          paper_plots.ipynb   main-text figures
                        SI_plots.ipynb      supplement figures

data/           Small reference/config files tracked in git; the full archive (checkpoints and larger caches) is downloaded separately
  CS3_outlook25/    Raw CS3 "Global Change Outlook 2025" AA/CT scenario files
  FaIR/             FaIR's own reference calibration and config files
  FaIR_IO/          Cached ScenarioMIP emissions and FaIR-simulated GMST
  GeoMIP/           Solved sulfur-injection profile for the GeoMIP scenarios
  saved_emissions/  Cached GeoMIP/CS3 emissions dicts

Figures/        Figure output, written by the notebooks in 08_plotting
tests/          pytest suite over the non-plotting pipeline code
```

Every stage of the pipeline is designed as a `.py` script that runs and checkpoints the computation so it can be scheduled and resumed, and a companion notebook of the same name that loads the result and plots it.

## Figures

`pipeline/08_plotting/paper_plots.ipynb` produces the main-text figures and `pipeline/08_plotting/SI_plots.ipynb` the supplement figures, each written into `Figures/` under the filename the manuscript includes. Running either notebook top to bottom regenerates its whole set; note that these functions may require data to be regenerated if you do not download the full datasets and optimization checkpoints (see below). Plots use [cmcrameri](https://www.fabiocrameri.ch/colourmaps/) for accessible colour maps.

## Tests

```bash
pytest tests/
```

Unit tests for the following components: the pure JAX SCM physics core, the most-reused pipeline functions, the vector (zonal-output) emulator path used for MESM, and the out-of-distribution scenario registry.

## Data

Only a few, small files from `data/` (required to run this project) are hosted on git. The rest of the data, along with `checkpoints/` (saved output from this project), can be found on [Zenodo](doi.org/10.5281/zenodo.22875971):

```
data/
├── CS3_outlook25/   Raw "Global Change Outlook 2025" AA/CT scenario files
├── FaIR/            FaIR's own reference calibration and config files
├── FaIR_IO/         Cached ScenarioMIP emissions and FaIR-simulated GMST
├── GeoMIP/          Solved sulfur-injection profile for the GeoMIP scenarios
├── RCMIP/           RCMIP emissions, used to build the out-of-distribution scenarios
├── saved_emissions/ Cached GeoMIP/CS3 emissions dicts
├── JAX_calibration/ SCM calibration checkpoints, FaIR- and MESM-targeted
├── MESM/            MIT Earth System Model ensemble output
├── SI_results/      Sensitivity sweeps, hyperparameter searches, seed spreads
└── plotting/        Aggregation pickles read directly by the figure notebooks
```
