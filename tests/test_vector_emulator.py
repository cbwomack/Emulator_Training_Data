"""
Tier 3: vector-target (zonal-output) emulator coverage, converted from the
old root-level test_vector_emulator.py (a #%%-cell script, not a real pytest
module) into real pytest test functions.

Fixes one real bug along the way: the original file's final cell called
utils_inverse.generate_and_eval_baseline_emulator_vector, which does not
exist anywhere in utils_inverse.py (confirmed via grep - zero definitions).
The function that actually exists and matches every call site elsewhere in
the codebase (e.g. 4c_evaluate_MESM_emulator.ipynb) is
generate_and_eval_emulator_vector - same kwargs, but it also returns a 5th
value (stats_X) when precomp_stats_X isn't supplied. This was a drop-in
call-site fix, not a deeper API mismatch.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import utils_inverse

AGENTS = utils_inverse.AGENTS_DEFAULT  # ("CO2", "CH4", "N2O", "Sulfur", "BC")
T = 144           # Time steps
OUTPUT_DIM = 46   # Vector-target dimension (e.g. latitude bins)
SCENARIOS = ["ssp126", "ssp370"]
LAT_COORDS = np.linspace(-88, 88, OUTPUT_DIM)


@pytest.fixture(scope="module")
def mock_emis_and_targets():
    rng = np.random.default_rng(0)
    n_agents = len(AGENTS)
    emis_dict_mock = {scen: rng.random((n_agents, T)).astype(np.float32) for scen in SCENARIOS}
    targets_dict_mock = {scen: rng.random((T, OUTPUT_DIM)).astype(np.float32) for scen in SCENARIOS}
    return emis_dict_mock, targets_dict_mock


# ---------------------------
# init_mlp_params_vector / mlp_forward_vector
# ---------------------------

def test_init_mlp_params_vector_shapes():
    key = jax.random.PRNGKey(42)
    input_dim = 25
    hidden_sizes = [16, 16]

    params = utils_inverse.init_mlp_params_vector(key, input_dim, hidden_sizes, OUTPUT_DIM)

    assert params[-1]["W"].shape == (hidden_sizes[-1], OUTPUT_DIM)
    assert params[-1]["b"].shape == (OUTPUT_DIM,)


def test_mlp_forward_vector_output_shape():
    key = jax.random.PRNGKey(42)
    input_dim = 25
    hidden_sizes = [16, 16]
    params = utils_inverse.init_mlp_params_vector(key, input_dim, hidden_sizes, OUTPUT_DIM)

    dummy_input = jnp.ones((5, input_dim))  # batch of 5
    output = utils_inverse.mlp_forward_vector(params, dummy_input)

    assert output.shape == (5, OUTPUT_DIM)


# ---------------------------
# train_mlp_sgd_vector
# ---------------------------

def test_train_mlp_sgd_vector_loss_stays_finite():
    key = jax.random.PRNGKey(42)
    input_dim = 25
    params = utils_inverse.init_mlp_params_vector(key, input_dim, [16, 16], OUTPUT_DIM)

    Xtr = jax.random.normal(key, (100, input_dim))
    ytr = jax.random.normal(key, (100, OUTPUT_DIM))

    _params_trained, losses = utils_inverse.train_mlp_sgd_vector(params, Xtr, ytr, K=10, lr=1e-2)

    assert not jnp.isnan(losses[-1])
    assert jnp.isfinite(losses[-1])


# ---------------------------
# build_dataset_vector_targets
# ---------------------------

def test_build_dataset_vector_targets_alignment(mock_emis_and_targets):
    emis_dict_mock, targets_dict_mock = mock_emis_and_targets

    dataset = utils_inverse.build_dataset_vector_targets(
        emis_dict_mock, targets_dict_mock, SCENARIOS, agents=AGENTS
    )

    assert len(dataset) == len(SCENARIOS)
    X_ex, y_ex, name_ex = dataset[0]

    expected_feat_dim = len(AGENTS) * 5
    assert X_ex.shape[1] == expected_feat_dim
    assert y_ex.shape[1] == OUTPUT_DIM
    assert X_ex.shape[0] == y_ex.shape[0]
    assert name_ex in SCENARIOS


# ---------------------------
# generate_and_eval_emulator_vector (fixed call site, see module docstring)
# ---------------------------

def test_generate_and_eval_emulator_vector_output_structure(mock_emis_and_targets):
    emis_dict_mock, targets_dict_mock = mock_emis_and_targets
    eval_emis_sets = {"Tier 1": emis_dict_mock}
    eval_targets_sets = {"Tier 1": targets_dict_mock}

    results, preds, truths, paramsK, stats_X = utils_inverse.generate_and_eval_emulator_vector(
        emis_dict_train=emis_dict_mock,
        targets_dict_train=targets_dict_mock,
        eval_emis_sets=eval_emis_sets,
        eval_targets_sets=eval_targets_sets,
        output_dim=OUTPUT_DIM,
        lat_coords=LAT_COORDS,
        hidden_sizes=[16],
        K=50,
        verbose=False,
    )

    t1_res = results["Tier 1"]
    global_mean = t1_res["mean"]["global"]
    zonal_mean = t1_res["mean"]["zonal"]

    assert isinstance(global_mean, float)
    assert np.isfinite(global_mean)
    assert zonal_mean.shape == (OUTPUT_DIM,)
    assert "global" in t1_res[SCENARIOS[0]]
    assert "zonal" in t1_res[SCENARIOS[0]]
