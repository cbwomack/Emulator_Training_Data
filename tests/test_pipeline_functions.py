# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5 and Gemini 3.1 Pro.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Tier 2: the most-reused notebook-facing pipeline functions in utils_inverse.py
(reuse counts from the Phase 4 planning pass: generate_eval_data in 12 files,
generate_init_params_and_train_data in 11, generate_and_eval_baseline_emulator
in 10, optimize_emissions_inverse in 9). These are expensive to run at full
scale, so:
  - generate_eval_data/generate_init_params_and_train_data/generate_and_eval_
    baseline_emulator are exercised once each (module-scoped fixture) against
    real repo data, restricted to a single agent (CO2) to keep runtime down -
    they wrap real FaIR/JAX data loading and don't expose a "small T" knob.
  - optimize_emissions_inverse is exercised with fully synthetic emissions and
    tiny num_updates/K_inner/T, since it only needs an emis_dict shaped
    correctly, not real scenario data.

These are regression/characterization tests of current behavior (output
shape, type, and invariants like "loss stays finite" or "checkpoint round-
trips exactly"), not correctness proofs.
"""
import os

import jax
import numpy as np
import pytest

import utils_inverse

AGENTS_CO2 = ["CO2"]
ACTIVE_CO2 = ("CO2",)


@pytest.fixture(scope="module")
def co2_pipeline_data():
    # Real repo data, single agent, to keep the (fixed-cost, JIT-heavy) K=400
    # inner MLP training in generate_init_params_and_train_data affordable.
    params0, emis_dict_train_JAX = utils_inverse.generate_init_params_and_train_data(
        AGENTS_CO2, ACTIVE_CO2, test_scen="historical", hidden_sizes=[8], idx_demo=None, verbose=False
    )
    eval_sets, *_ = utils_inverse.generate_eval_data(AGENTS_CO2, DECK=False, CS3=False, DAMIP=False, GeoMIP=False)
    return {
        "params0": params0,
        "emis_dict_train_JAX": emis_dict_train_JAX,
        "eval_sets": eval_sets,
    }


# ---------------------------
# generate_init_params_and_train_data
# ---------------------------

def test_generate_init_params_and_train_data_params_shape(co2_pipeline_data):
    params0 = co2_pipeline_data["params0"]
    # hidden_sizes=[8] -> 2 layers: (input_dim, 8), (8, 1); input_dim is
    # whatever the feature engineering produces, not asserted here.
    assert len(params0) == 2
    input_dim = params0[0]["W"].shape[0]
    assert params0[0]["W"].shape == (input_dim, 8)
    assert params0[0]["b"].shape == (8,)
    assert params0[1]["W"].shape == (8, 1)
    assert params0[1]["b"].shape == (1,)


def test_generate_init_params_and_train_data_train_dict_structure(co2_pipeline_data):
    emis_dict_train_JAX = co2_pipeline_data["emis_dict_train_JAX"]
    assert "historical" in emis_dict_train_JAX
    for scen, arr in emis_dict_train_JAX.items():
        arr = np.asarray(arr)
        assert arr.ndim == 2
        assert arr.shape[0] == 5  # rows padded/aligned to AGENTS_DEFAULT order


# ---------------------------
# generate_eval_data
# ---------------------------

def test_generate_eval_data_structure(co2_pipeline_data):
    eval_sets = co2_pipeline_data["eval_sets"]
    assert {"Tier 1", "Tier 2", "All"}.issubset(eval_sets.keys())
    for set_name, emis_dict in eval_sets.items():
        assert len(emis_dict) > 0
        for scen, arr in emis_dict.items():
            assert np.asarray(arr).shape[0] == 5


def test_generate_eval_data_all_is_union_of_tier1_and_tier2(co2_pipeline_data):
    eval_sets = co2_pipeline_data["eval_sets"]
    combined_keys = set(eval_sets["Tier 1"].keys()) | set(eval_sets["Tier 2"].keys())
    assert combined_keys.issubset(eval_sets["All"].keys())


# ---------------------------
# generate_and_eval_baseline_emulator
# ---------------------------

def test_generate_and_eval_baseline_emulator_output_structure(co2_pipeline_data):
    baseline_results, baseline_pred_delT, ground_truth_delT = utils_inverse.generate_and_eval_baseline_emulator(
        co2_pipeline_data["emis_dict_train_JAX"], co2_pipeline_data["eval_sets"], hidden_sizes=[8]
    )

    assert "Tier 1" in baseline_results
    assert "mean" in baseline_results["Tier 1"]

    for eval_set, per_scen in baseline_results.items():
        mean_nrmse = per_scen["mean"]
        assert np.isfinite(mean_nrmse)
        assert mean_nrmse >= 0.0
        # every non-'mean' entry is a finite, non-negative NRMSE
        for scen, val in per_scen.items():
            if scen == "mean":
                continue
            assert np.isfinite(val)
            assert val >= 0.0

    # predictions/ground truth are keyed the same way as the results
    assert set(baseline_pred_delT.keys()) == set(baseline_results.keys())
    assert set(ground_truth_delT.keys()) == set(baseline_results.keys())


# ---------------------------
# optimize_emissions_inverse (synthetic, tiny-scale)
# ---------------------------

@pytest.fixture
def synthetic_inverse_setup():
    T = 20
    agents = ("CO2",)
    key = jax.random.PRNGKey(0)
    params0 = utils_inverse.init_mlp_params(key, input_dim=5, hidden_sizes=[8])
    emis_dict = {"scen_a": (np.random.rand(1, T).astype(np.float32) * 10.0)}
    return {"T": T, "agents": agents, "params0": params0, "emis_dict": emis_dict}


def test_optimize_emissions_inverse_shapes_and_finite_loss(synthetic_inverse_setup):
    s = synthetic_inverse_setup
    out = utils_inverse.optimize_emissions_inverse(
        s["emis_dict"], s["params0"],
        num_updates=2, step_size=1e2, K_inner=3, lr_inner=5e-2, wd_inner=1e-2,
        agents=s["agents"], active_agents=s["agents"], init_cond="constant",
        T=s["T"], checkpoint_path=None, preds_every=1,
    )

    assert out["updates_done"] == 2
    assert len(out["U_traj"]) == 3  # initial state + 2 updates
    for u in out["U_traj"]:
        assert u["CO2"].shape == (s["T"],)

    errors = np.asarray(out["errors"])
    assert errors.shape == (3,)
    assert np.all(np.isfinite(errors))


def test_optimize_emissions_inverse_checkpoint_roundtrip(tmp_path, synthetic_inverse_setup):
    s = synthetic_inverse_setup
    ckpt_path = os.path.join(tmp_path, "ckpt.pkl")

    out = utils_inverse.optimize_emissions_inverse(
        s["emis_dict"], s["params0"],
        num_updates=2, step_size=1e2, K_inner=3, lr_inner=5e-2, wd_inner=1e-2,
        agents=s["agents"], active_agents=s["agents"], init_cond="constant",
        T=s["T"], checkpoint_path=ckpt_path, checkpoint_every=1, preds_every=1,
    )

    assert os.path.isfile(ckpt_path)
    loaded = utils_inverse.load_inverse_ckpt(ckpt_path)

    np.testing.assert_allclose(out["U_traj"][-1]["CO2"], loaded["U_traj"][-1]["CO2"])
    np.testing.assert_allclose(np.asarray(out["errors"]), np.asarray(loaded["errors"]))
    assert loaded["step_count"] == out["updates_done"]
