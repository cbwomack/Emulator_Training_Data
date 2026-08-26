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


# ---------------------------------------------------------------
# Resume equivalence
# ---------------------------------------------------------------
# The 1000 -> 2000 iteration migration resumes every existing checkpoint rather
# than rerunning from scratch, which is only sound if resuming is EXACTLY
# equivalent to a longer fresh run. Nothing exercised the resume_if_exists
# branch before this. These are bit-exactness assertions, not tolerance checks:
# the momentum trace round-trips through float32 numpy, and the inner-loop PRNG
# key is a pure function of the seed with no step dependence, so any difference
# at all would indicate a real defect rather than accumulated error.

def test_resume_is_bit_exact_with_a_longer_fresh_run(tmp_path, synthetic_inverse_setup):
    s = synthetic_inverse_setup
    common = dict(
        step_size=1e2, K_inner=3, lr_inner=5e-2, wd_inner=1e-2,
        agents=s["agents"], active_agents=s["agents"], init_cond="constant",
        T=s["T"], checkpoint_every=1, preds_every=1,
    )

    fresh = utils_inverse.optimize_emissions_inverse(
        s["emis_dict"], s["params0"], num_updates=4,
        checkpoint_path=os.path.join(tmp_path, "fresh.pkl"), **common)

    resumed_path = os.path.join(tmp_path, "resumed.pkl")
    utils_inverse.optimize_emissions_inverse(
        s["emis_dict"], s["params0"], num_updates=2,
        checkpoint_path=resumed_path, resume_if_exists=False, **common)
    resumed = utils_inverse.optimize_emissions_inverse(
        s["emis_dict"], s["params0"], num_updates=4,
        checkpoint_path=resumed_path, resume_if_exists=True, **common)

    assert resumed["updates_done"] == fresh["updates_done"] == 4
    assert len(resumed["U_traj"]) == len(fresh["U_traj"]) == 5

    # The final iterate is what every downstream artifact reads (U_traj[-1]).
    np.testing.assert_array_equal(
        np.asarray(resumed["U_traj"][-1]["CO2"]), np.asarray(fresh["U_traj"][-1]["CO2"]))
    # And the whole trajectory, so a mid-run divergence cannot hide.
    for k in range(5):
        np.testing.assert_array_equal(
            np.asarray(resumed["U_traj"][k]["CO2"]), np.asarray(fresh["U_traj"][k]["CO2"]),
            err_msg=f"trajectory diverged at iterate {k}")
    np.testing.assert_array_equal(
        np.asarray(resumed["errors"]), np.asarray(fresh["errors"]))


def test_resume_preserves_the_pre_resume_history(tmp_path, synthetic_inverse_setup):
    # The migration relies on the first 1001 entries of an extended checkpoint
    # still being the original run, so the appendix arm remains a true control.
    s = synthetic_inverse_setup
    common = dict(
        step_size=1e2, K_inner=3, lr_inner=5e-2, wd_inner=1e-2,
        agents=s["agents"], active_agents=s["agents"], init_cond="constant",
        T=s["T"], checkpoint_every=1, preds_every=1,
    )
    path = os.path.join(tmp_path, "extend.pkl")

    short = utils_inverse.optimize_emissions_inverse(
        s["emis_dict"], s["params0"], num_updates=2,
        checkpoint_path=path, resume_if_exists=False, **common)
    short_errors = np.asarray(short["errors"]).copy()
    short_final = np.asarray(short["U_traj"][-1]["CO2"]).copy()

    extended = utils_inverse.optimize_emissions_inverse(
        s["emis_dict"], s["params0"], num_updates=4,
        checkpoint_path=path, resume_if_exists=True, **common)

    np.testing.assert_array_equal(np.asarray(extended["errors"])[:3], short_errors)
    np.testing.assert_array_equal(
        np.asarray(extended["U_traj"][2]["CO2"]), short_final)


# ---------------------------------------------------------------
# Recovering true NRMSE from a checkpoint's recorded objective
# ---------------------------------------------------------------
# 'errors' stores nrmse + smoothness_weight * sum(dU)^2, so Figures 3 and 5
# have to subtract the penalty back out. These round-trip the recovery against
# runs whose smoothness_weight is known by construction, rather than only
# against the frozen real checkpoints it was developed on.

def _run_with_smoothness(setup, tmp_path, weight, num_updates=3):
    ckpt_path = os.path.join(tmp_path, f"ckpt_w{weight}.pkl")
    utils_inverse.optimize_emissions_inverse(
        setup["emis_dict"], setup["params0"],
        num_updates=num_updates, step_size=1e2, K_inner=3, lr_inner=5e-2, wd_inner=1e-2,
        agents=setup["agents"], active_agents=setup["agents"], init_cond="constant",
        T=setup["T"], smoothness_weight=weight,
        checkpoint_path=ckpt_path, checkpoint_every=1, preds_every=1,
    )
    import pickle
    with open(ckpt_path, "rb") as f:
        return pickle.load(f)


def test_recover_smoothness_weight_roundtrip(tmp_path, synthetic_inverse_setup):
    weight = 1e-2
    raw = _run_with_smoothness(synthetic_inverse_setup, tmp_path, weight)
    # Solve it back out ignoring the recorded meta, which is the path older
    # checkpoints (written before meta was populated) have to take.
    raw_no_meta = dict(raw, meta={})
    recovered, spread = utils_inverse.recover_smoothness_weight(raw_no_meta)

    assert recovered == pytest.approx(weight, rel=1e-3)
    # Over-determined: every sampled iteration must agree on the same scalar.
    assert spread < 1e-3 * weight


def test_recover_smoothness_weight_zero_when_unregularized(tmp_path, synthetic_inverse_setup):
    raw = _run_with_smoothness(synthetic_inverse_setup, tmp_path, 0.0)
    recovered, _ = utils_inverse.recover_smoothness_weight(dict(raw, meta={}))
    # Snapped to exactly zero, not left as a ratio of rounding errors.
    assert recovered == 0.0


def test_recover_nrmse_trajectory_is_noop_when_unregularized(tmp_path, synthetic_inverse_setup):
    raw = _run_with_smoothness(synthetic_inverse_setup, tmp_path, 0.0)
    nrmse = utils_inverse.recover_nrmse_trajectory(dict(raw, meta={}))
    np.testing.assert_array_equal(nrmse, np.asarray(raw["errors"], dtype=np.float64))


def test_recover_nrmse_trajectory_strictly_below_recorded_objective(tmp_path, synthetic_inverse_setup):
    weight = 1e-2
    raw = _run_with_smoothness(synthetic_inverse_setup, tmp_path, weight)
    errors = np.asarray(raw["errors"], dtype=np.float64)
    nrmse = utils_inverse.recover_nrmse_trajectory(raw)

    assert np.all(nrmse > 0.0)
    assert np.all(nrmse <= errors + 1e-12)
    # The initial iterate is a constant trajectory, so its penalty is exactly
    # zero and errors[0] is already pure NRMSE - a fixed point of the correction.
    assert nrmse[0] == pytest.approx(errors[0], rel=1e-12)
    # A later iterate is no longer constant, so the correction must bite.
    assert nrmse[-1] < errors[-1]


def test_recover_nrmse_trajectory_rejects_wrong_weight(tmp_path, synthetic_inverse_setup):
    raw = _run_with_smoothness(synthetic_inverse_setup, tmp_path, 1e-2)
    # NRMSE is non-negative by construction, so an overlarge weight must be
    # rejected rather than silently returning negative "errors".
    with pytest.raises(ValueError, match="non-positive"):
        utils_inverse.recover_nrmse_trajectory(raw, smoothness_weight=1e3)


# ---------------------------------------------------------------
# Seed-aggregation convention shared by Figures 3, 4 and 5
# ---------------------------------------------------------------

def test_aggregate_seeds_median_returns_median_and_iqr():
    import utils_plotting
    # Skewed on purpose: one large outlier, which is the case the convention
    # exists to handle. mean=24, median=3.
    stacked = np.array([[1.0], [2.0], [3.0], [4.0], [110.0]])
    centre, lo, hi = utils_plotting._aggregate_seeds(stacked, "median")
    assert centre[0] == pytest.approx(3.0)
    assert lo[0] == pytest.approx(2.0)
    assert hi[0] == pytest.approx(4.0)
    # The whole point: the outlier moves the mean far outside the IQR.
    assert stacked.mean() > hi[0]


def test_aggregate_seeds_mean_band_reproduces_each_figures_prior_behaviour():
    import utils_plotting
    stacked = np.array([[1.0], [2.0], [3.0], [4.0], [110.0]])

    centre, lo, hi = utils_plotting._aggregate_seeds(stacked, "mean", mean_band="minmax")
    assert (centre[0], lo[0], hi[0]) == pytest.approx((24.0, 1.0, 110.0))

    centre, lo, hi = utils_plotting._aggregate_seeds(stacked, "mean", mean_band="std")
    assert centre[0] == pytest.approx(24.0)
    assert lo[0] == pytest.approx(24.0 - stacked.std())
    assert hi[0] == pytest.approx(24.0 + stacked.std())


def test_aggregate_seeds_median_band_is_log_safe():
    import utils_plotting
    # Figure 3's y-axis is logarithmic, so a band edge at or below zero cannot
    # render. mean - std goes negative here; the IQR cannot, since both bounds
    # are order statistics of strictly positive data.
    stacked = np.array([[0.01], [0.02], [0.03], [0.04], [5.0]])
    _, mean_lo, _ = utils_plotting._aggregate_seeds(stacked, "mean", mean_band="std")
    _, med_lo, _ = utils_plotting._aggregate_seeds(stacked, "median")
    assert mean_lo[0] < 0.0
    assert med_lo[0] > 0.0


def test_aggregate_seeds_rejects_unknown_convention():
    import utils_plotting
    with pytest.raises(ValueError, match="unknown aggregation"):
        utils_plotting._aggregate_seeds(np.zeros((3, 1)), "iqr")


def test_recover_smoothness_weight_raises_rather_than_silently_returning_zero(
    tmp_path, synthetic_inverse_setup
):
    # Regression: a preds_every that samples nothing used to fall through to
    # "0.0", i.e. "unregularized", silently skipping a real correction. It must
    # fail loudly instead - a wrong weight is worse than no answer here.
    raw = _run_with_smoothness(synthetic_inverse_setup, tmp_path, 1e-2)
    with pytest.raises(ValueError, match="could not sample"):
        utils_inverse.recover_smoothness_weight(raw, preds_every=10_000)


def test_preds_every_is_inferred_from_trajectory_lengths(tmp_path, synthetic_inverse_setup):
    # The stride is exact, not a guess, so recovery works without being told it.
    raw = _run_with_smoothness(synthetic_inverse_setup, tmp_path, 1e-2)
    assert utils_inverse._infer_preds_every(raw) == 1
    recovered, _ = utils_inverse.recover_smoothness_weight(dict(raw, meta={}))
    assert recovered == pytest.approx(1e-2, rel=1e-3)


def test_checkpoint_meta_records_run_config(tmp_path, synthetic_inverse_setup):
    weight = 1e-2
    raw = _run_with_smoothness(synthetic_inverse_setup, tmp_path, weight)
    meta = raw["meta"]

    # meta was plumbed through save/load_inverse_ckpt but never populated, which
    # is why smoothness_weight had to be solved for on every existing artifact.
    assert meta["smoothness_weight"] == weight
    assert meta["K_inner"] == 3
    assert meta["T"] == synthetic_inverse_setup["T"]
    # With meta present the weight is read exactly; without it, it is solved
    # back out of float32-stored predictions and so agrees only to ~1e-4
    # relative. That gap is the reason for recording meta in the first place -
    # the two paths must agree, but only the recorded one is exact.
    np.testing.assert_allclose(
        utils_inverse.recover_nrmse_trajectory(raw),
        utils_inverse.recover_nrmse_trajectory(dict(raw, meta={})),
        rtol=1e-4,
    )


# ---------------------------------------------------------------------------
# Iteration-length independence of the plotting highlights (Stage E)
# ---------------------------------------------------------------------------
def _sel(n_all, max_lines=11):
    """Reproduce the selection utils_plotting builds for a fading history."""
    import numpy as _np
    if n_all <= max_lines:
        return _np.arange(n_all, dtype=int)
    return _np.unique(_np.linspace(0, n_all - 1, num=max_lines, dtype=int))


def test_highlight_indices_track_the_endpoint_at_any_run_length():
    """The final iterate must always be highlighted.

    The old hardcoded `i in [0, 2, 20]` only coincided with the selection at
    1000 iterations; at 2000 index 2 is not selected at all and index 20 is the
    midpoint. Regression guard for that whole class of bug.
    """
    import utils_plotting as up
    for n_preds in (21, 41, 11, 5, 101):        # 1000, 2000, and off-nominal
        sel = _sel(n_preds)
        hl = up._highlight_indices(sel)
        assert len(hl) == 3
        assert hl[0] == int(sel[0]) == 0
        assert hl[-1] == int(sel[-1]) == n_preds - 1, (
            f"final iterate not highlighted for n_preds={n_preds}")
        assert all(h in set(sel.tolist()) for h in hl), (
            "highlighted an index that is never plotted")


def test_highlight_indices_handles_empty_selection():
    import numpy as _np
    import utils_plotting as up
    assert up._highlight_indices(_np.array([], dtype=int)) == ()


def test_preds_stride_recovers_the_true_sampling_interval():
    """preds index -> outer iteration must scale by the real stride."""
    import utils_plotting as up
    # 1000 updates, preds every 50 -> 21 entries; 2000 -> 41.
    assert up._preds_stride({"errors": [0.0] * 1001}, 21) == 50
    assert up._preds_stride({"errors": [0.0] * 2001}, 41) == 50
    # index 20 is step 1000 in both cases - the label must not depend on length
    assert 20 * up._preds_stride({"errors": [0.0] * 1001}, 21) == 1000
    assert 20 * up._preds_stride({"errors": [0.0] * 2001}, 41) == 1000
    # and the last entry names the true endpoint
    assert 40 * up._preds_stride({"errors": [0.0] * 2001}, 41) == 2000


def test_preds_stride_falls_back_to_meta_then_default():
    import utils_plotting as up
    # ragged/unusable errors -> use meta
    assert up._preds_stride({"errors": [0.0] * 7, "meta": {"preds_every": 25}}, 41) == 25
    # nothing usable -> documented default
    assert up._preds_stride({}, 0) == 50


def test_ragged_seed_trajectories_raise_a_diagnostic_error(monkeypatch):
    """A half-migrated family must fail with an explanation, not a bare numpy error.

    Mid-migration, some seeds sit at 2001 entries and others at intermediate
    multiples of checkpoint_every. np.stack's own message names neither the
    agent nor the lengths, so a transient state reads as a code bug.
    """
    import matplotlib
    matplotlib.use("Agg")
    import pytest
    import utils_plotting as up

    seed_errs = [{"errors": [0.1] * 2001}, {"errors": [0.1] * 1401}]
    with pytest.raises(ValueError, match="partially migrated|different trajectory lengths"):
        up.plot_rmse_comparison_single(
            results_list=[None],
            baseline_error_list=[None],
            agents=["CO$_2$-only"],
            seed_errors_list=[seed_errs],
            seed_baseline_error_list=[[0.05, 0.05]],
        )
