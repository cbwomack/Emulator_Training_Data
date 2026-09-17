# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Stage 6i (Figure 6 OOD extension), Phase A: the scenario registry and its two
constructions - the CMIP7 cross and the RCMIP loader.

These pin the properties that a silent regression would otherwise destroy:
  - the CMIP7 cross really takes GHGs from one parent and aerosols from the
    other, and the parents it is given are pairwise distinct in aerosols (the
    CMIP7 set has only FOUR distinct aerosol pathways, not seven - verylow and
    verylow-overshoot differ only in CO2 FFI, so a careless donor choice
    silently produces two identical "rungs");
  - the RCMIP loader reads CO2 FFI, not total CO2 (the repo's 'CO2' agent is
    CO2 FFI - see SCENARIOS.md);
  - the RCMIP loader refuses to apply a unit divisor to an unexpected unit;
  - piControl-branched scenarios are built with NO historical context.

Everything here is emulator-free: no checkpoint, no hyperparameter file.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ood():
    """The Stage 6i script, imported by path (its filename is not an identifier)."""
    path = PROJECT_ROOT / "scripts" / "6i_fig6_ood_extension.py"
    spec = importlib.util.spec_from_file_location("ood_ext", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def eval_sets(ood):
    import utils_inverse
    sets, *_ = utils_inverse.generate_eval_data(
        tuple(ood.AGENTS), DECK=True, CS3=True, DAMIP=True, GeoMIP=True
    )
    return sets


# ---------------------------
# The CMIP7 cross
# ---------------------------

def test_cmip7_cross_takes_ghgs_and_aerosols_from_the_right_parents(ood, eval_sets):
    entry = dict(tag="t", group="cmip7_ramip", source="cmip7_cross", mip="test",
                 ghg_from="H-ext", aer_from="L-ext", prepend_historical=True)
    _years, emis = ood.build_emissions(entry, eval_sets)
    ghg = ood._cmip7_agent_arrays(eval_sets, "H-ext")
    aer = ood._cmip7_agent_arrays(eval_sets, "L-ext")

    for a in ood.GHG_AGENTS:
        np.testing.assert_allclose(emis[a], ghg[a])
    for a in ood.AER_AGENTS:
        np.testing.assert_allclose(emis[a], aer[a])

    # and the two parents really do differ in aerosols, or the cross is a no-op
    assert not np.allclose(ghg["Sulfur"], aer["Sulfur"])


def test_cmip7_aerosol_donors_are_pairwise_distinct(ood, eval_sets):
    """The CMIP7 set has only four distinct aerosol pathways. If a future edit
    picks two donors from the same group, the rungs become duplicates."""
    donors = {d: ood._cmip7_agent_arrays(eval_sets, d) for d in ood.CMIP7_AEROSOL_DONORS}
    names = list(donors)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.allclose(donors[names[i]]["Sulfur"], donors[names[j]]["Sulfur"]), (
                f"{names[i]} and {names[j]} have identical Sulfur - they belong to the "
                f"same CMIP7 aerosol-pathway group and cannot both be rungs"
            )


def test_cmip7_cross_year_end_truncates(ood, eval_sets):
    base = dict(tag="t", group="cmip7_ramip", source="cmip7_cross", mip="test",
                ghg_from="H-ext", aer_from="L-ext", prepend_historical=True)
    years_full, _ = ood.build_emissions(base, eval_sets)
    years_cut, emis_cut = ood.build_emissions({**base, "year_end": 2150}, eval_sets)

    assert years_full[-1] > 2150
    assert years_cut[-1] == 2150
    assert all(len(v) == len(years_cut) for v in emis_cut.values())


def test_cmip7_cross_rejects_length_mismatched_parents(ood, eval_sets):
    # 'M' is a 127-year Tier-1 scenario; 'H-ext' is 477. Splicing must fail loudly.
    entry = dict(tag="t", group="cmip7_ramip", source="cmip7_cross", mip="test",
                 ghg_from="H-ext", aer_from="M", prepend_historical=True)
    with pytest.raises(ValueError, match="lengths differ"):
        ood.build_emissions(entry, eval_sets)


# ---------------------------
# The RCMIP loader
# ---------------------------

def test_rcmip_reads_co2_ffi_not_total_co2(ood):
    """The repo's 'CO2' agent is CO2 FFI. Reading the bare 'Emissions|CO2' total
    overstates it by several percent - the bug this stage fixed."""
    assert ood.RCMIP_VARIABLES["CO2"] == "Emissions|CO2|MAGICC Fossil and Industrial"

    years = np.arange(2024, 2101)
    ffi = ood.interp_rcmip("ssp370", "AIM/CGE", "CO2", years)
    total = np.array([
        v for v in ood.load_rcmip_species("ssp370", "AIM/CGE", "CO2").values()
    ])
    assert np.all(ffi > 0)
    # FFI is strictly a component of the total, so it must be smaller.
    assert ffi.max() < total.max()


def test_rcmip_rejects_an_unexpected_unit(ood, monkeypatch):
    monkeypatch.setitem(ood.RCMIP_EXPECTED_UNIT, "CH4", "Gt CH4/yr")
    with pytest.raises(ValueError, match="Unit mismatch"):
        ood.load_rcmip_species("ssp370", "AIM/CGE", "CH4")


def test_rcmip_missing_row_raises(ood):
    with pytest.raises(ValueError, match="No RCMIP row"):
        ood.load_rcmip_species("not-a-scenario", "AIM/CGE", "CO2")


def test_co2_only_scenarios_have_zero_non_co2_emissions(ood):
    """esm-bell-* is CO2-only in the source data - asserted, not assumed."""
    years = np.arange(1850, 2050)
    for agent in ("CH4", "N2O", "Sulfur", "BC"):
        v = ood.interp_rcmip("esm-bell-1000PgC", "idealised", agent, years)
        assert np.allclose(v, 0.0), f"{agent} is not zero in esm-bell-1000PgC"
    co2 = ood.interp_rcmip("esm-bell-1000PgC", "idealised", "CO2", years)
    assert co2.max() > 50.0
    assert co2[-1] == pytest.approx(0.0, abs=1e-6)


def test_rcmip_cross_splits_ghg_and_aerosol_sources(ood, eval_sets):
    entry = dict(tag="t", group="cmip6_aerosol", source="rcmip_cross", mip="test",
                 ghg_scenario="ssp370", ghg_model="AIM/CGE",
                 aer_scenario="ssp126", aer_model="IMAGE",
                 year_range=(2024, 2100), prepend_historical=True)
    years, emis = ood.build_emissions(entry, eval_sets)
    for a in ood.GHG_AGENTS:
        np.testing.assert_allclose(emis[a], ood.interp_rcmip("ssp370", "AIM/CGE", a, years))
    for a in ood.AER_AGENTS:
        np.testing.assert_allclose(emis[a], ood.interp_rcmip("ssp126", "IMAGE", a, years))


# ---------------------------
# Feature construction
# ---------------------------

def test_no_historical_scenario_starts_from_a_clean_slate(ood, eval_sets):
    """piControl-branched experiments must not inherit historical context: the
    cumulative-emissions feature has to start at zero."""
    entry = next(e for e in ood.SCENARIOS if e["tag"] == "esm-bell-1000PgC")
    d = ood.build_features(entry, eval_sets)
    cum_col = ood.AGENTS.index("CO2") * 5 + 4  # per-agent block, Cum_prev is 5th
    assert d["X"][0, cum_col] == pytest.approx(0.0, abs=1e-6)
    assert d["y_scm"][0] == pytest.approx(0.0, abs=1e-3)


def test_historical_prepended_scenario_does_not_start_from_zero(ood, eval_sets):
    """The contrapositive: a scenario that prepends historical must carry the
    accumulated context, or build_dataset_from_runfair_dict's allowlist trap has
    silently dropped it."""
    entry = next(e for e in ood.SCENARIOS if e["tag"] == "ssp370-lowNTCF")
    d = ood.build_features(entry, eval_sets)
    cum_col = ood.AGENTS.index("CO2") * 5 + 4
    assert d["X"][0, cum_col] > 100.0     # ~2.5 TtCO2 of history by 2023
    assert d["y_scm"][0] > 0.2            # already ~1.2 K above preindustrial


def test_every_registry_entry_builds_and_is_finite(ood, eval_sets):
    for entry in ood.SCENARIOS:
        d = ood.build_features(entry, eval_sets)
        assert np.all(np.isfinite(d["X"])), f"{entry['tag']}: non-finite features"
        assert np.all(np.isfinite(d["y_scm"])), f"{entry['tag']}: non-finite truth"
        assert len(d["years"]) == len(d["y_scm"]) == d["X"].shape[0]


def test_registry_tags_are_unique(ood):
    tags = [e["tag"] for e in ood.SCENARIOS]
    assert len(tags) == len(set(tags))


# ---------------------------
# Harmonization (Gidden et al. 2018)
# ---------------------------

def test_convergence_factor_endpoints_and_linearity(ood):
    years = np.arange(2023, 2101)
    f = ood._convergence_factors(years, 2023, 2080)
    assert f[0] == pytest.approx(1.0)
    assert f[years == 2080][0] == pytest.approx(0.0)
    assert np.all(f[years > 2080] == 0.0)
    # linear in between, and never outside [0, 1]
    mid = f[years == 2051][0]
    assert mid == pytest.approx(1.0 - 28.0 / 57.0)
    assert np.all((f >= 0.0) & (f <= 1.0))


def test_ratio_and_offset_pin_the_harmonization_year_exactly(ood):
    """The defining property: at t_harm the corrected pathway equals history."""
    years = np.arange(2023, 2101)
    model = np.linspace(40.0, 10.0, len(years))
    hist = 32.0
    r = ood.harmonize_ratio(model, years, hist / model[0])
    o = ood.harmonize_offset(model, years, hist - model[0])
    assert float(r[0]) == pytest.approx(hist, rel=1e-5)
    assert float(o[0]) == pytest.approx(hist, rel=1e-5)


def test_correction_vanishes_at_the_convergence_year(ood):
    """A 'reduce' method must not overwrite the scenario's long-term signal."""
    years = np.arange(2023, 2101)
    model = np.linspace(40.0, 10.0, len(years))
    after = years >= 2080
    for corrected in (ood.harmonize_ratio(model, years, 0.7),
                      ood.harmonize_offset(model, years, -12.0)):
        np.testing.assert_allclose(corrected[after], model[after], rtol=1e-5, atol=1e-6)


def test_method_choice_uses_offset_only_where_the_pathway_crosses_zero(ood):
    years = np.arange(2023, 2101)
    positive = np.linspace(40.0, 5.0, len(years))
    assert ood.choose_harmonization_method(35.0, 40.0, positive, years=years) == "reduce_ratio"

    # goes net-negative before convergence: a ratio would scale it further negative
    crossing = np.linspace(40.0, -20.0, len(years))
    assert ood.choose_harmonization_method(35.0, 40.0, crossing, years=years) == "reduce_offset"

    # negative only AFTER convergence, where the correction is already zero
    late = np.concatenate([np.linspace(40.0, 1.0, 57), np.linspace(1.0, -20.0, len(years) - 57)])
    assert ood.choose_harmonization_method(35.0, 40.0, late, years=years) == "reduce_ratio"


def test_method_choice_refuses_unverified_branches(ood):
    """aneris' tree has branches this roster never reaches; guessing there is worse
    than failing loudly."""
    years = np.arange(2023, 2101)
    model = np.linspace(40.0, 5.0, len(years))
    with pytest.raises(ValueError, match="history is zero"):
        ood.choose_harmonization_method(0.0, 40.0, model, years=years)
    with pytest.raises(ValueError, match="identically zero"):
        ood.choose_harmonization_method(35.0, 0.0, np.zeros(len(years)), years=years)
    with pytest.raises(ValueError, match="~zero at the harmonization year"):
        ood.choose_harmonization_method(35.0, 1e-12, model, years=years)


def test_ssp534_over_co2_is_the_one_offset_case(ood, eval_sets):
    """The decision tree's split, read off the real roster rather than assumed."""
    methods = {}
    for entry in ood.SCENARIOS:
        d = ood.build_features(entry, eval_sets)
        for agent, m in d["harmonization"].items():
            methods[(entry["tag"], agent)] = m["method"]

    offsets = {k for k, v in methods.items() if v == "reduce_offset"}
    assert offsets == {("ssp534-over", "CO2")}, offsets
    assert len(methods) == 25


def test_harmonization_closes_the_inventory_step(ood, eval_sets):
    """The point of the exercise. Note what is and is not claimed.

    The method pins the pathway to the historical inventory AT THE HARMONIZATION
    YEAR (2023); it says nothing directly about the year-over-year step into 2024.
    Those usually move together, but not always: rcp45's raw CO2 happens to cross
    the historical value between 2023 and 2024, so its raw 2024 step is 0.2% by
    coincidence and harmonization takes it to 1.0%. That is not a regression - the
    2023 inventory mismatch it removes is real and the 2024 step stays negligible.
    So: every species must land within 5% at the handoff, and every species whose
    raw step was actually large must shrink.
    """
    _yh, eh = ood.utils_inverse.extract_years_and_emis(
        eval_sets["Tier 1"]["historical"], agents=ood.AGENTS
    )
    hist = {a: np.asarray(eh[a]) for a in ood.AGENTS}

    n_large = 0
    for entry in ood.SCENARIOS:
        if not entry.get("harmonize"):
            continue
        d = ood.build_features(entry, eval_sets)
        for a in ood.AGENTS:
            h = float(hist[a][-1])
            meta = d["harmonization"][a]

            # the defining property, on the real data
            corrected_at_harm = (meta["model_value"] * meta["ratio"]
                                 if meta["method"] == "reduce_ratio"
                                 else meta["model_value"] + meta["offset"])
            assert corrected_at_harm == pytest.approx(h, rel=1e-6)

            raw_step = abs(float(d["emis_raw"][a][0]) - h) / abs(h)
            new_step = abs(float(d["emis"][a][0]) - h) / abs(h)
            assert new_step < 0.05, f"{entry['tag']}/{a}: step still {new_step:.1%}"
            if raw_step > 0.05:
                n_large += 1
                assert new_step < raw_step, (
                    f"{entry['tag']}/{a}: raw step {raw_step:.1%} -> {new_step:.1%}"
                )
    assert n_large >= 15, f"only {n_large} species had a large raw step - roster changed?"


def test_unharmonized_groups_are_left_alone(ood, eval_sets):
    """CMIP7 crosses share the historical's own source file, and piControl-branched
    experiments have no historical to reconcile with. Neither may be touched."""
    for entry in ood.SCENARIOS:
        if entry.get("harmonize"):
            continue
        d = ood.build_features(entry, eval_sets)
        assert d["harmonization"] == {}
        assert d["harmonized"] is False
        for a in ood.AGENTS:
            np.testing.assert_allclose(d["emis"][a], d["emis_raw"][a])
