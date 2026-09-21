#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5 and Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Emulator evaluation (Phase B) for Figure 6 OOD extensions.

This script evaluates a multi-agent optimized emulator and its baseline on
all of them across 50 seeds, and exports the R^2 table.

`--config` selects which of the four training-scenario checkpoint families
Figure 6 compares (Opt. All / Opt. DAMIP / Opt. GeoMIP / Opt. Tier 1), so the
same downselected panel can be scored for each. 'all' writes the unsuffixed
cache and table paths; the others write `_{config}`-suffixed ones. `--mode
merge` combines all four into the long-format table Figure 6's third column
reads.


All 50 seeds are run and reported as median and IQR via
utils_plotting._aggregate_seeds. A single seed is not representative: the seed
spread straddles the baseline median.

Usage:
    python 07i_fig6_ood_evaluate.py --mode run-one --config all --seed 0
    python 07i_fig6_ood_evaluate.py --mode collect --config all
    python 07i_fig6_ood_evaluate.py --mode merge

"""
import os
import sys
from pathlib import Path
from paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

import csv
import json
import pickle
import argparse
import numpy as np
import jax
import jax.numpy as jnp

import utils_inverse

AGENTS = ["CO2", "CH4", "N2O", "Sulfur", "BC"]
ACTIVE_AGENTS = ("CO2", "CH4", "N2O", "Sulfur", "BC")
MODE = "FaIR"
EMA_WINDOWS = (5.0, 30.0, 100.0)

UNIFIED_CONFIG_PATH = "data/SI_results/hp_retune/multi/best_config_unified.json"
BASELINE_CONFIG_PATH = "data/SI_results/baseline_hp/k400_search_multi/best_baseline_config_K400.json"

OUT_DIR = Path("data/SI_results/fig6_ood")
SCENARIO_CACHE = OUT_DIR / "fig6_ood_scenarios.pkl"

RECON_TOL = 0.005

IN_OBJECTIVE_REFERENCE = ["M_GHG", "M_AER", "G6sulfur"]

CONFIGS = {
    "all": dict(
        ckpt_dir="checkpoints/multi_retuned_smooth/seed_sweep",
        ckpt_template="inverse_constant_all_all_agents_seed{seed}.pkl",
        label="Opt. All", train_group="all",
    ),
    "damip": dict(
        ckpt_dir="checkpoints/multi_retuned_smooth/seed_sweep",
        ckpt_template="inverse_constant_DAMIP_all_agents_seed{seed}.pkl",
        label="Opt. DAMIP", train_group="DAMIP",
    ),
    "geomip": dict(
        ckpt_dir="checkpoints/multi_retuned_smooth/seed_sweep",
        ckpt_template="inverse_constant_GeoMIP_all_agents_seed{seed}.pkl",
        label="Opt. GeoMIP", train_group="GeoMIP",
    ),
    "tier1": dict(
        ckpt_dir="checkpoints/multi_fig4_smooth/seed_sweep",
        ckpt_template="inverse_constant_tier1_multi_fig4_seed{seed}.pkl",
        label="Opt. Tier 1", train_group="tier1",
    ),
}


def _paths_for_config(config):
    """Config-specific output paths. 'all' keeps the original unsuffixed
    paths (so the existing cache/table are untouched); the other three
    configs get a `_{config}` tag so their runs never collide with 'all's."""
    suffix = "" if config == "all" else f"_{config}"
    return dict(
        partial_dir=OUT_DIR / f"partial_seed_sweep{suffix}",
        sweep_path=OUT_DIR / f"fig6_ood_seed_sweep{suffix}.pkl",
        r2_table_path=OUT_DIR / f"fig6_ood_r2_table{suffix}.csv",
    )


MERGE_SCENARIOS = [
    "H-ext-Maer", "rcp45", "H-ext-VLaer", "H-ext-Laer", "ssp370-lowNTCF",
    "ssp370-126aer", "rcp85", "ssp534-over", "esm-bell-1000PgC", "esm-bell-2000PgC",
] + IN_OBJECTIVE_REFERENCE
MERGE_TABLE_PATH = OUT_DIR / "fig6_ood_r2_table_fig6.csv"


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def _r2(yhat, ytrue):
    """Coefficient of determination against the SCM truth."""
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)
    ytrue = np.asarray(ytrue, dtype=np.float64).reshape(-1)
    ss_res = float(np.sum((ytrue - yhat) ** 2))
    ss_tot = float(np.sum((ytrue - np.mean(ytrue)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _nrmse(yhat, ytrue):
    return float(utils_inverse._nrmse(jnp.asarray(yhat), jnp.asarray(ytrue)))


# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
def hyperparameters_from_meta(ckpt, path):
    """The replay's hyperparameters, read from the checkpoint's own `meta` and
    cross-checked against the config files (the OOD checklist).
    """
    meta = ckpt.get("meta") or {}
    missing = [k for k in ("K_inner", "lr_inner", "wd_inner", "batch_size") if k not in meta]
    if missing:
        raise RuntimeError(
            f"{path}: meta is missing {missing}. Every checkpoint should carry "
            f"its own configuration; refusing to fall back to a config file "
            f"that may describe a different run."
        )
    hp = {k: meta[k] for k in ("K_inner", "lr_inner", "wd_inner", "batch_size")}

    expected = {
        "mode": MODE,
        "ema_windows_years": EMA_WINDOWS,
        "active_agents": list(ACTIVE_AGENTS),
        "init_cond": "constant",
    }
    for k, want in expected.items():
        got = meta.get(k)
        if got is None:
            continue
        mismatch = (tuple(got) != tuple(want) if isinstance(want, (list, tuple))
                    else got != want)
        if mismatch:
            raise RuntimeError(
                f"{path}: meta.{k} is {got!r}, expected {want!r}. This checkpoint "
                f"was not produced under the configuration this script replays."
            )

    cfg = json.load(open(UNIFIED_CONFIG_PATH))["config"]
    disagree = {k: (hp[k], cfg[k]) for k in hp if k in cfg and hp[k] != cfg[k]}
    if disagree:
        raise RuntimeError(
            f"{path}: checkpoint meta disagrees with {UNIFIED_CONFIG_PATH} on "
            f"{disagree} (checkpoint, config). A batch_size mismatch of exactly "
            f"this kind previously made the optimized emulator look 7-27x worse "
            f"than baseline. Resolve before trusting any number."
        )

    for k in ("smoothness_weight", "penalty_form", "init_cond", "mode"):
        if k in meta:
            print(f"  meta.{k} = {meta[k]}")
    return hp


# ------------------------------------------------------------------
# Replay
# ------------------------------------------------------------------
def reconstruct_optimized_paramsK(setup, ckpt, hp, seed, u_index=-1):
    """Deterministically replay the inner-loop training that produced paramsK
    from U_traj[u_index]. No re-optimization, no outer-loop compute.
    """
    U = ckpt["U_traj"][u_index]
    U_eff = utils_inverse._apply_active_mask_to_emis(U, ACTIVE_AGENTS, inactive_mode="zeros")

    train_updated = utils_inverse.build_train(
        U_eff, agents=AGENTS, ema_windows_years=EMA_WINDOWS,
        years_hist=None, emis_hist_dict=None, mode=MODE,
    )
    train_s, _test_s, stats = utils_inverse.split_and_scale(train_updated, train_updated)

    Xtr = jnp.concatenate([X for (X, _, _) in train_s], axis=0).astype(jnp.float32)
    ytr = jnp.concatenate([y for (_, y, _) in train_s], axis=0).astype(jnp.float32)

    paramsK, _losses = utils_inverse.train_mlp_sgd(
        setup["params0"], Xtr, ytr,
        K=hp["K_inner"], lr=hp["lr_inner"], weight_decay=hp["wd_inner"],
        batch_size=hp["batch_size"], key=jax.random.PRNGKey(seed),
    )
    return paramsK, stats


def evaluate_in_objective(setup, paramsK, stats, group="all"):
    """Per-scenario metrics on one of the checkpoint's own eval_sets groups.
    """
    group_emis_dicts = utils_inverse.build_group_emis_dicts(
        setup["emis_dict_train_JAX"], setup["eval_sets"]
    )
    test_dataset = utils_inverse.build_valid(
        group_emis_dicts[group], historical_name="historical", agents=AGENTS, mode=MODE,
    )
    test_scaled = [(utils_inverse.apply_scaler(X, stats), y, scen)
                   for (X, y, scen) in test_dataset]
    mean_nrmse = float(utils_inverse.avg_nrmse_over_tests(paramsK, test_scaled))

    per_scenario = {}
    for (Xs, y, scen) in test_scaled:
        yhat = utils_inverse.mlp_forward(paramsK, Xs)
        per_scenario[scen] = dict(nrmse=_nrmse(yhat, y), r2=_r2(yhat, y),
                                  n_years=int(np.asarray(y).reshape(-1).shape[0]))
    return mean_nrmse, per_scenario


def evaluate_scenarios(cache, paramsK, stats):
    """The Phase A roster. Features and SCM truth are already built and cached -
    this only scales, forward-passes and scores."""
    out = {}
    for tag, d in cache["scenarios"].items():
        Xs = utils_inverse.apply_scaler(jnp.asarray(d["X"]), stats)
        yhat = np.asarray(utils_inverse.mlp_forward(paramsK, Xs)).reshape(-1)
        ytrue = np.asarray(d["y_scm"]).reshape(-1)
        out[tag] = dict(nrmse=_nrmse(yhat, ytrue), r2=_r2(yhat, ytrue),
                        yhat=yhat, n_years=int(ytrue.shape[0]))
    return out


def baseline_for_seed(setup, cache, seed):
    """The baseline is genuinely RETRAINED at this seed - the fairness property
    the repo maintains everywhere (run_inverse_experiment_setup seeds params0
    and the baseline together, so both vary together across seeds).
    """
    cfg = json.load(open(BASELINE_CONFIG_PATH))["config"]
    train_s, _test_s, stats = utils_inverse.prepare_baseline_data(
        emis_dict_train=setup["eval_sets"]["Tier 1"],
        emis_dict_test=setup["eval_sets"]["Tier 1"],
        historical_name="historical", mode=MODE,
    )
    paramsK_base, _losses, _meta = utils_inverse.train_baseline_emulator(
        train_scaled=train_s, key=jax.random.PRNGKey(seed),
        K=cfg["K"], lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    scenarios = evaluate_scenarios(cache, paramsK_base, stats)
    _mean, in_obj = evaluate_in_objective(setup, paramsK_base, stats)
    return scenarios, in_obj


# ------------------------------------------------------------------
def run_one(seed, config="all", scenario_cache=SCENARIO_CACHE):
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {list(CONFIGS)}, got {config!r}")
    cfg = CONFIGS[config]
    partial_dir = _paths_for_config(config)["partial_dir"]

    partial_dir.mkdir(parents=True, exist_ok=True)
    out_path = partial_dir / f"seed{seed}.pkl"
    if out_path.exists():
        print(f"[seed={seed}] already done, skipping")
        return

    with open(scenario_cache, "rb") as f:
        cache = pickle.load(f)

    ckpt_path = f"{cfg['ckpt_dir']}/{cfg['ckpt_template'].format(seed=seed)}"
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"{ckpt_path} missing")

    print(f"[seed={seed}] loading {ckpt_path}")
    ckpt = utils_inverse.load_inverse_ckpt(ckpt_path)

    step_count = ckpt.get("step_count")
    expected_steps = (ckpt.get("meta") or {}).get("num_updates")
    print(f"  step_count = {step_count} (num_updates = {expected_steps})")
    if expected_steps is not None and step_count != expected_steps:
        raise RuntimeError(
            f"{ckpt_path}: step_count={step_count} but num_updates={expected_steps} "
            f"- this checkpoint is a mid-run snapshot, not a finished optimization."
        )

    hp = hyperparameters_from_meta(ckpt, ckpt_path)
    print(f"  hp = {hp}")

    setup = utils_inverse.run_inverse_experiment_setup(
        AGENTS, ACTIVE_AGENTS, mode=MODE, CS3=True, DAMIP=True, GeoMIP=True,
        idx_demo=None,
        seed=seed,
    )

    # -- self-check: validate the replay METHOD at u_index=-2
    paramsK_chk, stats_chk = reconstruct_optimized_paramsK(setup, ckpt, hp, seed, u_index=-2)
    recon, _per = evaluate_in_objective(setup, paramsK_chk, stats_chk, group=cfg["train_group"])
    stored = float(utils_inverse.recover_nrmse_trajectory(ckpt)[-1])
    raw_errors = float(ckpt["errors"][-1])
    print(f"  [validate] replay NRMSE={recon:.6f} vs stored={stored:.6f} "
          f"(penalty removed; raw errors[-1]={raw_errors:.6f}) diff={abs(recon - stored):.6f}")
    if abs(recon - stored) > RECON_TOL:
        raise RuntimeError(
            f"seed {seed}: replay from U_traj[-2] gives NRMSE {recon:.6f} but the "
            f"checkpoint's own stored NRMSE is {stored:.6f} (penalty removed) - "
            f"difference {abs(recon - stored):.6f} exceeds {RECON_TOL}. Stopping "
            f"rather than reporting a number built on a broken replay."
        )

    paramsK, stats = reconstruct_optimized_paramsK(setup, ckpt, hp, seed, u_index=-1)
    opt_scen = evaluate_scenarios(cache, paramsK, stats)
    _mean, opt_in_obj = evaluate_in_objective(setup, paramsK, stats)

    base_scen, base_in_obj = baseline_for_seed(setup, cache, seed)

    for tag in opt_scen:
        print(f"    {tag:20s} opt R2={opt_scen[tag]['r2']:+.4f} NRMSE={opt_scen[tag]['nrmse']:.4f}"
              f"   base R2={base_scen[tag]['r2']:+.4f} NRMSE={base_scen[tag]['nrmse']:.4f}")

    with open(out_path, "wb") as f:
        pickle.dump(dict(
            seed=seed, checkpoint=ckpt_path, hp=hp, step_count=step_count,
            validation=dict(reconstructed=recon, stored=stored, raw_errors=raw_errors),
            optimized=opt_scen, baseline=base_scen,
            optimized_in_objective=opt_in_obj, baseline_in_objective=base_in_obj,
        ), f)
    print(f"[seed={seed}] done -> {out_path}")


# ------------------------------------------------------------------
def collect(n_seeds=50, config="all", scenario_cache=SCENARIO_CACHE):
    import utils_plotting

    if config not in CONFIGS:
        raise ValueError(f"config must be one of {list(CONFIGS)}, got {config!r}")
    paths = _paths_for_config(config)
    partial_dir, sweep_path, r2_table_path = (
        paths["partial_dir"], paths["sweep_path"], paths["r2_table_path"])

    results, missing = {}, []
    for seed in range(n_seeds):
        p = partial_dir / f"seed{seed}.pkl"
        if not p.exists():
            missing.append(seed)
            continue
        with open(p, "rb") as f:
            results[seed] = pickle.load(f)
    if missing:
        print(f"missing seeds: {missing} ({len(missing)}/{n_seeds}) - not writing final cache")
        return
    print(f"collected {len(results)} seeds")

    with open(scenario_cache, "rb") as f:
        cache = pickle.load(f)

    with open(sweep_path, "wb") as f:
        pickle.dump(results, f)
    print(f"wrote {sweep_path}")

    seeds = sorted(results)
    rows = []

    def _agg(values):
        c, lo, hi = utils_plotting._aggregate_seeds(np.asarray(values)[:, None], "median")
        return float(c[0]), float(lo[0]), float(hi[0])

    # -- the Phase A roster: out-of-objective
    for tag, d in cache["scenarios"].items():
        for emulator, key in (("optimized", "optimized"), ("baseline", "baseline")):
            r2 = [results[s][key][tag]["r2"] for s in seeds]
            nr = [results[s][key][tag]["nrmse"] for s in seeds]
            r2_med, r2_lo, r2_hi = _agg(r2)
            nr_med, nr_lo, nr_hi = _agg(nr)
            rows.append(dict(
                scenario=tag, group=d["group"], mip=d["mip"], source=d["source"],
                harmonized=d.get("harmonized", False),
                n_years=d["X"].shape[0], in_objective=False,
                parents_in_objective=(d["source"] == "cmip7_cross"),
                emulator=emulator,
                r2_median=round(r2_med, 6), r2_p25=round(r2_lo, 6), r2_p75=round(r2_hi, 6),
                nrmse_median=round(nr_med, 6), nrmse_p25=round(nr_lo, 6),
                nrmse_p75=round(nr_hi, 6), n_seeds=len(seeds),
            ))

    # -- Figure 6's existing panels: in-objective, for contrast
    for tag in IN_OBJECTIVE_REFERENCE:
        if tag not in results[seeds[0]]["optimized_in_objective"]:
            print(f"  note: '{tag}' not in the eval group - skipping reference row")
            continue
        for emulator, key in (("optimized", "optimized_in_objective"),
                              ("baseline", "baseline_in_objective")):
            r2 = [results[s][key][tag]["r2"] for s in seeds]
            nr = [results[s][key][tag]["nrmse"] for s in seeds]
            r2_med, r2_lo, r2_hi = _agg(r2)
            nr_med, nr_lo, nr_hi = _agg(nr)
            rows.append(dict(
                scenario=tag, group="in_objective_reference",
                mip="DAMIP / GeoMIP", source="eval_sets['All']", harmonized=False,
                n_years=results[seeds[0]][key][tag]["n_years"], in_objective=True,
                parents_in_objective=True, emulator=emulator,
                r2_median=round(r2_med, 6), r2_p25=round(r2_lo, 6), r2_p75=round(r2_hi, 6),
                nrmse_median=round(nr_med, 6), nrmse_p25=round(nr_lo, 6),
                nrmse_p75=round(nr_hi, 6), n_seeds=len(seeds),
            ))

    with open(r2_table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {r2_table_path}  ({len(rows)} rows)")

    print(f"\n{'scenario':20s} {'emul':10s} {'R2 median [IQR]':30s} {'NRMSE median [IQR]'}")
    for r in rows:
        print(f"{r['scenario']:20s} {r['emulator']:10s} "
              f"{r['r2_median']:+.4f} [{r['r2_p25']:+.4f}, {r['r2_p75']:+.4f}]   "
              f"{r['nrmse_median']:.4f} [{r['nrmse_p25']:.4f}, {r['nrmse_p75']:.4f}]")


# ------------------------------------------------------------------
def merge():
    """Combine the four configs' collected R^2 tables into the single
    long-format table Figure 6's column 3 (the ledger-style forest plot)
    reads: one row per (scenario, emulator), emulator in {Baseline Em.,
    Opt. Tier 1, Opt. DAMIP, Opt. GeoMIP, Opt. All}, restricted to
    MERGE_SCENARIOS.
    """
    per_config = {}
    for config in CONFIGS:
        r2_table_path = _paths_for_config(config)["r2_table_path"]
        if not r2_table_path.exists():
            raise FileNotFoundError(
                f"{r2_table_path} missing - run --mode collect --config {config} first"
            )
        with open(r2_table_path, newline="") as f:
            per_config[config] = {
                (row["scenario"], row["emulator"]): row for row in csv.DictReader(f)
            }

    baseline_tol = 0.02  # R^2 units; run-to-run float noise across configs' independent baseline retrains
    rows = []
    for scenario in MERGE_SCENARIOS:
        base_r2s = {}
        for config in CONFIGS:
            row = per_config[config].get((scenario, "baseline"))
            if row is None:
                raise KeyError(f"{scenario!r}/baseline missing from config {config!r}'s table")
            base_r2s[config] = float(row["r2_median"])
        spread = max(base_r2s.values()) - min(base_r2s.values())
        if spread > baseline_tol:
            raise RuntimeError(
                f"{scenario}: baseline median R^2 disagrees across configs by "
                f"{spread:.4f} (> {baseline_tol}) - {base_r2s}. The baseline is "
                f"supposed to be config-invariant; investigate before trusting "
                f"the merged table."
            )
        canonical_base = per_config["all"][(scenario, "baseline")]
        rows.append(dict(
            scenario=scenario, emulator="Baseline Em.",
            r2_median=canonical_base["r2_median"], r2_p25=canonical_base["r2_p25"],
            r2_p75=canonical_base["r2_p75"], n_seeds=canonical_base["n_seeds"],
        ))
        for config in CONFIGS:
            row = per_config[config][(scenario, "optimized")]
            rows.append(dict(
                scenario=scenario, emulator=CONFIGS[config]["label"],
                r2_median=row["r2_median"], r2_p25=row["r2_p25"],
                r2_p75=row["r2_p75"], n_seeds=row["n_seeds"],
            ))

    with open(MERGE_TABLE_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "emulator", "r2_median", "r2_p25", "r2_p75", "n_seeds"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {MERGE_TABLE_PATH}  ({len(rows)} rows)")
    for r in rows:
        print(f"{r['scenario']:20s} {r['emulator']:14s} "
              f"{float(r['r2_median']):+.4f} [{float(r['r2_p25']):+.4f}, {float(r['r2_p75']):+.4f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["run-one", "collect", "merge"], required=True)
    ap.add_argument("--config", choices=list(CONFIGS), default="all",
                     help="run-one/collect only: which training-scenario checkpoint family to evaluate")
    ap.add_argument("--seed", type=int, default=None, help="run-one only")
    ap.add_argument("--n-seeds", type=int, default=50, help="collect only")
    ap.add_argument("--scenario-cache", default=str(SCENARIO_CACHE))
    args = ap.parse_args()

    if args.mode == "merge":
        merge()
        return

    if args.mode == "run-one":
        if args.seed is None:
            raise SystemExit("run-one requires --seed")
        run_one(args.seed, args.config, Path(args.scenario_cache))
    else:
        collect(args.n_seeds, args.config, Path(args.scenario_cache))


if __name__ == "__main__":
    main()
