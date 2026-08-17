#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Phase 0, unattended pipeline orchestrator for CH4/N2O/Sulfur/BC.

Chains Stage 0b (cheap-search -> select -> validate -> finalize), Stage C's
baseline K=400 search (search -> finalize), and Stage 0c (50-seed checkpoint
regeneration) into one self-submitting SLURM DAG, so a full agent can be
started once and run to completion unsupervised - no manual polling or
step-by-step submission needed.

Why self-chaining rather than one big dependency graph submitted up front:
the cluster account's MaxSubmit=500 means submitting the whole DAG at once
(cheap-search's 600 tasks alone exceeds it) would be rejected outright - a
dependency-pending job still counts against the limit immediately. Instead,
each stage, once it actually finishes running, submits the next stage via
`sbatch` itself - so at any moment only the currently-running stage (plus a
handful of tiny pending trigger jobs) occupies a slot, never the full
downstream cascade at once. This is the same constraint (and split-batch
technique) already used manually for CO2 and CH4 earlier in Phase 0; this
script just automates the "wait, then submit the next piece" steps a human
was doing by hand.

The optimizer-config chain (cheap-search -> select -> validate -> finalize)
and the baseline chain (K=400 search -> finalize) run independently/in
parallel - the final regeneration stage waits for BOTH via a small
lockfile-guarded join (whichever chain finishes second submits the
regeneration array).

Usage (only the first call is made by a human/agent - everything after
chains itself automatically):
    python agent_pipeline_runner.py --agent N2O --stage start

Internal stages (each submits the next itself - not meant to be invoked
directly, but harmless/idempotent if re-run manually for debugging):
    cheap1, cheap2, select, validate, hp_finalize,
    baseline1, baseline2, baseline_finalize, join, regen
"""
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

GROUPS = ['H-ext', 'tier1', 'tier2', 'DECK', 'CS3', 'all']
TOP_K = 5          # matches 0b_hyperparameter_retune_agent.py's TOP_K default
VALIDATE_SEEDS = 10  # matches its --validate-seeds default (0-9)
REGEN_SEEDS = 50    # matches 0c_regenerate_checkpoints_agent.py's --seed default (0-49)
N_CHEAP_CONFIGS = 100
N_BASELINE_CONFIGS = 100
BASELINE_SEEDS = 5

MODULE_LOAD = "module load miniforge/23.11.0-0"
SBATCH_COMMON = "#SBATCH --partition=mit_normal\n#SBATCH --cpus-per-task=2\n#SBATCH --mem=8G"


def hp_dir(agent):
    return Path(f"data/SI_results/hp_retune/{agent}")


def baseline_dir(agent):
    return Path(f"data/SI_results/baseline_hp/k400_search_{agent}")


def agent_lower(agent):
    return agent if agent in ("Sulfur", "BC") else agent.lower()


def unified_config_path(agent):
    return hp_dir(agent) / "best_config_unified.json"


def baseline_config_path(agent):
    return baseline_dir(agent) / "best_baseline_config_K400.json"


def sbatch(script_path, array=None, depend_jobid=None, extra_args="",
           max_wait_seconds=1800, poll_seconds=30):
    """
    Wraps `sbatch --parsable`, retrying on ANY submission failure rather than
    crashing immediately. Originally this only retried the specific
    QOSMaxSubmitJobPerUserLimit string (several agents' chains overlapping
    can transiently push the account over its submit-count limit even though
    each chain individually respects it), but a second, unrelated transient
    failure ("Batch job submission failed: Connection reset by peer" - a
    momentary slurmctld connectivity blip) hit the un-retried fallback and
    crashed a running chain. A malformed script would fail identically on
    every retry and still surface via the eventual raise below, so retrying
    broadly costs nothing for a real bug while covering transient failures
    generically instead of chasing every specific error string one at a time.
    """
    cmd = ["sbatch", "--parsable"]
    if array:
        cmd.append(f"--array={array}")
    if depend_jobid:
        cmd.append(f"--dependency=afterok:{depend_jobid}")
    if extra_args:
        cmd.extend(extra_args.split())
    cmd.append(str(script_path))

    waited = 0
    while True:
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode == 0:
            jobid = out.stdout.strip().split(";")[0]
            print(f"submitted {script_path} (array={array}, depend={depend_jobid}) -> job {jobid}", flush=True)
            return jobid
        if waited < max_wait_seconds:
            print(f"[retry] {script_path}: sbatch failed ({out.stderr.strip()}), waiting {poll_seconds}s "
                  f"(waited {waited}s so far)...", flush=True)
            time.sleep(poll_seconds)
            waited += poll_seconds
            continue
        print(f"sbatch failed for {script_path}: {out.stderr}", flush=True)
        raise RuntimeError(f"sbatch failed (rc={out.returncode}): {out.stderr}")


def write_slurm(path, job_name, out_dir, time_limit, body):
    out_dir.mkdir(parents=True, exist_ok=True)
    text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
{SBATCH_COMMON}
#SBATCH --time={time_limit}
#SBATCH --output={out_dir}/%A_%a.out
set -euo pipefail
{MODULE_LOAD}
PROJECT_DIR={PROJECT_ROOT}
cd "$PROJECT_DIR"
{body}
"""
    path.write_text(text)
    path.chmod(0o755)


def write_combos(agent):
    hp_dir(agent).mkdir(parents=True, exist_ok=True)
    baseline_dir(agent).mkdir(parents=True, exist_ok=True)

    cheap_path = hp_dir(agent) / "cheap_combos.txt"
    if not cheap_path.exists():
        with open(cheap_path, "w") as f:
            for cfg in range(N_CHEAP_CONFIGS):
                for g in GROUPS:
                    f.write(f"{cfg} {g}\n")

    validate_path = hp_dir(agent) / "validate_combos.txt"
    if not validate_path.exists():
        with open(validate_path, "w") as f:
            for cand in range(TOP_K):
                for g in GROUPS:
                    for s in range(VALIDATE_SEEDS):
                        f.write(f"{g} {cand} {s}\n")

    baseline_combos_path = baseline_dir(agent) / "combos.txt"
    if not baseline_combos_path.exists():
        with open(baseline_combos_path, "w") as f:
            for cfg in range(N_BASELINE_CONFIGS):
                for s in range(BASELINE_SEEDS):
                    f.write(f"{cfg} {s}\n")


def stage_cheap(agent, batch):
    lo, hi = (1, 300) if batch == 1 else (301, 600)
    script = PROJECT_ROOT / f"scripts/_gen_cheap_array_{agent}.slurm"
    write_slurm(
        script, f"hp0b_cheap_{agent}_b{batch}", hp_dir(agent) / "slurm_logs", "00:15:00",
        f'COMBOS=$PROJECT_DIR/data/SI_results/hp_retune/{agent}/cheap_combos.txt\n'
        f'LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMBOS")\n'
        f'CFG=$(echo "$LINE" | awk \'{{print $1}}\')\n'
        f'GROUP=$(echo "$LINE" | awk \'{{print $2}}\')\n'
        f'conda run -n project2 python -u scripts/0b_hyperparameter_retune_agent.py '
        f'--agent {agent} --mode cheap-search-one --config-idx "$CFG" --group "$GROUP"'
    )
    jobid = sbatch(script, array=f"{lo}-{hi}%80")
    if batch == 1:
        submit_control(agent, "cheap2", after=jobid)
    else:
        submit_control(agent, "select", after=jobid)


def stage_select(agent):
    script = PROJECT_ROOT / f"scripts/_gen_select_{agent}.slurm"
    write_slurm(
        script, f"hp0b_select_{agent}", hp_dir(agent) / "slurm_logs", "01:00:00",
        f'conda run -n project2 python -u scripts/0b_hyperparameter_retune_agent.py --agent {agent} --mode select\n'
        f'conda run -n project2 python -u scripts/agent_pipeline_runner.py --agent {agent} --stage validate'
    )
    sbatch(script)


def stage_validate(agent):
    script = PROJECT_ROOT / f"scripts/_gen_validate_array_{agent}.slurm"
    write_slurm(
        script, f"hp0b_validate_{agent}", hp_dir(agent) / "slurm_logs", "00:15:00",
        f'COMBOS=$PROJECT_DIR/data/SI_results/hp_retune/{agent}/validate_combos.txt\n'
        f'LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMBOS")\n'
        f'GROUP=$(echo "$LINE" | awk \'{{print $1}}\')\n'
        f'CAND=$(echo "$LINE" | awk \'{{print $2}}\')\n'
        f'SEED=$(echo "$LINE" | awk \'{{print $3}}\')\n'
        f'conda run -n project2 python -u scripts/0b_hyperparameter_retune_agent.py '
        f'--agent {agent} --mode validate-one --group "$GROUP" --cand-idx "$CAND" '
        f'--validate-seeds "$SEED" --validate-num-updates 1000'
    )
    n = TOP_K * len(GROUPS) * VALIDATE_SEEDS
    jobid = sbatch(script, array=f"1-{n}%80")
    submit_control(agent, "hp_finalize", after=jobid)


def stage_hp_finalize(agent):
    script = PROJECT_ROOT / f"scripts/_gen_hp_finalize_{agent}.slurm"
    seeds = ",".join(str(s) for s in range(VALIDATE_SEEDS))
    write_slurm(
        script, f"hp0b_finalize_{agent}", hp_dir(agent) / "slurm_logs", "01:00:00",
        f'conda run -n project2 python -u scripts/0b_hyperparameter_retune_agent.py --agent {agent} --mode collect\n'
        f'conda run -n project2 python -u scripts/0b_hyperparameter_retune_agent.py --agent {agent} '
        f'--mode finalize --validate-seeds {seeds}\n'
        f'conda run -n project2 python -u scripts/agent_pipeline_runner.py --agent {agent} --stage join'
    )
    sbatch(script)


def stage_baseline(agent, batch):
    lo, hi = (1, 250) if batch == 1 else (251, 500)
    script = PROJECT_ROOT / f"scripts/_gen_baseline_array_{agent}.slurm"
    write_slurm(
        script, f"6e-k400_{agent}_b{batch}", baseline_dir(agent) / "slurm_logs", "00:10:00",
        f'COMBOS=$PROJECT_DIR/data/SI_results/baseline_hp/k400_search_{agent}/combos.txt\n'
        f'LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMBOS")\n'
        f'CFG=$(echo "$LINE" | awk \'{{print $1}}\')\n'
        f'SEED=$(echo "$LINE" | awk \'{{print $2}}\')\n'
        f'conda run -n project2 python -u scripts/6e_baseline_hp_search_k400_agent.py '
        f'--agent {agent} --mode search-one --config-idx "$CFG" --seed "$SEED"'
    )
    jobid = sbatch(script, array=f"{lo}-{hi}%80")
    if batch == 1:
        submit_control(agent, "baseline2", after=jobid)
    else:
        submit_control(agent, "baseline_finalize", after=jobid)


def stage_baseline_finalize(agent):
    script = PROJECT_ROOT / f"scripts/_gen_baseline_finalize_{agent}.slurm"
    write_slurm(
        script, f"6e-k400_finalize_{agent}", baseline_dir(agent) / "slurm_logs", "01:00:00",
        f'conda run -n project2 python -u scripts/6e_baseline_hp_search_k400_agent.py --agent {agent} --mode collect\n'
        f'conda run -n project2 python -u scripts/6e_baseline_hp_search_k400_agent.py --agent {agent} --mode finalize\n'
        f'conda run -n project2 python -u scripts/agent_pipeline_runner.py --agent {agent} --stage join'
    )
    sbatch(script)


def stage_join(agent):
    """Whichever chain (HP or baseline) finishes second submits the
    regeneration array - guarded by an atomic mkdir lock so exactly one
    of the two chains does it, even in the (very unlikely, given how
    differently-timed the two chains are) case both check at once."""
    lock_dir = Path(f"data/SI_results/hp_retune/{agent}/.join_lock")
    if not (unified_config_path(agent).exists() and baseline_config_path(agent).exists()):
        print(f"[{agent}] join: still waiting on the other chain, nothing to do yet")
        return
    try:
        lock_dir.mkdir()
    except FileExistsError:
        print(f"[{agent}] join: regen already submitted by the other chain, skipping")
        return
    stage_regen(agent)


def stage_regen(agent):
    script = PROJECT_ROOT / f"scripts/_gen_regen_array_{agent}.slurm"
    combos_path = Path(f"checkpoints/.{agent}_regen_combos.txt")
    combos_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combos_path, "w") as f:
        for seed in range(REGEN_SEEDS):
            for g in GROUPS:
                f.write(f"{seed} {g}\n")
    write_slurm(
        script, f"0c_regen_{agent}", Path(f"checkpoints/{agent_lower(agent)}_retuned/slurm_logs"), "00:15:00",
        f'COMBOS=$PROJECT_DIR/{combos_path}\n'
        f'LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMBOS")\n'
        f'SEED=$(echo "$LINE" | awk \'{{print $1}}\')\n'
        f'GROUP=$(echo "$LINE" | awk \'{{print $2}}\')\n'
        f'conda run -n project2 python -u scripts/0c_regenerate_checkpoints_agent.py '
        f'--agent {agent} --group "$GROUP" --seed "$SEED" '
        f'--baseline-config-path {baseline_config_path(agent)}'
    )
    n = REGEN_SEEDS * len(GROUPS)  # 50*6=300, fits under MaxSubmit in one submission
    sbatch(script, array=f"1-{n}%80")


def submit_control(agent, stage, after=None):
    """Tiny 1-task job whose only job is to invoke this same script with
    --stage <stage> once its dependency clears - the recursive trigger
    that keeps the DAG from ever needing >1 job pending at a time for the
    'control' steps."""
    script = PROJECT_ROOT / f"scripts/_gen_trigger_{agent}_{stage}.slurm"
    write_slurm(
        script, f"trigger_{agent}_{stage}", Path("data/SI_results/pipeline_logs"), "01:00:00",
        f'conda run -n project2 python -u scripts/agent_pipeline_runner.py --agent {agent} --stage {stage}'
    )
    sbatch(script, depend_jobid=after)


def _run_with_recovery(agent, stage_name, func, *a, **kw):
    """
    Runs one stage's submission logic; if it fails because sbatch()
    exhausted its retry budget (QOSMaxSubmitJobPerUserLimit persisted for
    the full 30 min), reschedules the SAME stage as a fresh, independent
    trigger job (submit_control, no dependency - runs as soon as it's
    picked up) instead of propagating the failure. This makes stalls
    self-healing indefinitely: each retry cycle is bounded (30 min), but
    the chain of cycles is not - a stage keeps getting re-attempted until
    the account's shared submit-limit actually has room, rather than
    silently stalling forever the way a single job's walltime-bounded
    retry loop can (this is exactly what happened to N2O's validate
    submission and Sulfur's cheap-search submission before this fix).
    Dispatching each of "start"'s two independent sub-stages (cheap1,
    baseline1) through this wrapper separately means a failure in one
    doesn't block or duplicate-resubmit the other.
    """
    try:
        func(*a, **kw)
        return
    except RuntimeError as e:
        print(f"[recover] {agent} stage={stage_name} failed ({e}); "
              f"rescheduling as a fresh trigger job", flush=True)

    # The resubmission itself uses the same (now broadly-retrying) sbatch(),
    # but as defense in depth - this exact recovery call already failed once
    # in practice on a transient connectivity error - give it a few more
    # attempts with a short cooldown before giving up entirely.
    for attempt in range(5):
        try:
            submit_control(agent, stage_name, after=None)
            return
        except RuntimeError as e:
            print(f"[recover] resubmission attempt {attempt + 1}/5 for {agent} "
                  f"stage={stage_name} also failed ({e}); retrying in 60s", flush=True)
            time.sleep(60)
    raise RuntimeError(f"{agent} stage={stage_name}: exhausted all resubmission attempts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=["CH4", "N2O", "Sulfur", "BC"])
    parser.add_argument("--stage", required=True,
                         choices=["start", "cheap1", "cheap2", "select", "validate", "hp_finalize",
                                  "baseline1", "baseline2", "baseline_finalize", "join", "regen"])
    args = parser.parse_args()
    agent, stage = args.agent, args.stage

    write_combos(agent)

    if stage == "start":
        _run_with_recovery(agent, "cheap1", stage_cheap, agent, batch=1)
        _run_with_recovery(agent, "baseline1", stage_baseline, agent, batch=1)
    elif stage == "cheap1":
        _run_with_recovery(agent, "cheap1", stage_cheap, agent, batch=1)
    elif stage == "cheap2":
        _run_with_recovery(agent, "cheap2", stage_cheap, agent, batch=2)
    elif stage == "select":
        _run_with_recovery(agent, "select", stage_select, agent)
    elif stage == "validate":
        _run_with_recovery(agent, "validate", stage_validate, agent)
    elif stage == "hp_finalize":
        _run_with_recovery(agent, "hp_finalize", stage_hp_finalize, agent)
    elif stage == "baseline1":
        _run_with_recovery(agent, "baseline1", stage_baseline, agent, batch=1)
    elif stage == "baseline2":
        _run_with_recovery(agent, "baseline2", stage_baseline, agent, batch=2)
    elif stage == "baseline_finalize":
        _run_with_recovery(agent, "baseline_finalize", stage_baseline_finalize, agent)
    elif stage == "join":
        _run_with_recovery(agent, "join", stage_join, agent)
    elif stage == "regen":
        _run_with_recovery(agent, "regen", stage_regen, agent)


if __name__ == "__main__":
    main()
