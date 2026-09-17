#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Submits Stage 6o's two compute stages to Engaging, each as a single 150-task
SLURM array - one task per (seed, init_cond) pair, 50 seeds x
constant/sine/gaussian, both reading the same combos file.

    --stage opt   scripts/6o_fig7_MESM_opt_seed_sweep.py  (~1 h/task)
    --stage eval  scripts/6o_fig7_MESM_opt_evaluate.py    (~3 min/task)

`eval` requires `opt` to have finished - every (seed, init_cond) checkpoint
must exist - so the two are submitted separately rather than chained, matching
how submit_fig7_seed_spread_MESM.py sits downstream of
submit_MESM_seed_sweep_regen.py.

One task per PAIR rather than per seed (the axis
submit_SIf_sensitivity_seed_sweep.py uses) because at NUM_UPDATES=2000 a single
optimization is roughly an hour: looping all three initial conditions inside one
task would put it near four hours and make a wall-clock timeout cost three runs
instead of one. Two axes means a combos file read by $SLURM_ARRAY_TASK_ID,
matching submit_MESM_seed_sweep_regen.py's pattern rather than
submit_SIf_sensitivity_seed_sweep.py's single-integer one.

150 tasks is comfortably under the account's ~448 effective MaxSubmitJobsPerUser
limit, so this submits directly - no batching or self-chaining DAG.

Timeouts are safe to just resubmit: 6o_fig7_MESM_opt_seed_sweep.py runs with
resume_if_exists=True and checkpoints every 50 updates, and
optimize_emissions_inverse refuses to resume a checkpoint whose recorded
hyperparameters differ from the incoming call, so a re-run either continues the
same optimization or stops loudly. The eval stage skips any (seed, init_cond)
whose partial result already exists.

Usage:
    python scripts/submit_6o_fig7_MESM.py --stage opt
    python scripts/submit_6o_fig7_MESM.py --stage eval
    python scripts/submit_6o_fig7_MESM.py --stage opt --dry-run
"""
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent_pipeline_runner import sbatch, write_slurm  # noqa: E402

SEEDS = range(50)
INIT_CONDS = ["constant", "sine", "gaussian"]
THROTTLE = 50  # concurrent tasks

COMBOS_PATH = Path("checkpoints/.fig7_MESM_opt_combos.txt")

STAGES = {
    "opt": {
        "script": "scripts/6o_fig7_MESM_opt_seed_sweep.py",
        "args": '--seed "$SEED" --init-cond "$IC"',
        "job_name": "fig7_MESM_opt",
        # ~40 min/task measured locally at ~1.1 s/update; generous, and a
        # timeout just resumes from the last checkpoint.
        "time": "04:00:00",
        "log_dir": "checkpoints/co2_MESM_retuned/slurm_logs",
    },
    "eval": {
        "script": "scripts/6o_fig7_MESM_opt_evaluate.py",
        "args": '--mode run-one --seed "$SEED" --init-cond "$IC"',
        "job_name": "fig7_MESM_eval",
        "time": "00:40:00",
        "log_dir": "data/SI_results/seed_uncertainty/slurm_logs_fig7v2",
    },
}


def queue_depth():
    try:
        out = subprocess.run(["squeue", "-u", "cwomack", "-h"], capture_output=True, text=True)
    except FileNotFoundError:
        print("[warn] squeue not found - not on a SLURM node, cannot pre-check queue depth")
        return None
    if out.returncode != 0:
        print(f"[warn] squeue check failed ({out.stderr.strip()}) - proceeding without a pre-check")
        return None
    return len(out.stdout.strip().splitlines())


def combos_path(init_cond=None):
    """Filtered submissions get their own combos file, so a re-submission for one
    initial condition can never rewrite the file an already-queued full array is
    still reading line-by-line via $SLURM_ARRAY_TASK_ID."""
    return COMBOS_PATH if init_cond is None else \
        COMBOS_PATH.with_name(f".fig7_MESM_opt_combos_{init_cond}.txt")


def write_combos(init_cond=None):
    ics = [init_cond] if init_cond else INIT_CONDS
    path = combos_path(init_cond)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for seed in SEEDS:
            for ic in ics:
                f.write(f"{seed} {ic}\n")
    n = len(list(SEEDS)) * len(ics)
    print(f"wrote {path} ({n} tasks)")
    return n, path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    parser.add_argument("--init-cond", choices=INIT_CONDS, default=None,
                        help="Submit only this initial condition (50 tasks instead of 150). "
                             "Used to extend or re-evaluate one IC without touching the others.")
    parser.add_argument("--num-updates", type=int, default=None,
                        help="opt stage only: override the script's own num_updates. Raising it "
                             "on existing checkpoints resumes them to the new total.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write the combos file and SLURM script, print the array spec, do not sbatch")
    args = parser.parse_args()

    if args.num_updates is not None and args.stage != "opt":
        parser.error("--num-updates only applies to --stage opt")

    cfg = STAGES[args.stage]
    n, cpath = write_combos(args.init_cond)

    extra = f' --num-updates {args.num_updates}' if args.num_updates else ''

    depth = queue_depth()
    if depth is not None:
        print(f"Current queue depth for cwomack: {depth} jobs")

    suffix = f"_{args.init_cond}" if args.init_cond else ""
    script = PROJECT_ROOT / f"scripts/_gen_fig7_MESM_{args.stage}{suffix}_array.slurm"
    write_slurm(
        script, cfg["job_name"], Path(cfg["log_dir"]), cfg["time"],
        f'COMBOS=$PROJECT_DIR/{cpath}\n'
        f'LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMBOS")\n'
        f'SEED=$(echo "$LINE" | awk \'{{print $1}}\')\n'
        f'IC=$(echo "$LINE" | awk \'{{print $2}}\')\n'
        f'echo "task ${{SLURM_ARRAY_TASK_ID}}: seed=$SEED init_cond=$IC"\n'
        f'conda run -n project2 python -u {cfg["script"]} {cfg["args"]}{extra}'
    )

    array = f"1-{n}%{THROTTLE}"
    if args.dry_run:
        print(f"[dry-run] would submit {script} with --array={array}")
        print(script.read_text())
        return

    jobid = sbatch(script, array=array)
    print(f"submitted job {jobid} (stage={args.stage}, array={array})")


if __name__ == "__main__":
    main()
