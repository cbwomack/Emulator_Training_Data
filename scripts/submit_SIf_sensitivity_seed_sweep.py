#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Submits scripts/SIf_sensitivity_seed_sweep.py to Engaging as a set of
per-condition, 50-task SLURM arrays (one array per initial-condition sweep,
per architecture, and per feature-set - $SLURM_ARRAY_TASK_ID is the seed).
Reuses agent_pipeline_runner's retry-hardened sbatch()/write_slurm() helpers,
matching submit_stage_e_array.py's simple one-integer-axis-per-array pattern
rather than a combos.txt+sed line-file (each array here only varies over
seed - the sweep/condition axis is fixed per array).

Total new tasks across all arrays: 50 (IC) + 4*50 (architecture) + 3*50
(features) = 400 - comfortably under the account's ~448 effective
MaxSubmitJobsPerUser limit (mit_normal QOS on top of the account-wide
MaxSubmit=500), so every array can be submitted directly without the
multi-batch splitting or self-chaining DAG this repo uses for larger
(1000+ task) campaigns.

Usage:
    python scripts/submit_SIf_sensitivity_seed_sweep.py --sweep ic
    python scripts/submit_SIf_sensitivity_seed_sweep.py --sweep architecture
    python scripts/submit_SIf_sensitivity_seed_sweep.py --sweep features
    python scripts/submit_SIf_sensitivity_seed_sweep.py --sweep all   # all three, sequentially
"""
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent_pipeline_runner import sbatch, write_slurm  # noqa: E402

N_SEEDS = 50
ARCH_CONDITIONS = ['8', '16', '32', '16_16']
FEATURE_CONDITIONS = ['short', 'medium', 'long']
THROTTLE = 50  # concurrent tasks per array


def queue_depth():
    try:
        out = subprocess.run(["squeue", "-u", "cwomack", "-h"], capture_output=True, text=True)
    except FileNotFoundError:
        print("[warn] squeue not found - not running on a SLURM login/compute node, cannot pre-check queue depth")
        return None
    if out.returncode != 0:
        print(f"[warn] squeue check failed ({out.stderr.strip()}) - proceeding without a pre-check")
        return None
    return len(out.stdout.strip().splitlines())


def submit_one(sweep, condition, log_subdir):
    condition_arg = f' --condition {condition}' if condition else ''
    condition_tag = f'_{condition}' if condition else ''
    script = PROJECT_ROOT / f"scripts/_gen_SIf_{sweep}{condition_tag}.slurm"
    out_dir = Path(f"data/SI_results/{log_subdir}/slurm_logs")
    write_slurm(
        script, f"SIf_{sweep}{condition_tag}", out_dir, "01:30:00",
        f'conda run -n project2 python -u scripts/SIf_sensitivity_seed_sweep.py '
        f'--sweep {sweep}{condition_arg} --seed "$SLURM_ARRAY_TASK_ID"'
    )
    return sbatch(script, array=f"0-{N_SEEDS - 1}%{THROTTLE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True, choices=["ic", "architecture", "features", "all"])
    args = parser.parse_args()

    depth = queue_depth()
    if depth is not None:
        print(f"Current queue depth for cwomack: {depth} jobs")

    sweeps = ["ic", "architecture", "features"] if args.sweep == "all" else [args.sweep]

    for sweep in sweeps:
        if sweep == "ic":
            jobs = [submit_one("ic", None, "sensitivity_initial_condition")]
        elif sweep == "architecture":
            jobs = [submit_one("architecture", c, "sensitivity_architecture") for c in ARCH_CONDITIONS]
        elif sweep == "features":
            jobs = [submit_one("features", c, "sensitivity_features") for c in FEATURE_CONDITIONS]
        print(f"{sweep}: submitted job ids {jobs}")


if __name__ == "__main__":
    main()
