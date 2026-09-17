#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Submits Stage 6i Phase B (scripts/6j_fig6_ood_evaluate.py --mode run-one) as a
50-task SLURM array per config, one task per seed.

'all' (Opt. All vs. baseline) was already run and collected in a prior stage.
Figure 6's 3-column rebuild needs the same replay/scoring pass for the other
three training-scenario checkpoint families the figure compares - damip,
geomip, tier1 - so those are what get submitted by default.

Deliberately self-contained rather than built on agent_pipeline_runner's
write_slurm, which hardcodes two values wrong for this job:
  - MODULE_LOAD is `miniforge/23.11.0-0`; OOD.md section 7 specifies
    `miniforge/25.11.0-0`, which is what the validated seed-0 run used.
  - SBATCH_COMMON requests --mem=8G; each task loads a 32 MB checkpoint and
    runs the FaIR SCM, and the seed-0 run was validated at 32G.
Changing those constants globally would affect every other pipeline that
depends on them, so this job carries its own.

Usage:
    python scripts/submit_fig6_ood_eval.py                    # submit damip, geomip, tier1
    python scripts/submit_fig6_ood_eval.py --config damip      # submit just one config
    python scripts/submit_fig6_ood_eval.py --dry-run           # write the .slurm files only
"""
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

N_SEEDS = 50
# QOS caps submitted jobs at 500 per user (OOD.md section 7); 25 concurrent per
# array leaves room for other chains even when multiple configs are submitted
# at once.
MAX_CONCURRENT = 25

# The four checkpoint families scripts/6j_fig6_ood_evaluate.py knows about.
# 'all' is left out of the default submission list since it was already run.
ALL_CONFIGS = ["all", "damip", "geomip", "tier1"]
DEFAULT_CONFIGS = ["damip", "geomip", "tier1"]

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=fig6_ood_eval_{config}
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output={out_dir}/%A_%a.out
set -euo pipefail
module load miniforge/25.11.0-0
cd {project_dir}
conda run -n project2 python -u scripts/6j_fig6_ood_evaluate.py \\
    --mode run-one --config {config} --seed "$SLURM_ARRAY_TASK_ID"
"""


def submit_one(config, n_seeds, dry_run):
    out_dir = Path("data/SI_results/fig6_ood/slurm_logs") / config
    out_dir.mkdir(parents=True, exist_ok=True)

    script = PROJECT_ROOT / f"scripts/_gen_fig6_ood_eval_array_{config}.slurm"
    script.write_text(SLURM_TEMPLATE.format(out_dir=out_dir, project_dir=PROJECT_ROOT, config=config))
    script.chmod(0o755)
    print(f"wrote {script}")

    if dry_run:
        return

    cmd = ["sbatch", "--parsable", f"--array=0-{n_seeds - 1}%{MAX_CONCURRENT}",
           str(script)]
    print(" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"sbatch failed for config={config}: {res.stderr.strip()}")
    print(f"submitted job {res.stdout.strip()} (config={config})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--config", choices=ALL_CONFIGS, default=None,
                     help="Submit just this one config. Default: damip, geomip, tier1.")
    args = ap.parse_args()

    configs = [args.config] if args.config else DEFAULT_CONFIGS
    for config in configs:
        submit_one(config, args.n_seeds, args.dry_run)


if __name__ == "__main__":
    main()
