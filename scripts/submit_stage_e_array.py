#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Submits Stage E's per-seed SI-extended-results cache build
(build_seed_spread_cache_agent.py --mode run-one) as a 50-task SLURM array
for a single agent, one task per seed. Reuses agent_pipeline_runner's
retry-hardened sbatch()/write_slurm() helpers rather than duplicating them.

Usage:
    python scripts/submit_stage_e_array.py --agent CH4
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent_pipeline_runner import sbatch, write_slurm  # noqa: E402

N_SEEDS = 50


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True, choices=["CH4", "N2O", "Sulfur", "BC"])
    args = parser.parse_args()
    agent = args.agent

    script = PROJECT_ROOT / f"scripts/_gen_stage_e_array_{agent}.slurm"
    out_dir = Path("data/SI_results/seed_uncertainty/slurm_logs")
    write_slurm(
        script, f"stageE_{agent}", out_dir, "00:20:00",
        f'conda run -n project2 python -u scripts/build_seed_spread_cache_agent.py '
        f'--mode run-one --agent {agent} --seed "$SLURM_ARRAY_TASK_ID"'
    )
    sbatch(script, array=f"0-{N_SEEDS - 1}%50")


if __name__ == "__main__":
    main()
