#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Submits Figure 4's multi-agent seed-spread cache build
(build_fig4_seed_spread_cache_multi.py --mode run-one) as a 50-task SLURM
array, one task per seed. Mirrors submit_stage_e_array.py exactly.

Usage:
    python scripts/submit_fig4_seed_spread_multi.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent_pipeline_runner import sbatch, write_slurm  # noqa: E402

N_SEEDS = 50


def main():
    script = PROJECT_ROOT / "scripts/_gen_fig4_seed_spread_multi_array.slurm"
    out_dir = Path("data/SI_results/seed_uncertainty/slurm_logs")
    write_slurm(
        script, "fig4_seedspread_multi", out_dir, "00:20:00",
        f'conda run -n project2 python -u scripts/build_fig4_seed_spread_cache_multi.py '
        f'--mode run-one --seed "$SLURM_ARRAY_TASK_ID"'
    )
    sbatch(script, array=f"0-{N_SEEDS - 1}%50")


if __name__ == "__main__":
    main()
