#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Submits build_multi_dynrich_grid_cache.py --mode run-one as a SLURM array,
one task per seed, for the Table S2 Multi-row "dynamically rich" full
eval-grid extension. Seeds 0-1 are already done (seed 0 locally, seed 1 via
an interactive srun smoke test), so this submits only seeds 2-49 by default.

Usage:
    python scripts/submit_multi_dynrich_grid_cache.py
    python scripts/submit_multi_dynrich_grid_cache.py --start 0 --end 49
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent_pipeline_runner import sbatch, write_slurm  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=49)
    args = parser.parse_args()

    script = PROJECT_ROOT / "scripts/_gen_multi_dynrich_grid.slurm"
    out_dir = Path("data/SI_results/seed_uncertainty/slurm_logs")
    write_slurm(
        script, "multi_dynrich", out_dir, "00:20:00",
        'conda run -n project2 python -u scripts/build_multi_dynrich_grid_cache.py '
        '--mode run-one --seed "$SLURM_ARRAY_TASK_ID"'
    )
    sbatch(script, array=f"{args.start}-{args.end}%50")


if __name__ == "__main__":
    main()
