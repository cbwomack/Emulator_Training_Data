#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Submits the multi-agent checkpoint regeneration (0c_regenerate_checkpoints_
multi.py) as a SLURM array, one task per (seed, group) - 50 seeds x 5 groups
= 250 tasks, under the ~448-job submit cap so no batching needed. Each task
runs the full production length (NUM_UPDATES=1000), so allow more walltime
per task than the cheap-search/validate arrays.

Usage:
    python scripts/submit_multi_regen.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent_pipeline_runner import sbatch, write_slurm  # noqa: E402

GROUPS = ["H-ext", "tier1", "DAMIP", "GeoMIP", "all"]
SEEDS = range(50)


def main():
    combos_path = Path("checkpoints/.multi_regen_combos.txt")
    combos_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combos_path, "w") as f:
        for seed in SEEDS:
            for group in GROUPS:
                f.write(f"{seed} {group}\n")

    script = PROJECT_ROOT / "scripts/_gen_multi_regen_array.slurm"
    out_dir = Path("checkpoints/multi_retuned/slurm_logs")
    write_slurm(
        script, "0c_regen_multi", out_dir, "00:20:00",
        f'COMBOS=$PROJECT_DIR/{combos_path}\n'
        f'LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMBOS")\n'
        f'SEED=$(echo "$LINE" | awk \'{{print $1}}\')\n'
        f'GROUP=$(echo "$LINE" | awk \'{{print $2}}\')\n'
        f'conda run -n project2 python -u scripts/0c_regenerate_checkpoints_multi.py '
        f'--group "$GROUP" --seed "$SEED"'
    )
    n = len(list(SEEDS)) * len(GROUPS)
    sbatch(script, array=f"1-{n}%50")


if __name__ == "__main__":
    main()
