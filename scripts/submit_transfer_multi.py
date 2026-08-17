#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Submits the multi-agent baseline transfer check (6e_baseline_transfer_check_
multi.py --mode run-one) as a small SLURM array - 2 configs x 5 seeds = 10
tasks, well under any submit-limit concern, so no batching needed.

Usage:
    python scripts/submit_transfer_multi.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent_pipeline_runner import sbatch, write_slurm  # noqa: E402

CONFIGS = ["original", "co2_tuned"]
SEEDS = range(5)


def main():
    combos_path = Path("checkpoints/.transfer_multi_combos.txt")
    combos_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combos_path, "w") as f:
        for cfg in CONFIGS:
            for seed in SEEDS:
                f.write(f"{cfg} {seed}\n")

    script = PROJECT_ROOT / "scripts/_gen_transfer_multi_array.slurm"
    out_dir = Path("data/SI_results/baseline_hp/transfer_check_multi/slurm_logs")
    write_slurm(
        script, "transfer_multi", out_dir, "00:15:00",
        f'COMBOS=$PROJECT_DIR/{combos_path}\n'
        f'LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMBOS")\n'
        f'CFG=$(echo "$LINE" | awk \'{{print $1}}\')\n'
        f'SEED=$(echo "$LINE" | awk \'{{print $2}}\')\n'
        f'conda run -n project2 python -u scripts/6e_baseline_transfer_check_multi.py '
        f'--mode run-one --config "$CFG" --seed "$SEED"'
    )
    n = len(CONFIGS) * len(list(SEEDS))
    sbatch(script, array=f"1-{n}")


if __name__ == "__main__":
    main()
