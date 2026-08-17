#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Submits the multi-agent baseline K=400 search (6e_baseline_hp_search_k400_
multi.py --mode search-one) as a SLURM array, one task per (config_idx,
seed). 100 configs x 5 seeds = 500 tasks, exceeding the ~448-job submit cap
in one call, so this submits in BATCH_SIZE-task chunks, reusing agent_
pipeline_runner's retry-hardened sbatch()/write_slurm() helpers.

Usage:
    python scripts/submit_multi_baseline_k400.py
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent_pipeline_runner import sbatch, write_slurm  # noqa: E402

SEEDS = range(5)
BATCH_SIZE = 200


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-configs", type=int, default=100)
    args = parser.parse_args()

    combos_path = Path("checkpoints/.multi_baseline_k400_combos.txt")
    combos_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combos_path, "w") as f:
        for idx in range(args.n_configs):
            for seed in SEEDS:
                f.write(f"{idx} {seed}\n")

    script = PROJECT_ROOT / "scripts/_gen_multi_baseline_k400_array.slurm"
    out_dir = Path("data/SI_results/baseline_hp/k400_search_multi/slurm_logs")
    write_slurm(
        script, "multi_baseline_k400", out_dir, "00:10:00",
        f'COMBOS=$PROJECT_DIR/{combos_path}\n'
        f'LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMBOS")\n'
        f'IDX=$(echo "$LINE" | awk \'{{print $1}}\')\n'
        f'SEED=$(echo "$LINE" | awk \'{{print $2}}\')\n'
        f'conda run -n project2 python -u scripts/6e_baseline_hp_search_k400_multi.py '
        f'--mode search-one --config-idx "$IDX" --seed "$SEED"'
    )

    n = args.n_configs * len(list(SEEDS))
    for lo in range(1, n + 1, BATCH_SIZE):
        hi = min(lo + BATCH_SIZE - 1, n)
        sbatch(script, array=f"{lo}-{hi}%50")


if __name__ == "__main__":
    main()
