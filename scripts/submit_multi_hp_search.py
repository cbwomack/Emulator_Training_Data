#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Submits the multi-agent hyperparameter cheap-search (0b_hyperparameter_
retune_multi.py --mode cheap-search-one) as a SLURM array, one task per
(config_idx, group) combo. N_CONFIGS x len(GROUPS) = 100 x 5 = 500 tasks,
which exceeds the account's ~448-job submit cap in one `sbatch --array`
call, so this submits in BATCH_SIZE-task chunks, reusing agent_pipeline_
runner's retry-hardened sbatch()/write_slurm() helpers. Each task is
independently idempotent (cheap_search_one skips any combo whose result
file already exists), so re-running this script after a partial submission
is always safe.

Usage:
    python scripts/submit_multi_hp_search.py
    python scripts/submit_multi_hp_search.py --n-configs 100
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent_pipeline_runner import sbatch, write_slurm  # noqa: E402

GROUPS = ["H-ext", "tier1", "DAMIP", "GeoMIP", "all"]
BATCH_SIZE = 200


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-configs", type=int, default=100)
    args = parser.parse_args()

    combos_path = Path("checkpoints/.multi_hp_cheap_combos.txt")
    combos_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combos_path, "w") as f:
        for idx in range(args.n_configs):
            for group in GROUPS:
                f.write(f"{idx} {group}\n")

    script = PROJECT_ROOT / "scripts/_gen_multi_hp_cheap_array.slurm"
    out_dir = Path("data/SI_results/hp_retune/multi/slurm_logs")
    write_slurm(
        script, "multi_hp_cheap", out_dir, "00:15:00",
        f'COMBOS=$PROJECT_DIR/{combos_path}\n'
        f'LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMBOS")\n'
        f'IDX=$(echo "$LINE" | awk \'{{print $1}}\')\n'
        f'GROUP=$(echo "$LINE" | awk \'{{print $2}}\')\n'
        f'conda run -n project2 python -u scripts/0b_hyperparameter_retune_multi.py '
        f'--mode cheap-search-one --config-idx "$IDX" --group "$GROUP"'
    )

    n = args.n_configs * len(GROUPS)
    for lo in range(1, n + 1, BATCH_SIZE):
        hi = min(lo + BATCH_SIZE - 1, n)
        sbatch(script, array=f"{lo}-{hi}%50")


if __name__ == "__main__":
    main()
