#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Submits the multi-agent hyperparameter validate stage (0b_hyperparameter_
retune_multi.py --mode validate-one) as a SLURM array, one task per
(cand_idx, group, seed) - 5 candidates x 5 groups x 10 seeds = 250 tasks,
under the ~448-job submit cap so no batching needed.

Usage:
    python scripts/submit_multi_hp_validate.py
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent_pipeline_runner import sbatch, write_slurm  # noqa: E402

GROUPS = ["H-ext", "tier1", "DAMIP", "GeoMIP", "all"]
SEEDS = range(10)
VALIDATE_NUM_UPDATES = 1000


def main():
    candidates_path = Path("data/SI_results/hp_retune/multi/top_candidates_unified.json")
    if not candidates_path.exists():
        raise SystemExit(f"{candidates_path} missing - run --mode select first")
    n_candidates = len(json.load(open(candidates_path)))

    combos_path = Path("checkpoints/.multi_hp_validate_combos.txt")
    combos_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combos_path, "w") as f:
        for cand_idx in range(n_candidates):
            for group in GROUPS:
                for seed in SEEDS:
                    f.write(f"{cand_idx} {group} {seed}\n")

    script = PROJECT_ROOT / "scripts/_gen_multi_hp_validate_array.slurm"
    out_dir = Path("data/SI_results/hp_retune/multi/slurm_logs")
    write_slurm(
        script, "multi_hp_validate", out_dir, "00:20:00",
        f'COMBOS=$PROJECT_DIR/{combos_path}\n'
        f'LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMBOS")\n'
        f'CAND=$(echo "$LINE" | awk \'{{print $1}}\')\n'
        f'GROUP=$(echo "$LINE" | awk \'{{print $2}}\')\n'
        f'SEED=$(echo "$LINE" | awk \'{{print $3}}\')\n'
        f'conda run -n project2 python -u scripts/0b_hyperparameter_retune_multi.py '
        f'--mode validate-one --cand-idx "$CAND" --group "$GROUP" --validate-seeds "$SEED" '
        f'--validate-num-updates {VALIDATE_NUM_UPDATES}'
    )
    n = n_candidates * len(GROUPS) * len(list(SEEDS))
    sbatch(script, array=f"1-{n}%50")


if __name__ == "__main__":
    main()
