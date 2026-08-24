#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Submits the MESM vector-emulator optimized-side HP search
(4c_MESM_optimized_hp_search.py --mode search-one) as a SLURM array, one
task per (config_idx, seed). 100 configs x 5 seeds = 500 tasks, batched like
submit_MESM_baseline_hp_search.py.

Only submits the default "both" --train-variant (see 4c_MESM_optimized_hp_
search.py's module docstring for why one shared search covers all three
optimized-side variants) - use --train-variant to submit the escape-hatch
per-variant search instead, if a later smoke test shows the shared config
underperforms specifically on constant-only or sine-only.

Usage:
    python scripts/submit_MESM_optimized_hp_search.py
    python scripts/submit_MESM_optimized_hp_search.py --train-variant constant
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
    parser.add_argument("--train-variant", choices=["constant", "sine", "both"], default="both")
    args = parser.parse_args()

    combos_path = Path(f"checkpoints/.MESM_optimized_hp_{args.train_variant}_combos.txt")
    combos_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combos_path, "w") as f:
        for idx in range(args.n_configs):
            for seed in SEEDS:
                f.write(f"{idx} {seed}\n")

    script = PROJECT_ROOT / f"scripts/_gen_MESM_optimized_hp_{args.train_variant}_array.slurm"
    out_dir = Path(f"data/SI_results/hp_retune/MESM_vector/optimized_search"
                    f"{'' if args.train_variant == 'both' else '_' + args.train_variant}/slurm_logs")
    write_slurm(
        script, f"MESM_opt_hp_{args.train_variant}", out_dir, "00:10:00",
        f'COMBOS=$PROJECT_DIR/{combos_path}\n'
        f'LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMBOS")\n'
        f'IDX=$(echo "$LINE" | awk \'{{print $1}}\')\n'
        f'SEED=$(echo "$LINE" | awk \'{{print $2}}\')\n'
        f'conda run -n project2 python -u scripts/4c_MESM_optimized_hp_search.py '
        f'--mode search-one --config-idx "$IDX" --seed "$SEED" --train-variant {args.train_variant}'
    )

    n = args.n_configs * len(list(SEEDS))
    for lo in range(1, n + 1, BATCH_SIZE):
        hi = min(lo + BATCH_SIZE - 1, n)
        sbatch(script, array=f"{lo}-{hi}%50")


if __name__ == "__main__":
    main()
