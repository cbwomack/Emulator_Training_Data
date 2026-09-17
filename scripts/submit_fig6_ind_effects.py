#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Opus 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Submits Figure 6's individual-effects seed-spread cache build
(build_fig6_ind_effects_cache.py --mode run-one) as a 50-task SLURM
array, one task per seed. Mirrors submit_stage_e_array.py exactly.

Usage:
    python scripts/submit_fig6_ind_effects.py
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
    parser.add_argument(
        "--multi-checkpoint-dir", default=None,
        help="passed through to build_fig6_ind_effects_cache.py; defaults "
             "to checkpoints/multi_retuned_smooth/seed_sweep.")
    parser.add_argument(
        "--suffix", default="",
        help="passed through to build_fig6_ind_effects_cache.py; required "
             "whenever --multi-checkpoint-dir is given.")
    args = parser.parse_args()

    # Mirror the builder's own guard here so a bad submission fails before it
    # costs 50 array tasks rather than after each one exits.
    if args.multi_checkpoint_dir is not None and not args.suffix:
        raise SystemExit("--multi-checkpoint-dir requires --suffix")

    extra = ""
    if args.multi_checkpoint_dir is not None:
        extra += f' --multi-checkpoint-dir {args.multi_checkpoint_dir}'
    if args.suffix:
        extra += f' --suffix {args.suffix}'

    script = PROJECT_ROOT / f"scripts/_gen_fig6_ind_effects_array{args.suffix}.slurm"
    out_dir = Path("data/SI_results/seed_uncertainty/slurm_logs")
    write_slurm(
        script, f"fig6_indeffects{args.suffix}", out_dir, "00:20:00",
        f'conda run -n project2 python -u scripts/build_fig6_ind_effects_cache.py '
        f'--mode run-one --seed "$SLURM_ARRAY_TASK_ID"{extra}'
    )
    sbatch(script, array=f"0-{N_SEEDS - 1}%50")


if __name__ == "__main__":
    main()
