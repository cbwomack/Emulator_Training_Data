#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Submits the MESM revision vector-emulator 1000-seed sweep
(scripts/6p_fig7_revision_vector_emulator.py) as batched SLURM arrays: 1000
seeds x 3 variants {constant, sine, both} = 3000 tasks (no baseline - reuse
the already-computed data/plotting/MESM_seed_sweep/baseline_seed*.pkl files
by copying them into data/plotting/MESM_revision_seed_sweep/ first, since
baseline training/eval never touches the optimized-side revision data; no
gaussian - per user direction, this sweep is constant/sine/both only).

Mirrors submit_MESM_seed_sweep_regen_ext.py's batching pattern exactly
(batches well under the account's MaxSubmit=500 cap, waiting for each batch
to clear before submitting the next) but self-contained - generates its own
combos file rather than expecting one to be scp'd in first.

Usage:
    python scripts/submit_fig7_revision_seed_sweep.py
"""
import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent_pipeline_runner import sbatch, write_slurm  # noqa: E402

VARIANTS = ["constant", "sine", "both"]
SEEDS = range(1000)
BATCH_SIZE = 380
N_TOTAL = len(list(SEEDS)) * len(VARIANTS)  # 3000

COMBOS_PATH = Path("checkpoints/.fig7_revision_seed_sweep_combos.txt")


def wait_for_queue_empty(poll_seconds=30):
    while True:
        who = subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip()
        out = subprocess.run(["squeue", "-u", who, "-h"], capture_output=True, text=True)
        n = len([l for l in out.stdout.splitlines() if l.strip()])
        if n == 0:
            return
        print(f"  {n} jobs still in queue, waiting {poll_seconds}s...", flush=True)
        time.sleep(poll_seconds)


def main():
    COMBOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COMBOS_PATH, "w") as f:
        for seed in SEEDS:
            for variant in VARIANTS:
                f.write(f"{seed} {variant}\n")

    script = PROJECT_ROOT / "scripts/_gen_fig7_revision_seed_sweep_array.slurm"
    out_dir = Path("data/plotting/MESM_revision_seed_sweep/slurm_logs")
    write_slurm(
        script, "fig7_revision_seed_sweep", out_dir, "00:20:00",
        f'COMBOS=$PROJECT_DIR/{COMBOS_PATH}\n'
        f'LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMBOS")\n'
        f'SEED=$(echo "$LINE" | awk \'{{print $1}}\')\n'
        f'VARIANT=$(echo "$LINE" | awk \'{{print $2}}\')\n'
        f'conda run -n project2 python -u scripts/6p_fig7_revision_vector_emulator.py '
        f'--mode run --seed "$SEED" --variant "$VARIANT"'
    )

    n_batches = (N_TOTAL + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(n_batches):
        lo = b * BATCH_SIZE + 1
        hi = min((b + 1) * BATCH_SIZE, N_TOTAL)
        print(f"=== batch {b + 1}/{n_batches}: array {lo}-{hi} ({hi - lo + 1} tasks) ===", flush=True)
        sbatch(script, array=f"{lo}-{hi}%50")
        wait_for_queue_empty()
        print(f"  batch {b + 1} clear.", flush=True)

    print("All batches submitted and cleared.")


if __name__ == "__main__":
    main()
