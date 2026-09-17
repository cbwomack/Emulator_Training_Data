#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Extension of submit_MESM_seed_sweep_regen.py: seeds 50-999 (950 additional
seeds x 4 variants = 3800 tasks), for the 1000-seed exploratory rerun of the
Figure 7 (MESM vector emulator) baseline-vs-optimized comparison. A one-off
ablation, not part of the reviewer-response pipeline proper.

Reuses the combos-file / array pattern exactly, but the task count (3800)
is well over the account's MaxSubmit=500, so - mirroring agent_pipeline_
runner.py's stage_cheap batching - this submits sequential array batches
(380 tasks each, 10 batches) into ONE shared combos file, waiting for each
batch to fully clear (squeue empty) before submitting the next, so the
instantaneous submitted+pending count never approaches the cap.

Usage:
    python scripts/submit_MESM_seed_sweep_regen_ext.py
"""
import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent_pipeline_runner import sbatch, write_slurm  # noqa: E402

VARIANTS = ["baseline", "constant", "sine", "both"]
SEEDS = range(50, 1000)  # 950 seeds
BATCH_SIZE = 380
N_TOTAL = len(list(SEEDS)) * len(VARIANTS)  # 3800

COMBOS_PATH = Path("checkpoints/.MESM_seed_sweep_combos_ext.txt")


def wait_for_queue_empty(poll_seconds=30):
    while True:
        out = subprocess.run(["squeue", "-u", subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip(),
                               "-h"], capture_output=True, text=True)
        n = len([l for l in out.stdout.splitlines() if l.strip()])
        if n == 0:
            return
        print(f"  {n} jobs still in queue, waiting {poll_seconds}s...", flush=True)
        time.sleep(poll_seconds)


def main():
    assert COMBOS_PATH.exists(), f"{COMBOS_PATH} missing - scp it from the local container first"
    with open(COMBOS_PATH) as f:
        n_lines = sum(1 for _ in f)
    assert n_lines == N_TOTAL, f"combos file has {n_lines} lines, expected {N_TOTAL}"

    script = PROJECT_ROOT / "scripts/_gen_MESM_seed_sweep_regen_ext_array.slurm"
    out_dir = Path("data/plotting/MESM_seed_sweep/slurm_logs_ext")
    write_slurm(
        script, "MESM_seed_sweep_ext", out_dir, "00:20:00",
        f'COMBOS=$PROJECT_DIR/{COMBOS_PATH}\n'
        f'LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMBOS")\n'
        f'SEED=$(echo "$LINE" | awk \'{{print $1}}\')\n'
        f'VARIANT=$(echo "$LINE" | awk \'{{print $2}}\')\n'
        f'conda run -n project2 python -u scripts/4c_evaluate_MESM_emulator_seed_sweep.py '
        f'--seed "$SEED" --variant "$VARIANT"'
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
