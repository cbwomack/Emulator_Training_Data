#!/usr/bin/env python
# ============================================================
# Author: Christopher B. Womack
# Coding assistance provided by Claude Sonnet 5.
# Responsibility for the final manuscript/code lies entirely with the authors.
# GAI tools are not listed as authors and do not bear responsibility for the
# final outcomes.
# ============================================================

"""
Submits Stage 6b's --mode control combos (6b_iteration0_ablation.py) as a
SLURM array, one task per (agent, group, ic, seed) - a full 50-seed sweep per
(agent, group, ic), matching Stage 6a/E's UQ protocol so the iteration-0
controls report a seed range like every other result (per user direction).
Skips any agent whose baseline K=400 config doesn't exist yet (e.g. Sulfur,
still mid-pipeline) rather than submitting tasks guaranteed to fail - rerun
once that agent's ready.

At 50 seeds, the full combo set (~900 tasks once every agent is ready)
exceeds the account's ~448-job submit cap in one `sbatch --array` call, so
this submits in BATCH_SIZE-task chunks, reusing agent_pipeline_runner's
retry-hardened sbatch() to absorb the resulting QOS contention between
chunks/agents rather than requiring them to be spaced out by hand. Each task
is independently idempotent (6b_iteration0_ablation.py's control() skips any
combo whose checkpoint already exists), so re-running this script after a
partial submission is always safe.

Usage:
    python scripts/submit_stage_6b_controls.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent_pipeline_runner import sbatch, write_slurm  # noqa: E402
import importlib.util

BATCH_SIZE = 200


def _load_ablation_module():
    spec = importlib.util.spec_from_file_location(
        "_stage6b", PROJECT_ROOT / "scripts" / "6b_iteration0_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _baseline_ready(agent):
    path = (
        Path("data/SI_results/baseline_hp/k400_search/best_baseline_config_K400.json") if agent == "CO2"
        else Path(f"data/SI_results/baseline_hp/k400_search_{agent}/best_baseline_config_K400.json")
    )
    return path.exists()


def main():
    stage6b = _load_ablation_module()
    combos = stage6b.all_control_combos()

    ready = [c for c in combos if _baseline_ready(c[0])]
    skipped_agents = sorted({c[0] for c in combos if c not in ready})
    if skipped_agents:
        n_skipped = len(combos) - len(ready)
        print(f"skipping {n_skipped} combos for agent(s) not ready yet: {skipped_agents}")
    if not ready:
        print("nothing ready to submit")
        return

    combos_path = Path("checkpoints/.stage6b_control_combos.txt")
    combos_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combos_path, "w") as f:
        for agent, group, ic, seed in ready:
            f.write(f"{agent} {group} {ic} {seed}\n")

    script = PROJECT_ROOT / "scripts/_gen_stage_6b_control_array.slurm"
    out_dir = Path("data/SI_results/iteration0_ablation/slurm_logs")
    write_slurm(
        script, "stage6b_control", out_dir, "00:20:00",
        f'COMBOS=$PROJECT_DIR/{combos_path}\n'
        f'LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMBOS")\n'
        f'AGENT=$(echo "$LINE" | awk \'{{print $1}}\')\n'
        f'GROUP=$(echo "$LINE" | awk \'{{print $2}}\')\n'
        f'IC=$(echo "$LINE" | awk \'{{print $3}}\')\n'
        f'SEED=$(echo "$LINE" | awk \'{{print $4}}\')\n'
        f'conda run -n project2 python -u scripts/6b_iteration0_ablation.py '
        f'--mode control --agent "$AGENT" --group "$GROUP" --ic "$IC" --seed "$SEED"'
    )

    n = len(ready)
    for lo in range(1, n + 1, BATCH_SIZE):
        hi = min(lo + BATCH_SIZE - 1, n)
        sbatch(script, array=f"{lo}-{hi}%50")


if __name__ == "__main__":
    main()
