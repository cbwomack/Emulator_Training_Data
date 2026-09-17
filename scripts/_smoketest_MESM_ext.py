import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from agent_pipeline_runner import sbatch, write_slurm  # noqa: E402

script = Path("scripts/_gen_MESM_seed_sweep_ext_smoketest.slurm")
write_slurm(
    script, "MESM_ext_smoke", Path("data/plotting/MESM_seed_sweep/slurm_logs_ext"), "00:20:00",
    'COMBOS=$PROJECT_DIR/checkpoints/.MESM_seed_sweep_combos_ext.txt\n'
    'LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMBOS")\n'
    'SEED=$(echo "$LINE" | awk \'{print $1}\')\n'
    'VARIANT=$(echo "$LINE" | awk \'{print $2}\')\n'
    'conda run -n project2 python -u scripts/4c_evaluate_MESM_emulator_seed_sweep.py '
    '--seed "$SEED" --variant "$VARIANT"'
)
jobid = sbatch(script, array="1-1")
print("JOBID", jobid)
