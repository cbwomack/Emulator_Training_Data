#!/bin/bash
#SBATCH --job-name=6h-preflight
#SBATCH --partition=mit_normal
#SBATCH --array=1-10
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/orcd/pool/005/cwomack/Project2/data/SI_results/convergence_preflight/slurm_logs/preflight_%A_%a.out

# Stage A gate: out-of-sample NRMSE at outer iterates 0..1000, sharded one task
# per (family, seed). Setup alone peaks at ~2.8 GB, hence --mem=8G; seven
# iterates of K=400 inner training follow, hence the 1h wall.
set -euo pipefail
module load miniforge/25.11.0-0
PROJECT_DIR=/orcd/pool/005/cwomack/Project2
COMBOS=$PROJECT_DIR/data/SI_results/convergence_preflight/preflight_combos.txt
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMBOS")
FAMILY=$(echo "$LINE" | awk '{print $1}')
SEED=$(echo "$LINE" | awk '{print $2}')
echo "task $SLURM_ARRAY_TASK_ID -> family=$FAMILY seed=$SEED"
cd "$PROJECT_DIR"
conda run -n project2 python -u scripts/6h_convergence_preflight.py \
    --mode shard --family "$FAMILY" --seed-list "$SEED"
