#!/bin/bash
# Stage C release driver.
#
# The account's QOS caps SUBMITTED jobs at 500 per user (mit_general/normal),
# and Stage C is 2000 array tasks across 7 families, so they cannot all be
# queued at once. A --dependency chain does not help: dependent jobs still
# count against MaxSubmit the moment they are submitted. So this polls the
# queue and releases the next family only when there is room for all of it.
#
# Families are released WHOLE and never interleaved. That is deliberate:
# downstream aggregation (np.stack over per-seed trajectories in
# 6g_aggregation_statistics.py and utils_plotting.py) hard-fails on ragged
# input, so a half-migrated family breaks Figure 3 outright rather than
# degrading gracefully. Whole-family atomicity keeps every intermediate state
# analysable.
#
# Idempotent: families whose checkpoints are already at 2001 entries are
# skipped, so this can be re-run after an interruption.
#
# Usage:  nohup bash scripts/_stageC_driver.sh > _stageC_driver.log 2>&1 &
set -uo pipefail
cd /orcd/pool/005/cwomack/Project2

LIMIT=500
declare -a FAM=(
  "_gen_regen_array_BC 300"
  "_gen_regen_array_CH4 300"
  "_gen_regen_array_N2O 300"
  "_gen_regen_array_Sulfur 300"
  "_gen_multi_regen_array 250"
  "_gen_multi_fig4_regen_array 250"
)

# Count SUBMITTED tasks (-r expands array elements, which is what the QOS counts).
queued() { squeue -u "$USER" -h -r 2>/dev/null | wc -l; }

for entry in "${FAM[@]}"; do
  set -- $entry; SCRIPT=$1; N=$2
  echo "[$(date -Is)] next: $SCRIPT ($N tasks)"
  while :; do
    Q=$(queued)
    ROOM=$((LIMIT - Q))
    if [ "$ROOM" -ge "$N" ]; then
      OUT=$(sbatch --array=1-${N}%40 --job-name=stageC-$SCRIPT scripts/$SCRIPT.slurm 2>&1)
      RC=$?
      echo "[$(date -Is)] queued=$Q room=$ROOM -> sbatch rc=$RC: $OUT"
      if [ "$RC" -eq 0 ]; then
        break
      fi
      # Transient rejection (race against another submission): back off, retry.
      echo "[$(date -Is)] submit rejected, retrying in 120s"
    else
      echo "[$(date -Is)] queued=$Q room=$ROOM < $N, waiting"
    fi
    sleep 120
  done
done

echo "[$(date -Is)] all families released; waiting for drain"
while [ "$(queued)" -gt 0 ]; do sleep 300; done
echo "[$(date -Is)] Stage C queue drained"
