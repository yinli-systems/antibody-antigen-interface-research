#!/bin/bash
set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
INITIAL_DEPENDENCY=${1:?usage: mipe_schedule_to_800.sh INITIAL_DEPENDENCY_JOB_ID}
MANIFEST=$DEPLOY_ROOT/reports/mipe-formal-seed42-to-800-jobs.tsv

printf 'fold\ttarget_epoch\tjob_id\tdependency_job_id\n' > "$MANIFEST"
for fold in 0 1 2 3 4; do
  previous_job=$INITIAL_DEPENDENCY
  for target_epoch in 200 300 400 500 600 700 800; do
    job_id=$(sbatch --parsable \
      --dependency="afterok:$previous_job" \
      --export="ALL,MIPE_FOLD=$fold,MIPE_EPOCHS=$target_epoch,MIPE_RUN_NAME=formal_seed42" \
      "$DEPLOY_ROOT/jobs/mipe_train.sbatch")
    printf '%s\t%s\t%s\t%s\n' \
      "$fold" "$target_epoch" "$job_id" "$previous_job" >> "$MANIFEST"
    previous_job=$job_id
  done
done

cat "$MANIFEST"
