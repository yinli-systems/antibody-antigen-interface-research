#!/bin/bash
# Submit zero-shot and trained evaluations after a successful smoke job.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 SMOKE_JOB_ID" >&2
  exit 2
fi

SMOKE_JOB_ID=$1
DEPLOY_ROOT=/cluster/project/model_deployment
JOB_ROOT=$DEPLOY_ROOT/jobs/sppider_seq
MODEL_ROOT=$DEPLOY_ROOT/models/sppider_seq
REPORT_PATH=$DEPLOY_ROOT/reports/sppider-seq-formal-jobs.tsv
RUN_NAME=trained_seed42

mkdir -p "$(dirname "$REPORT_PATH")"
printf 'stage\tvariant\trole\tjob_id\tdependency\n' > "$REPORT_PATH"

submit_inference_and_assembly() {
  local variant=$1
  local role=$2
  local checkpoint=$3
  local dependency=$4
  local infer_job
  local assemble_job
  infer_job=$(sbatch --parsable \
    --dependency="afterok:$dependency" \
    --export="ALL,SPPIDER_ROLE=$role,SPPIDER_RUN_NAME=$variant,SPPIDER_CHECKPOINT=$checkpoint" \
    "$JOB_ROOT/sppider_seq_infer.sbatch")
  printf 'infer\t%s\t%s\t%s\t%s\n' "$variant" "$role" "$infer_job" "$dependency" >> "$REPORT_PATH"
  assemble_job=$(sbatch --parsable \
    --dependency="afterok:$infer_job" \
    --export="ALL,SPPIDER_ROLE=$role,SPPIDER_RUN_NAME=$variant" \
    "$JOB_ROOT/sppider_seq_assemble.sbatch")
  printf 'assemble\t%s\t%s\t%s\t%s\n' "$variant" "$role" "$assemble_job" "$infer_job" >> "$REPORT_PATH"
}

for role in paratope epitope; do
  submit_inference_and_assembly \
    zero_shot_receptor \
    "$role" \
    "$MODEL_ROOT/crossattn_rec_run27_best.pt" \
    "$SMOKE_JOB_ID"
  submit_inference_and_assembly \
    zero_shot_peptide \
    "$role" \
    "$MODEL_ROOT/crossattn_pep_run07_best.pt" \
    "$SMOKE_JOB_ID"
done

for role in paratope epitope; do
  dependency=$SMOKE_JOB_ID
  for segment in $(seq 1 10); do
    train_job=$(sbatch --parsable \
      --dependency="afterok:$dependency" \
      --export="ALL,SPPIDER_ROLE=$role,SPPIDER_RUN_NAME=$RUN_NAME,SPPIDER_EPOCHS_THIS_RUN=5" \
      "$JOB_ROOT/sppider_seq_train.sbatch")
    printf 'train_%02d\t%s\t%s\t%s\t%s\n' "$segment" "$RUN_NAME" "$role" "$train_job" "$dependency" >> "$REPORT_PATH"
    dependency=$train_job
  done
  submit_inference_and_assembly \
    "$RUN_NAME" \
    "$role" \
    "$DEPLOY_ROOT/results/sabdab2/sppider_seq/$RUN_NAME/$role/checkpoints/best.pt" \
    "$dependency"
done

cat "$REPORT_PATH"
