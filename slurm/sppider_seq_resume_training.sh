#!/bin/bash
# Resume one role-specific training chain and schedule its final evaluation.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 ROLE [RUN_NAME] [REPORT_PATH]" >&2
  exit 2
fi

ROLE=$1
RUN_NAME=${2:-trained_seed42}
DEPLOY_ROOT=/cluster/project/model_deployment
JOB_ROOT=$DEPLOY_ROOT/jobs/sppider_seq
RUN_ROOT=$DEPLOY_ROOT/results/sabdab2/sppider_seq/$RUN_NAME/$ROLE
HISTORY_PATH=$RUN_ROOT/checkpoints/history.json
REPORT_PATH=${3:-$DEPLOY_ROOT/reports/sppider-seq-resume-$RUN_NAME-$ROLE.tsv}
PYTHON=$DEPLOY_ROOT/envs/epiformer/bin/python

case "$ROLE" in
  paratope|epitope) ;;
  *)
    echo "ROLE must be paratope or epitope" >&2
    exit 2
    ;;
esac

if [[ ! -f "$HISTORY_PATH" ]]; then
  echo "Missing training history: $HISTORY_PATH" >&2
  exit 1
fi

current_epoch=$(
  "$PYTHON" -c \
    'import json, sys; history=json.load(open(sys.argv[1]))["history"]; print(int(history[-1]["epoch"]))' \
    "$HISTORY_PATH"
)
if (( current_epoch < 1 || current_epoch > 50 )); then
  echo "Unexpected completed epoch: $current_epoch" >&2
  exit 1
fi

mkdir -p "$(dirname "$REPORT_PATH")"
printf 'stage\tvariant\trole\tjob_id\tdependency\n' > "$REPORT_PATH"

dependency=
remaining_epochs=$((50 - current_epoch))
segment_count=$(((remaining_epochs + 4) / 5))
for segment in $(seq 1 "$segment_count"); do
  if [[ -n "$dependency" ]]; then
    train_job=$(sbatch --parsable \
      --dependency="afterok:$dependency" \
      --export="ALL,SPPIDER_ROLE=$ROLE,SPPIDER_RUN_NAME=$RUN_NAME,SPPIDER_EPOCHS_THIS_RUN=5" \
      "$JOB_ROOT/sppider_seq_train.sbatch")
  else
    train_job=$(sbatch --parsable \
      --export="ALL,SPPIDER_ROLE=$ROLE,SPPIDER_RUN_NAME=$RUN_NAME,SPPIDER_EPOCHS_THIS_RUN=5" \
      "$JOB_ROOT/sppider_seq_train.sbatch")
  fi
  printf 'resume_%02d\t%s\t%s\t%s\t%s\n' \
    "$segment" "$RUN_NAME" "$ROLE" "$train_job" "${dependency:-none}" \
    >> "$REPORT_PATH"
  dependency=$train_job
done

checkpoint=$RUN_ROOT/checkpoints/best.pt
if [[ -n "$dependency" ]]; then
  infer_job=$(sbatch --parsable \
    --dependency="afterok:$dependency" \
    --export="ALL,SPPIDER_ROLE=$ROLE,SPPIDER_RUN_NAME=$RUN_NAME,SPPIDER_CHECKPOINT=$checkpoint" \
    "$JOB_ROOT/sppider_seq_infer.sbatch")
else
  infer_job=$(sbatch --parsable \
    --export="ALL,SPPIDER_ROLE=$ROLE,SPPIDER_RUN_NAME=$RUN_NAME,SPPIDER_CHECKPOINT=$checkpoint" \
    "$JOB_ROOT/sppider_seq_infer.sbatch")
fi
printf 'infer\t%s\t%s\t%s\t%s\n' \
  "$RUN_NAME" "$ROLE" "$infer_job" "${dependency:-none}" >> "$REPORT_PATH"

assemble_job=$(sbatch --parsable \
  --dependency="afterok:$infer_job" \
  --export="ALL,SPPIDER_ROLE=$ROLE,SPPIDER_RUN_NAME=$RUN_NAME" \
  "$JOB_ROOT/sppider_seq_assemble.sbatch")
printf 'assemble\t%s\t%s\t%s\t%s\n' \
  "$RUN_NAME" "$ROLE" "$assemble_job" "$infer_job" >> "$REPORT_PATH"

cat "$REPORT_PATH"
