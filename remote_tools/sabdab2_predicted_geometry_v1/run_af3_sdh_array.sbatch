#!/bin/bash
#SBATCH --job-name=sabdab2-sdh-af3
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

set -euo pipefail

CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-/data/run01/scxi253/fanxuezhe/projects/20260424_af3_qiangsheng/scripts/run_5090_singularity.sh}"
BATCH_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
BATCH_DIR=$(printf '%s/batches/batch-%03d' "$CAMPAIGN_ROOT" "$BATCH_INDEX")

if [[ ! -d "$BATCH_DIR/inputs" ]]; then
  echo "AF3_SDH_INPUT_ERROR missing batch input directory: $BATCH_DIR/inputs" >&2
  exit 2
fi

mkdir -p "$BATCH_DIR/results" "$CAMPAIGN_ROOT/slurm"
cd "$BATCH_DIR"

echo "AF3_SDH_BATCH_START batch=$BATCH_INDEX job=$SLURM_JOB_ID host=$(hostname) time=$(date -Is)"
bash "$RUNNER_SCRIPT" inputs results

MODEL_COUNT=$(find results -mindepth 2 -maxdepth 2 -type f -name '*_model.cif' | wc -l)
INPUT_COUNT=$(find inputs -maxdepth 1 -type f -name '*.json' | wc -l)
if [[ "$MODEL_COUNT" -ne "$INPUT_COUNT" ]]; then
  echo "AF3_SDH_OUTPUT_ERROR batch=$BATCH_INDEX inputs=$INPUT_COUNT models=$MODEL_COUNT" >&2
  exit 3
fi

find results -type f -print0 | sort -z | xargs -0 sha256sum > "$BATCH_DIR/RESULT_SHA256SUMS.txt"
echo "AF3_SDH_BATCH_PASS batch=$BATCH_INDEX inputs=$INPUT_COUNT models=$MODEL_COUNT time=$(date -Is)"
