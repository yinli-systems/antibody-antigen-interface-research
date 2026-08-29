#!/bin/bash
#SBATCH --job-name=fc-bound-unbound-eval-v1
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/bound-unbound-geometry-v1/logs/evaluate-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/bound-unbound-geometry-v1/logs/evaluate-%A_%a.err

set -euo pipefail
test "${SLURM_JOB_PARTITION:-}" = "gpu_4090"
: "${PROTOCOL_SHA256:?PROTOCOL_SHA256 is required}"

PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/bound-unbound-geometry-v1
PROTOCOL=${ROOT}/authority/PROTOCOL_BOUND_UNBOUND_MEASUREMENT_V1.json
DATA_ROOT=${ROOT}/measurement-data-v1
DATA_MANIFEST=${DATA_ROOT}/BOUND_UNBOUND_MEASUREMENT_DATA_MANIFEST_V1.json
DATA_RECEIPT=${DATA_ROOT}/BOUND_UNBOUND_MEASUREMENT_DATA_RECEIPT_V1.json
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ARMS=(baseline_current baseline_current baseline_current edge_topk2_supervised edge_topk2_supervised edge_topk2_supervised)
SEEDS=(17 29 43 17 29 43)
SOURCE_ROOTS=(
  /ssd/scxi253/model_deployment/immutable_stage_a1_candidates/matched_local_training_source_v3
  /ssd/scxi253/model_deployment/immutable_stage_a1_candidates/matched_local_training_source_v3
  /ssd/scxi253/model_deployment/immutable_stage_a1_candidates/matched_local_training_source_v3
  /ssd/scxi253/model_deployment/immutable_stage_a1_candidates/edge_topk2_training_source_v1
  /ssd/scxi253/model_deployment/immutable_stage_a1_candidates/edge_topk2_training_source_v1
  /ssd/scxi253/model_deployment/immutable_stage_a1_candidates/edge_topk2_training_source_v1
)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID}]}
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
MODEL_SOURCE=${SOURCE_ROOTS[${SLURM_ARRAY_TASK_ID}]}
OUT_ROOT=${ROOT}/measurement-components-v1/${ARM}/seed-${SEED}
DATA_SHA=$(sha256sum "${DATA_MANIFEST}" | awk '{print $1}')

test "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" = "${PROTOCOL_SHA256}"
"${PYTHON}" - "${DATA_RECEIPT}" "${DATA_SHA}" <<'PY'
import json,sys
receipt=json.load(open(sys.argv[1]))
if receipt.get('manifest_sha256') != sys.argv[2] or receipt.get('selection_performed') is not False:
    raise SystemExit('data receipt differs')
PY
"${PYTHON}" "${SOURCE}/evaluate_bound_unbound_component_v1.py" \
  --protocol "${PROTOCOL}" \
  --protocol-sha256 "${PROTOCOL_SHA256}" \
  --data-root "${DATA_ROOT}" \
  --data-manifest-sha256 "${DATA_SHA}" \
  --arm "${ARM}" \
  --seed "${SEED}" \
  --source-root "${MODEL_SOURCE}" \
  --output "${OUT_ROOT}/RESULT.json" \
  --receipt "${OUT_ROOT}/RECEIPT.json"
chmod -R a-w "${OUT_ROOT}"
