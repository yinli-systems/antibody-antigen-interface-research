#!/bin/bash
#SBATCH --job-name=fc-bound-unbound-analyze-v1
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/bound-unbound-geometry-v1/logs/analyze-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/bound-unbound-geometry-v1/logs/analyze-%j.err

set -euo pipefail
test "${SLURM_JOB_PARTITION:-}" = "gpu_4090"
: "${PROTOCOL_SHA256:?PROTOCOL_SHA256 is required}"

PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/bound-unbound-geometry-v1
PROTOCOL=${ROOT}/authority/PROTOCOL_BOUND_UNBOUND_MEASUREMENT_V1.json
DATA_ROOT=${ROOT}/measurement-data-v1
DATA_MANIFEST=${DATA_ROOT}/BOUND_UNBOUND_MEASUREMENT_DATA_MANIFEST_V1.json
DATA_SHA=$(sha256sum "${DATA_MANIFEST}" | awk '{print $1}')

test "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" = "${PROTOCOL_SHA256}"
"${PYTHON}" "${SOURCE}/analyze_bound_unbound_measurement_v1.py" \
  --protocol "${PROTOCOL}" \
  --protocol-sha256 "${PROTOCOL_SHA256}" \
  --data-root "${DATA_ROOT}" \
  --data-manifest-sha256 "${DATA_SHA}" \
  --components-root "${ROOT}/measurement-components-v1" \
  --output "${ROOT}/BOUND_UNBOUND_MEASUREMENT_RESULT_V1.json" \
  --receipt "${ROOT}/BOUND_UNBOUND_MEASUREMENT_RESULT_V1_RECEIPT.json"
chmod a-w "${ROOT}/BOUND_UNBOUND_MEASUREMENT_RESULT_V1.json" "${ROOT}/BOUND_UNBOUND_MEASUREMENT_RESULT_V1_RECEIPT.json"
