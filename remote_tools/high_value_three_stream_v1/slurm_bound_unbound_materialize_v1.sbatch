#!/bin/bash
#SBATCH --job-name=fc-bound-unbound-data-v1
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/bound-unbound-geometry-v1/logs/materialize-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/bound-unbound-geometry-v1/logs/materialize-%j.err

set -euo pipefail
test "${SLURM_JOB_PARTITION:-}" = "gpu_4090"
: "${PROTOCOL_SHA256:?PROTOCOL_SHA256 is required}"

PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/bound-unbound-geometry-v1
PROTOCOL=${ROOT}/authority/PROTOCOL_BOUND_UNBOUND_MEASUREMENT_V1.json
LABEL_FREE=${ROOT}/BOUND_UNBOUND_LABEL_FREE_MANIFEST_V1.json
OUTPUT=${ROOT}/measurement-data-v1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/ssd/scxi253/model_deployment/envs/epiformer/lib/python3.10/site-packages

test "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" = "${PROTOCOL_SHA256}"
"${PYTHON}" "${SOURCE}/materialize_bound_unbound_measurement_v1.py" \
  --label-free-manifest "${LABEL_FREE}" \
  --label-free-manifest-sha256 772969d7d40bf945e0d1b41f7e38d3978f14051967c4b1f8dc9dc8c8539adfc4 \
  --protocol "${PROTOCOL}" \
  --protocol-sha256 "${PROTOCOL_SHA256}" \
  --source-root /ssd/scxi253/model_deployment/immutable_stage_a1_candidates/edge_topk2_training_source_v1 \
  --model-root /ssd/scxi253/model_deployment/models/sppider_seq/esm2_t33_650M_UR50D \
  --output-root "${OUTPUT}"
chmod -R a-w "${OUTPUT}"
