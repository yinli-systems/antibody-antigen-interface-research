#!/usr/bin/env bash
#SBATCH -J esmc-cdr-merge-v1
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/merge-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/merge-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1
readonly PLAN=${ROOT}/authority/PAIRED_DECONTAMINATION_EXECUTION_PLAN_V1.json
readonly PLAN_SHA256=${PAIRED_DECONTAMINATION_PLAN_SHA256:?missing frozen plan SHA-256}

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi

"${PYTHON}" "${SOURCE}/merge_paired_hit_shards_v1.py" \
  --plan "${PLAN}" --plan-sha256 "${PLAN_SHA256}" \
  --reductions-root "${ROOT}/shard-reductions-v1" \
  --output-root "${ROOT}/merged-exclusions-v1"
