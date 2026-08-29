#!/usr/bin/env bash
#SBATCH -J esmc-cdr-filter-v1
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/filter-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/filter-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE_CODE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1
readonly DATA=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/oas_paired_20260821
readonly PLAN=${ROOT}/authority/PAIRED_DECONTAMINATION_EXECUTION_PLAN_V1.json
readonly PLAN_SHA256=${PAIRED_DECONTAMINATION_PLAN_SHA256:?missing frozen plan SHA-256}

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi

"${PYTHON}" "${SOURCE_CODE}/materialize_paired_devclean_v1.py" \
  --plan "${PLAN}" --plan-sha256 "${PLAN_SHA256}" \
  --merged-root "${ROOT}/merged-exclusions-v1" \
  --output "${DATA}/oas_stage_b_devclean_50id80cov_sharded_v1.tsv.gz" \
  --receipt "${DATA}/oas_stage_b_devclean_50id80cov_sharded_v1.receipt.json"
