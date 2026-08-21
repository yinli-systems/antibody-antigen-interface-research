#!/usr/bin/env bash
#SBATCH -J abpairfm-oas-stageb
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-oas-stageb-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-oas-stageb-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly RAW_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/raw/oas_paired_20260821/data_units
readonly SOURCE_RECEIPT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/raw/oas_paired_20260821.materialization_v1.json
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/oas_paired_20260821
readonly OUTPUT="${OUTPUT_ROOT}/oas_stage_b_exact_pairs_v2.tsv.gz"
readonly RECEIPT="${OUTPUT_ROOT}/oas_stage_b_exact_pairs_v2.receipt.json"

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden OAS preprocessing allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
test -f "${SOURCE_RECEIPT}"
mkdir -p "${OUTPUT_ROOT}"
"${PYTHON}" "${PROJECT}/scripts/build_oas_stage_b_pairs.py" \
  --oas-root "${RAW_ROOT}" \
  --materialization-receipt "${SOURCE_RECEIPT}" \
  --output "${OUTPUT}" \
  --receipt "${RECEIPT}"
