#!/usr/bin/env bash
#SBATCH -J oas-pair-decontam-audit
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/logs/pairwise-decontamination-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/logs/pairwise-decontamination-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/oas_paired_continuation_v1_20260825_v2/source
readonly DATA=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/oas_paired_20260821
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1
readonly SCRIPT_SHA256=8b17f23990bbf8153facce1be6aaa24d7545645911eb8e0ba39cb69be7df439d
readonly HITS_SHA256=edf4c952061ffef3e48b24ddb61a30d303a9cca26a3fa47c5ddcb6ce8f8f8338

readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden audit allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
test -x "${PYTHON}"
test -f "${PROJECT}/audit_pairwise_decontamination.py"
test -f "${PROJECT}/PROTOCOL_PAIRWISE_DECONTAMINATION_AUDIT_V1.json"
test -f "${DATA}/oas_stage_b_development_homology_hits_reversed_v3.tsv.gz"
test -f "${DATA}/oas_stage_b_exact_pairs_v2.receipt.json"
mkdir -p "${OUTPUT_ROOT}/logs"
test ! -e "${OUTPUT_ROOT}/PAIRWISE_DECONTAMINATION_AUDIT_V1.json"

readonly OBSERVED_SCRIPT_SHA256=$(sha256sum "${PROJECT}/audit_pairwise_decontamination.py" | awk '{print $1}')
if [[ "${OBSERVED_SCRIPT_SHA256}" != "${SCRIPT_SHA256}" ]]; then
  echo "Frozen audit script SHA-256 differs" >&2
  exit 2
fi

"${PYTHON}" "${PROJECT}/audit_pairwise_decontamination.py" \
  --hits "${DATA}/oas_stage_b_development_homology_hits_reversed_v3.tsv.gz" \
  --hits-sha256 "${HITS_SHA256}" \
  --source-receipt "${DATA}/oas_stage_b_exact_pairs_v2.receipt.json" \
  --output "${OUTPUT_ROOT}/PAIRWISE_DECONTAMINATION_AUDIT_V1.json"

sha256sum "${OUTPUT_ROOT}/PAIRWISE_DECONTAMINATION_AUDIT_V1.json"
