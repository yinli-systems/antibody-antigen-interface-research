#!/usr/bin/env bash
#SBATCH -J oas-pair-decontam-audit-v3
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/logs/pairwise-decontamination-v3-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/logs/pairwise-decontamination-v3-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/oas_paired_continuation_v1_20260825_v6/source
readonly DATA=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/oas_paired_20260821
readonly RUN_ROOT=/ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/pairwise-decontamination-v3
readonly SCRIPT_SHA256=e105a5b59951dd23a005fc5f9560d212cb248717000df88696902aa4e6978f61
readonly PROTOCOL_SHA256=9f8b355ae8a7b55e6dbdad5fa4ddd9cf47a9cc78b549394ce85667e50bc0fc68
readonly HITS_SHA256=edf4c952061ffef3e48b24ddb61a30d303a9cca26a3fa47c5ddcb6ce8f8f8338

readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden audit allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
test -x "${PYTHON}"
test ! -e "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"

[[ "$(sha256sum "${PROJECT}/audit_pairwise_decontamination.py" | awk '{print $1}')" == "${SCRIPT_SHA256}" ]]
[[ "$(sha256sum "${PROJECT}/PROTOCOL_PAIRWISE_DECONTAMINATION_AUDIT_V3.json" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]]

"${PYTHON}" "${PROJECT}/audit_pairwise_decontamination.py" \
  --hits "${DATA}/oas_stage_b_development_homology_hits_reversed_v3.tsv.gz" \
  --hits-sha256 "${HITS_SHA256}" \
  --source-receipt "${DATA}/oas_stage_b_exact_pairs_v2.receipt.json" \
  --work-database "${RUN_ROOT}/pairwise_decontamination.work.sqlite" \
  --exclusions-output "${RUN_ROOT}/OAS_PAIRED_DEVELOPMENT_EXCLUSIONS_COMPOSITE_V1.txt.gz" \
  --output "${RUN_ROOT}/PAIRWISE_DECONTAMINATION_AUDIT_V3.json"

sha256sum "${RUN_ROOT}/PAIRWISE_DECONTAMINATION_AUDIT_V3.json" \
  "${RUN_ROOT}/OAS_PAIRED_DEVELOPMENT_EXCLUSIONS_COMPOSITE_V1.txt.gz"
