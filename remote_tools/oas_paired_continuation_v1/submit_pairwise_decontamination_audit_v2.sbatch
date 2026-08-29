#!/usr/bin/env bash
#SBATCH -J oas-pair-decontam-audit-v2
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/logs/pairwise-decontamination-v2-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/logs/pairwise-decontamination-v2-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/oas_paired_continuation_v1_20260825_v4/source
readonly DATA=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/oas_paired_20260821
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1
readonly SCRIPT_SHA256=28d4db91a78e18001e69cee731fc1ca7b8731f2cc4fe1f7729c66b16ee6c78e6
readonly PROTOCOL_SHA256=89afee8d73326f77a335621daa44526aef24f5e2864292ca5c17864008294d7e
readonly HITS_SHA256=edf4c952061ffef3e48b24ddb61a30d303a9cca26a3fa47c5ddcb6ce8f8f8338

readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden audit allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
test -x "${PYTHON}"
mkdir -p "${OUTPUT_ROOT}/logs"
test ! -e "${OUTPUT_ROOT}/PAIRWISE_DECONTAMINATION_AUDIT_V2.json"
test ! -e "${OUTPUT_ROOT}/pairwise_decontamination_v2.work.sqlite"

[[ "$(sha256sum "${PROJECT}/audit_pairwise_decontamination.py" | awk '{print $1}')" == "${SCRIPT_SHA256}" ]]
[[ "$(sha256sum "${PROJECT}/PROTOCOL_PAIRWISE_DECONTAMINATION_AUDIT_V2.json" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]]

"${PYTHON}" "${PROJECT}/audit_pairwise_decontamination.py" \
  --hits "${DATA}/oas_stage_b_development_homology_hits_reversed_v3.tsv.gz" \
  --hits-sha256 "${HITS_SHA256}" \
  --source-receipt "${DATA}/oas_stage_b_exact_pairs_v2.receipt.json" \
  --work-database "${OUTPUT_ROOT}/pairwise_decontamination_v2.work.sqlite" \
  --output "${OUTPUT_ROOT}/PAIRWISE_DECONTAMINATION_AUDIT_V2.json"

sha256sum "${OUTPUT_ROOT}/PAIRWISE_DECONTAMINATION_AUDIT_V2.json"
