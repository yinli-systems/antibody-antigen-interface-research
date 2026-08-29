#!/usr/bin/env bash
#SBATCH -J oas-paired-quality-audit-v1
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/logs/corpus-quality-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/logs/corpus-quality-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/oas_paired_continuation_v1_20260825_v5/source
readonly DATA=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/oas_paired_20260821
readonly RUN_ROOT=/ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/corpus-quality-v1
readonly SCRIPT_SHA256=7e3929661a4a45612560d123933ea1bd657cf2f678fa7e850df7b9c4c8a7bc9f
readonly PAIRED_SHA256=fcfb69f016681a8d801db213e49f6ff61252f6882d28eb56753cac3dbbdac58a
readonly PROTOCOL_SHA256=8fb84faff75f41907f39bd2adfc0bf7b36d558b75c9e35efa722479cc206def5
readonly PAIRS_SHA256=87c907d6e6edaab7b9b22957e5c789b6b80906fb45adca3da7ac59d8758d93ff

readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden audit allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
test -x "${PYTHON}"
test ! -e "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"

[[ "$(sha256sum "${PROJECT}/audit_paired_corpus_quality.py" | awk '{print $1}')" == "${SCRIPT_SHA256}" ]]
[[ "$(sha256sum "${PROJECT}/paired_pretraining.py" | awk '{print $1}')" == "${PAIRED_SHA256}" ]]
[[ "$(sha256sum "${PROJECT}/PROTOCOL_PAIRED_CORPUS_QUALITY_AUDIT_V1.json" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]]

cd "${PROJECT}"
"${PYTHON}" audit_paired_corpus_quality.py \
  --pairs "${DATA}/oas_stage_b_exact_pairs_v2.tsv.gz" \
  --pairs-sha256 "${PAIRS_SHA256}" \
  --source-receipt "${DATA}/oas_stage_b_exact_pairs_v2.receipt.json" \
  --work-database "${RUN_ROOT}/row_hashes.work.sqlite" \
  --output "${RUN_ROOT}/OAS_PAIRED_CORPUS_QUALITY_AUDIT_V1.json"

sha256sum "${RUN_ROOT}/OAS_PAIRED_CORPUS_QUALITY_AUDIT_V1.json"
