#!/usr/bin/env bash
#SBATCH -J oas-paired-devclean-materialize-v2
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/logs/materialize-v2-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/logs/materialize-v2-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/oas_paired_continuation_v1_20260825_v9/source
readonly DECONTAM_SOURCE=/ssd/scxi253/model_deployment/oas_paired_continuation_v1_20260825_v6/source
readonly DATA=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/oas_paired_20260821
readonly AUDIT_ROOT=/ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1
readonly OUTPUT_ROOT=${DATA}/oas_paired_development_clean_composite_v1
readonly SCRIPT_SHA256=f19546326a4400df4da78622bbf3d171f4d196da318670895f779f940e4f5919
readonly QUALITY_SCRIPT_SHA256=7e3929661a4a45612560d123933ea1bd657cf2f678fa7e850df7b9c4c8a7bc9f
readonly PAIRED_SHA256=fcfb69f016681a8d801db213e49f6ff61252f6882d28eb56753cac3dbbdac58a
readonly PROTOCOL_SHA256=35c19666fd3f63fc3bb8e30940a7b2e96cf2d2fc5a16925b684fec4330f5ca06

readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden materialization allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
test -x "${PYTHON}"
test ! -e "${OUTPUT_ROOT}"
[[ "$(sha256sum "${PROJECT}/materialize_paired_corpus.py" | awk '{print $1}')" == "${SCRIPT_SHA256}" ]]
[[ "$(sha256sum "${PROJECT}/audit_paired_corpus_quality.py" | awk '{print $1}')" == "${QUALITY_SCRIPT_SHA256}" ]]
[[ "$(sha256sum "${PROJECT}/paired_pretraining.py" | awk '{print $1}')" == "${PAIRED_SHA256}" ]]
[[ "$(sha256sum "${PROJECT}/PROTOCOL_PAIRED_DEVELOPMENT_CLEAN_MATERIALIZATION_V1.json" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]]

cd "${PROJECT}"
"${PYTHON}" materialize_paired_corpus.py \
  --pairs "${DATA}/oas_stage_b_exact_pairs_v2.tsv.gz" \
  --source-receipt "${DATA}/oas_stage_b_exact_pairs_v2.receipt.json" \
  --quality-result "${AUDIT_ROOT}/corpus-quality-v1/OAS_PAIRED_CORPUS_QUALITY_AUDIT_V1.json" \
  --decontamination-result "${AUDIT_ROOT}/pairwise-decontamination-v4/PAIRWISE_DECONTAMINATION_AUDIT_V3.json" \
  --decontamination-script "${DECONTAM_SOURCE}/audit_pairwise_decontamination.py" \
  --decontamination-protocol "${DECONTAM_SOURCE}/PROTOCOL_PAIRWISE_DECONTAMINATION_AUDIT_V3.json" \
  --exclusions "${AUDIT_ROOT}/pairwise-decontamination-v4/OAS_PAIRED_DEVELOPMENT_EXCLUSIONS_COMPOSITE_V1.txt.gz" \
  --protocol "${PROJECT}/PROTOCOL_PAIRED_DEVELOPMENT_CLEAN_MATERIALIZATION_V1.json" \
  --output-root "${OUTPUT_ROOT}"

sha256sum "${OUTPUT_ROOT}/MANIFEST.json"
