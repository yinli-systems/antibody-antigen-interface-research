#!/usr/bin/env bash
#SBATCH -J abpairfm-mm-foldseek-3di-audit-r1
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm_pinder_multimodal_v1/foldseek-audit-recovery-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm_pinder_multimodal_v1/foldseek-audit-recovery-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/remote_tools/abpairfm_multimodal_pretraining_v1
readonly OUTPUT=${ROOT}/results/pretraining/abpairfm_pinder_multimodal_v1
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly FOLDSEEK=${ROOT}/data/abpairfm_3b_v1/tools/foldseek-10-941cd33/bin/foldseek
readonly FOLDSEEK_SHA=4db94a7825d409e416643772860ddf7cfe328d582679b47e6dffda7df4d41ea0
readonly AUDITOR=${SOURCE}/audit_foldseek_3di_v1.py
readonly AUDITOR_SHA=c2d9f63ee1fe88d9c8eb890987496114ac0e1e014ae0e452defed088fa5e40f3
readonly CACHE_RESULT=${OUTPUT}/PINDER_PREDICTED_STRUCTURE_CACHE_RESULT_V1.json
readonly STRUCTURE_MANIFEST=${OUTPUT}/predicted_structure_cache_v1/PREDICTED_STRUCTURE_CACHE_MANIFEST_V1.tsv
readonly RAW_OUTPUT=${OUTPUT}/.foldseek-3di-1534865.tsv
readonly TOKEN_MANIFEST=${OUTPUT}/PINDER_PREDICTED_STRUCTURE_3DI_V1.tsv.gz
readonly RESULT=${OUTPUT}/PINDER_PREDICTED_STRUCTURE_3DI_RESULT_V1.json

test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum "${FOLDSEEK}" | cut -d' ' -f1)" = "${FOLDSEEK_SHA}"
test "$(sha256sum "${AUDITOR}" | cut -d' ' -f1)" = "${AUDITOR_SHA}"
test -s "${RAW_OUTPUT}"
test ! -e "${TOKEN_MANIFEST}"
test ! -e "${RESULT}"

readonly STRUCTURE_MANIFEST_SHA=$(sha256sum "${STRUCTURE_MANIFEST}" | cut -d' ' -f1)
"${PYTHON}" - "${CACHE_RESULT}" "${STRUCTURE_MANIFEST_SHA}" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="ascii"))
if result.get("status") != "PASS_AUTHENTICATED_PREDICTED_STRUCTURE_CACHE":
    raise SystemExit("predicted-structure cache did not pass")
if result.get("outputs", {}).get("manifest_sha256") != sys.argv[2]:
    raise SystemExit("predicted-structure manifest SHA differs from cache result")
PY

"${PYTHON}" "${AUDITOR}" \
  --structure-manifest "${STRUCTURE_MANIFEST}" \
  --structure-manifest-sha256 "${STRUCTURE_MANIFEST_SHA}" \
  --foldseek-binary "${FOLDSEEK}" \
  --foldseek-binary-sha256 "${FOLDSEEK_SHA}" \
  --foldseek-output "${RAW_OUTPUT}" \
  --output-tokens "${TOKEN_MANIFEST}" \
  --output-result "${RESULT}"
sha256sum "${TOKEN_MANIFEST}" "${RESULT}"
