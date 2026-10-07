#!/usr/bin/env bash
#SBATCH -J abpairfm-mm-foldseek-3di
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm_pinder_multimodal_v1/foldseek-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm_pinder_multimodal_v1/foldseek-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/remote_tools/abpairfm_multimodal_pretraining_v1
readonly OUTPUT=${ROOT}/results/pretraining/abpairfm_pinder_multimodal_v1
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly FOLDSEEK=${ROOT}/data/abpairfm_3b_v1/tools/foldseek-10-941cd33/bin/foldseek
readonly FOLDSEEK_SHA=4db94a7825d409e416643772860ddf7cfe328d582679b47e6dffda7df4d41ea0
readonly AUDITOR=${SOURCE}/audit_foldseek_3di_v1.py
readonly AUDITOR_SHA=43626859010022348d6c023336752f7aaa5821e6459053d839a020d5ed3c67c4
readonly CACHE_RESULT=${OUTPUT}/PINDER_PREDICTED_STRUCTURE_CACHE_RESULT_V1.json
readonly CACHE_ROOT=${OUTPUT}/predicted_structure_cache_v1
readonly STRUCTURE_MANIFEST=${CACHE_ROOT}/PREDICTED_STRUCTURE_CACHE_MANIFEST_V1.tsv
readonly RAW_OUTPUT=${OUTPUT}/.foldseek-3di-${SLURM_JOB_ID}.tsv
readonly TOKEN_MANIFEST=${OUTPUT}/PINDER_PREDICTED_STRUCTURE_3DI_V1.tsv.gz
readonly RESULT=${OUTPUT}/PINDER_PREDICTED_STRUCTURE_3DI_RESULT_V1.json

test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum "${FOLDSEEK}" | cut -d' ' -f1)" = "${FOLDSEEK_SHA}"
test "$(sha256sum "${AUDITOR}" | cut -d' ' -f1)" = "${AUDITOR_SHA}"
test "$("${FOLDSEEK}" version)" = "941cd33ff0771cd2e3f144e3293e22a2b87e9fda"
test ! -e "${RAW_OUTPUT}"
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

"${FOLDSEEK}" structureto3didescriptor \
  -v 0 \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --chain-name-mode 1 \
  "${CACHE_ROOT}/pdb" \
  "${RAW_OUTPUT}"

"${PYTHON}" "${AUDITOR}" \
  --structure-manifest "${STRUCTURE_MANIFEST}" \
  --structure-manifest-sha256 "${STRUCTURE_MANIFEST_SHA}" \
  --foldseek-binary "${FOLDSEEK}" \
  --foldseek-binary-sha256 "${FOLDSEEK_SHA}" \
  --foldseek-output "${RAW_OUTPUT}" \
  --output-tokens "${TOKEN_MANIFEST}" \
  --output-result "${RESULT}"
rm "${RAW_OUTPUT}"
sha256sum "${TOKEN_MANIFEST}" "${RESULT}"
