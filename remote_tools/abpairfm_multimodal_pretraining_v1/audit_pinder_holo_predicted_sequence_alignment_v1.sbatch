#!/usr/bin/env bash
#SBATCH -J abpairfm-mm-sequence-alignment
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm_pinder_multimodal_v1/alignment-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm_pinder_multimodal_v1/alignment-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/remote_tools/abpairfm_multimodal_pretraining_v1
readonly OUTPUT=${ROOT}/results/pretraining/abpairfm_pinder_multimodal_v1
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly AUDITOR=${SOURCE}/audit_pinder_holo_predicted_sequence_alignment_v1.py
readonly AUDITOR_SHA=acd493242942841197ae03548beae65718b11a9fa7309c89021dd9b3071d6a90
readonly PROTOCOL=${SOURCE}/PROTOCOL_PINDER_MULTIMODAL_ALIGNMENT_AMENDMENT_V2.json
readonly PROTOCOL_SHA=ab493517f153dd4ae91d05d2fff18e19725cd3d115083b757f4758cc6a47cc53
readonly INVENTORY=${OUTPUT}/PINDER_MULTIMODAL_BFULL_INVENTORY_V1.tsv.gz
readonly INVENTORY_SHA=914fee292921061b68822f6bde7cd49c0e13802b99c05e0b73558a04277e9a11
readonly CACHE_RESULT=${OUTPUT}/PINDER_PREDICTED_STRUCTURE_CACHE_RESULT_V1.json
readonly STRUCTURE_MANIFEST=${OUTPUT}/predicted_structure_cache_v1/PREDICTED_STRUCTURE_CACHE_MANIFEST_V1.tsv
readonly ALIGNMENT_MANIFEST=${OUTPUT}/PINDER_HOLO_TO_PREDICTED_ALIGNMENT_V1.tsv.gz
readonly ALIGNMENT_RESULT=${OUTPUT}/PINDER_HOLO_TO_PREDICTED_ALIGNMENT_RESULT_V1.json

test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum "${AUDITOR}" | cut -d' ' -f1)" = "${AUDITOR_SHA}"
test "$(sha256sum "${PROTOCOL}" | cut -d' ' -f1)" = "${PROTOCOL_SHA}"
test "$(sha256sum "${INVENTORY}" | cut -d' ' -f1)" = "${INVENTORY_SHA}"
test ! -e "${ALIGNMENT_MANIFEST}"
test ! -e "${ALIGNMENT_RESULT}"

readonly STRUCTURE_MANIFEST_SHA=$(sha256sum "${STRUCTURE_MANIFEST}" | cut -d' ' -f1)
"${PYTHON}" - "${CACHE_RESULT}" "${STRUCTURE_MANIFEST_SHA}" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="ascii"))
if result.get("status") != "PASS_AUTHENTICATED_PREDICTED_STRUCTURE_CACHE":
    raise SystemExit("predicted-structure cache did not pass")
if result.get("outputs", {}).get("manifest_sha256") != sys.argv[2]:
    raise SystemExit("predicted-structure manifest SHA differs from cache result")
if result.get("test_access") != {
    "bound_coordinates_used_as_model_input": False,
    "pinder_test_structures_read": False,
    "pinder_validation_structures_read": False,
}:
    raise SystemExit("predicted-structure cache access boundary differs")
PY

"${PYTHON}" "${AUDITOR}" \
  --inventory "${INVENTORY}" \
  --inventory-sha256 "${INVENTORY_SHA}" \
  --structure-manifest "${STRUCTURE_MANIFEST}" \
  --structure-manifest-sha256 "${STRUCTURE_MANIFEST_SHA}" \
  --protocol-amendment "${PROTOCOL}" \
  --protocol-amendment-sha256 "${PROTOCOL_SHA}" \
  --minimum-exact-mapped-residues 8 \
  --minimum-exact-coverage 0.5 \
  --minimum-both-usable-systems 250000 \
  --minimum-median-coverage 0.9 \
  --output-manifest "${ALIGNMENT_MANIFEST}" \
  --output-result "${ALIGNMENT_RESULT}"
sha256sum "${ALIGNMENT_MANIFEST}" "${ALIGNMENT_RESULT}"
