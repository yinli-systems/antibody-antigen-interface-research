#!/usr/bin/env bash
#SBATCH -J pinder-ab-s2-separator
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/development/pinder_antibody_s2_v1/train-validation-vertex-separator-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/development/pinder_antibody_s2_v1/train-validation-vertex-separator-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/remote_tools/abf_paircontact_v1/pinder_scale_v1
readonly OUTPUT=${ROOT}/results/development/pinder_antibody_s2_v1
readonly RELATION=${OUTPUT}/full_four_view_relation_v1
readonly PROOF=${OUTPUT}/full_closure_infeasibility_v1
readonly CURRENT=${OUTPUT}/train_validation_vertex_separator_v1
readonly CURRENT_AUDIT=${CURRENT}/exact_audit_v1
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10

readonly BUILDER=${SOURCE}/build_pinder_antibody_s2_train_validation_vertex_separator_v1.py
readonly BUILDER_SHA=3865fafcc85a9dddba1d6e43b68e39c99c300119c2df417749b1f5172b2bfeb6
readonly PROTOCOL=${SOURCE}/PINDER_ANTIBODY_S2_TRAIN_VALIDATION_VERTEX_SEPARATOR_PROTOCOL_V1.json
readonly PROTOCOL_SHA=0130faedce2d28845b2cf606d6ecfef90d9b79bc08cc0f76dc1ed7ec2a1af5be
readonly AUDITOR=${SOURCE}/audit_pinder_antibody_s2_train_validation_exact_v1.py
readonly AUDITOR_SHA=6371dba39313d6ab530c24bea96e32600c9974041b82be7e4f5be80b7fc08784

readonly CANONICAL=${OUTPUT}/assets/PINDER_ANTIBODY_S2_CANONICAL_V1.csv
readonly CANONICAL_SHA=624612079838536934c81fa3918486e322c16caed19fbba70b6f8b33adc5fb6c
readonly RELATION_RESULT=${RELATION}/PINDER_ANTIBODY_S2_FULL_FOUR_VIEW_RELATION_RESULT_V1.json
readonly RELATION_RESULT_SHA=4cc8ec876c6a6e813013d3e01056799b47992c0010bafe4d528f1b6da943df28
readonly INFEASIBILITY=${PROOF}/PINDER_ANTIBODY_S2_FULL_CLOSURE_INFEASIBILITY_RECEIPT_V1.json
readonly INFEASIBILITY_SHA=fd2178a00540beda315f5e129134bb429adfa17966764c3b9e5a9db48bb9abf9
readonly ASSIGNMENT=${CURRENT}/PINDER_ANTIBODY_S2_TRAIN_VALIDATION_PROVISIONAL_VERTEX_SEPARATOR_V1.json

test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum "${BUILDER}" | cut -d' ' -f1)" = "${BUILDER_SHA}"
test "$(sha256sum "${PROTOCOL}" | cut -d' ' -f1)" = "${PROTOCOL_SHA}"
test "$(sha256sum "${AUDITOR}" | cut -d' ' -f1)" = "${AUDITOR_SHA}"
test "$(sha256sum "${CANONICAL}" | cut -d' ' -f1)" = "${CANONICAL_SHA}"
test "$(sha256sum "${RELATION_RESULT}" | cut -d' ' -f1)" = "${RELATION_RESULT_SHA}"
test "$(sha256sum "${INFEASIBILITY}" | cut -d' ' -f1)" = "${INFEASIBILITY_SHA}"
test "$(sha256sum "${RELATION}/vh.identity80_coverage80.complete_edges.jsonl" | cut -d' ' -f1)" = "1904fde0f38c7a299bbf68e753b89bcfc4e9d1467c2a4223831c01613d5ca134"
test "$(sha256sum "${RELATION}/vl.identity80_coverage80.complete_edges.jsonl" | cut -d' ' -f1)" = "a85f0755501ca2fdb996a4fb1c184d27b9d1faa2e5364e213b2d1ba7cc8631e0"
test "$(sha256sum "${RELATION}/concatenated_antibody.identity80_coverage80.complete_edges.jsonl" | cut -d' ' -f1)" = "d917f42c8f9b1ae1b6eb227418ab1ce2b25d564bfb92ee6df96fcdd2c500e3ed"
test "$(sha256sum "${RELATION}/antigen.identity80_coverage80.complete_edges.jsonl" | cut -d' ' -f1)" = "90d41556822b39b271a656986a34cf74c992309de80fc313810b0a8b6d52d0b3"
test "$("${PYTHON}" -c 'import Bio; print(Bio.__version__)')" = "1.87"

mkdir -p "${CURRENT}" "${CURRENT_AUDIT}"
test ! -e "${ASSIGNMENT}"
test ! -e "${CURRENT_AUDIT}/PINDER_ANTIBODY_S2_TRAIN_VALIDATION_EXACT_AUDIT_RESULT_V1.json"
test ! -e "${CURRENT_AUDIT}/PINDER_ANTIBODY_S2_TRAIN_VALIDATION_EXACT_AUDIT_RECEIPT_V1.json"

"${PYTHON}" "${BUILDER}" \
  --protocol "${PROTOCOL}" \
  --protocol-sha256 "${PROTOCOL_SHA}" \
  --source-sha256 "${BUILDER_SHA}" \
  --canonical-manifest "${CANONICAL}" \
  --full-relation-result "${RELATION_RESULT}" \
  --infeasibility-receipt "${INFEASIBILITY}" \
  --vh-edges "${RELATION}/vh.identity80_coverage80.complete_edges.jsonl" \
  --vl-edges "${RELATION}/vl.identity80_coverage80.complete_edges.jsonl" \
  --concatenated-antibody-edges "${RELATION}/concatenated_antibody.identity80_coverage80.complete_edges.jsonl" \
  --antigen-edges "${RELATION}/antigen.identity80_coverage80.complete_edges.jsonl" \
  --output "${ASSIGNMENT}"

readonly ASSIGNMENT_SHA=$(sha256sum "${ASSIGNMENT}" | cut -d' ' -f1)
readonly ASSIGNMENT_SCIENTIFIC_SHA=$("${PYTHON}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["scientific_payload_sha256"])' \
  "${ASSIGNMENT}")

"${PYTHON}" "${AUDITOR}" \
  --assignment "${ASSIGNMENT}" \
  --assignment-sha256 "${ASSIGNMENT_SHA}" \
  --assignment-scientific-sha256 "${ASSIGNMENT_SCIENTIFIC_SHA}" \
  --canonical-manifest "${CANONICAL}" \
  --canonical-manifest-sha256 "${CANONICAL_SHA}" \
  --auditor-source-sha256 "${AUDITOR_SHA}" \
  --output-dir "${CURRENT_AUDIT}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --chunk-size 256
