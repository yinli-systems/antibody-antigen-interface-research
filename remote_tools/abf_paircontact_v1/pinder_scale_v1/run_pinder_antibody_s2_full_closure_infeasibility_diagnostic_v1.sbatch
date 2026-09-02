#!/usr/bin/env bash
#SBATCH -J pinder-ab-s2-proof
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/development/pinder_antibody_s2_v1/full-closure-infeasibility-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/development/pinder_antibody_s2_v1/full-closure-infeasibility-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/remote_tools/abf_paircontact_v1/pinder_scale_v1
readonly OUTPUT=${ROOT}/results/development/pinder_antibody_s2_v1
readonly RELATION=${OUTPUT}/full_four_view_relation_v1
readonly PROOF=${OUTPUT}/full_closure_infeasibility_v1
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10

readonly DIAGNOSTIC=${SOURCE}/diagnose_pinder_antibody_s2_full_closure_infeasibility_v1.py
readonly DIAGNOSTIC_SHA=031dccfdc49b4262da7edd8f5d7a0727a50cc7e5bf5876ca3e3d8fbb181072f5
readonly CANONICAL=${OUTPUT}/assets/PINDER_ANTIBODY_S2_CANONICAL_V1.csv
readonly CANONICAL_SHA=624612079838536934c81fa3918486e322c16caed19fbba70b6f8b33adc5fb6c
readonly RELATION_RESULT=${RELATION}/PINDER_ANTIBODY_S2_FULL_FOUR_VIEW_RELATION_RESULT_V1.json
readonly RELATION_RESULT_SHA=4cc8ec876c6a6e813013d3e01056799b47992c0010bafe4d528f1b6da943df28
readonly RESULT=${PROOF}/PINDER_ANTIBODY_S2_FULL_CLOSURE_INFEASIBILITY_RECEIPT_V1.json

test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum "${DIAGNOSTIC}" | cut -d' ' -f1)" = "${DIAGNOSTIC_SHA}"
test "$(sha256sum "${CANONICAL}" | cut -d' ' -f1)" = "${CANONICAL_SHA}"
test "$(sha256sum "${RELATION_RESULT}" | cut -d' ' -f1)" = "${RELATION_RESULT_SHA}"
test "$(sha256sum "${RELATION}/vh.identity80_coverage80.complete_edges.jsonl" | cut -d' ' -f1)" = "1904fde0f38c7a299bbf68e753b89bcfc4e9d1467c2a4223831c01613d5ca134"
test "$(sha256sum "${RELATION}/vl.identity80_coverage80.complete_edges.jsonl" | cut -d' ' -f1)" = "a85f0755501ca2fdb996a4fb1c184d27b9d1faa2e5364e213b2d1ba7cc8631e0"
test "$(sha256sum "${RELATION}/concatenated_antibody.identity80_coverage80.complete_edges.jsonl" | cut -d' ' -f1)" = "d917f42c8f9b1ae1b6eb227418ab1ce2b25d564bfb92ee6df96fcdd2c500e3ed"
test "$(sha256sum "${RELATION}/antigen.identity80_coverage80.complete_edges.jsonl" | cut -d' ' -f1)" = "90d41556822b39b271a656986a34cf74c992309de80fc313810b0a8b6d52d0b3"

mkdir -p "${PROOF}"
test ! -e "${RESULT}"

"${PYTHON}" "${DIAGNOSTIC}" \
  --source-sha256 "${DIAGNOSTIC_SHA}" \
  --canonical-manifest "${CANONICAL}" \
  --canonical-manifest-sha256 "${CANONICAL_SHA}" \
  --full-relation-result "${RELATION_RESULT}" \
  --full-relation-result-sha256 "${RELATION_RESULT_SHA}" \
  --vh-edges "${RELATION}/vh.identity80_coverage80.complete_edges.jsonl" \
  --vl-edges "${RELATION}/vl.identity80_coverage80.complete_edges.jsonl" \
  --concatenated-antibody-edges "${RELATION}/concatenated_antibody.identity80_coverage80.complete_edges.jsonl" \
  --antigen-edges "${RELATION}/antigen.identity80_coverage80.complete_edges.jsonl" \
  --output "${RESULT}"

sha256sum "${RESULT}"
