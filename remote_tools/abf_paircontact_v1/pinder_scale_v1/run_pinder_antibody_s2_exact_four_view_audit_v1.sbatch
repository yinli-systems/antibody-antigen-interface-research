#!/usr/bin/env bash
#SBATCH -J pinder-ab-s2-exact
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/development/pinder_antibody_s2_v1/exact-four-view-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/development/pinder_antibody_s2_v1/exact-four-view-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/remote_tools/abf_paircontact_v1/pinder_scale_v1
readonly OUTPUT=${ROOT}/results/development/pinder_antibody_s2_v1
readonly AUDIT_OUTPUT=${OUTPUT}/exact_four_view_audit_v1
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly AUDITOR=${SOURCE}/audit_pinder_antibody_s2_exact_four_view_v1.py
readonly AUDITOR_SHA=3184283f03a77bdc2098877b4a9bdcb5271467c5d361aa27a70ede0ecb6984a9
readonly ASSIGNMENT=${OUTPUT}/provisional/PINDER_ANTIBODY_PROVISIONAL_S2_FOLDS_V1.json
readonly ASSIGNMENT_SHA=cf367c6ae54f2b5cd4eede1df09a6e3a907d61c56ea510e5a6d83495a4ae5ce2
readonly ASSIGNMENT_SCIENTIFIC_SHA=96481cbca99c10037a4b1bc821f87e77b874e5c97982b7bedc7d849a6d5eac58
readonly CANONICAL=${OUTPUT}/assets/PINDER_ANTIBODY_S2_CANONICAL_V1.csv
readonly CANONICAL_SHA=624612079838536934c81fa3918486e322c16caed19fbba70b6f8b33adc5fb6c

test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum "${AUDITOR}" | cut -d' ' -f1)" = "${AUDITOR_SHA}"
test "$(sha256sum "${ASSIGNMENT}" | cut -d' ' -f1)" = "${ASSIGNMENT_SHA}"
test "$(sha256sum "${CANONICAL}" | cut -d' ' -f1)" = "${CANONICAL_SHA}"
test "$("${PYTHON}" -c 'import Bio; print(Bio.__version__)')" = "1.87"

mkdir -p "${AUDIT_OUTPUT}"
test ! -e "${AUDIT_OUTPUT}/PINDER_ANTIBODY_S2_EXACT_FOUR_VIEW_AUDIT_RESULT_V1.json"
test ! -e "${AUDIT_OUTPUT}/PINDER_ANTIBODY_S2_EXACT_FOUR_VIEW_AUDIT_RECEIPT_V1.json"

"${PYTHON}" "${AUDITOR}" \
  --assignment "${ASSIGNMENT}" \
  --assignment-sha256 "${ASSIGNMENT_SHA}" \
  --assignment-scientific-sha256 "${ASSIGNMENT_SCIENTIFIC_SHA}" \
  --canonical-manifest "${CANONICAL}" \
  --canonical-manifest-sha256 "${CANONICAL_SHA}" \
  --auditor-source-sha256 "${AUDITOR_SHA}" \
  --output-dir "${AUDIT_OUTPUT}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --chunk-size 256
