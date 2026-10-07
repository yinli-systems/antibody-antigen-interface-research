#!/usr/bin/env bash
#SBATCH -J pinder-ab-s2-tv-exact
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/development/pinder_antibody_s2_v1/train-validation-exact-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/development/pinder_antibody_s2_v1/train-validation-exact-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/remote_tools/abf_paircontact_v1/pinder_scale_v1
readonly OUTPUT=${ROOT}/results/development/pinder_antibody_s2_v1
readonly ASSIGNMENT=${OUTPUT}/train_validation_iteration_001/PINDER_ANTIBODY_S2_TRAIN_VALIDATION_PROVISIONAL_V1.json
readonly ASSIGNMENT_SHA=7e0cccdb90089ccc4d4a443ec73719d11a73f048f2cb581685134d9e847e28d3
readonly ASSIGNMENT_SCIENTIFIC_SHA=b5b7f232e722473f11ca7e1306bee1bccdfd81c14deb34a21168ed82d4609a69
readonly CANONICAL=${OUTPUT}/assets/PINDER_ANTIBODY_S2_CANONICAL_V1.csv
readonly CANONICAL_SHA=624612079838536934c81fa3918486e322c16caed19fbba70b6f8b33adc5fb6c
readonly AUDITOR=${SOURCE}/audit_pinder_antibody_s2_train_validation_exact_v1.py
readonly AUDITOR_SHA=6371dba39313d6ab530c24bea96e32600c9974041b82be7e4f5be80b7fc08784
readonly AUDIT_OUTPUT=${OUTPUT}/train_validation_iteration_001/exact_audit_v1
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10

test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum "${AUDITOR}" | cut -d' ' -f1)" = "${AUDITOR_SHA}"
test "$(sha256sum "${ASSIGNMENT}" | cut -d' ' -f1)" = "${ASSIGNMENT_SHA}"
test "$(sha256sum "${CANONICAL}" | cut -d' ' -f1)" = "${CANONICAL_SHA}"
test "$("${PYTHON}" -c 'import Bio; print(Bio.__version__)')" = "1.87"

mkdir -p "${AUDIT_OUTPUT}"
test ! -e "${AUDIT_OUTPUT}/PINDER_ANTIBODY_S2_TRAIN_VALIDATION_EXACT_AUDIT_RESULT_V1.json"
test ! -e "${AUDIT_OUTPUT}/PINDER_ANTIBODY_S2_TRAIN_VALIDATION_EXACT_AUDIT_RECEIPT_V1.json"

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
