#!/usr/bin/env bash
#SBATCH -J pinder-ab-s2-tv-i3
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/development/pinder_antibody_s2_v1/train-validation-iteration-003-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/development/pinder_antibody_s2_v1/train-validation-iteration-003-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/remote_tools/abf_paircontact_v1/pinder_scale_v1
readonly OUTPUT=${ROOT}/results/development/pinder_antibody_s2_v1
readonly PREVIOUS=${OUTPUT}/train_validation_iteration_002
readonly PREVIOUS_AUDIT=${PREVIOUS}/exact_audit_v1
readonly CURRENT=${OUTPUT}/train_validation_iteration_003
readonly CURRENT_AUDIT=${CURRENT}/exact_audit_v1
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10

readonly BUILDER=${SOURCE}/build_pinder_antibody_s2_train_validation_iteration_v2.py
readonly BUILDER_SHA=d0b3425be70f9031c0a35269eb791f90de186ea27411fbdac932bf289b632de8
readonly PROTOCOL=${SOURCE}/PINDER_ANTIBODY_S2_TRAIN_VALIDATION_ITERATION_003_PROTOCOL_V2.json
readonly PROTOCOL_SHA=848268095e283de491db91e4e0f1478cb629f7b9b510ff7f47ad890e969f4077
readonly AUDITOR=${SOURCE}/audit_pinder_antibody_s2_train_validation_exact_v1.py
readonly AUDITOR_SHA=6371dba39313d6ab530c24bea96e32600c9974041b82be7e4f5be80b7fc08784

readonly CANONICAL=${OUTPUT}/assets/PINDER_ANTIBODY_S2_CANONICAL_V1.csv
readonly CANONICAL_SHA=624612079838536934c81fa3918486e322c16caed19fbba70b6f8b33adc5fb6c
readonly PREVIOUS_ASSIGNMENT=${PREVIOUS}/PINDER_ANTIBODY_S2_TRAIN_VALIDATION_PROVISIONAL_V2.json
readonly PREVIOUS_ASSIGNMENT_SHA=142e671dba2ee9d761d54a6a4e869344ae2f5f5f75607f4574167b9d132b2db4
readonly PREVIOUS_AUDIT_RESULT=${PREVIOUS_AUDIT}/PINDER_ANTIBODY_S2_TRAIN_VALIDATION_EXACT_AUDIT_RESULT_V1.json
readonly PREVIOUS_AUDIT_RESULT_SHA=0a51edc1dce9caf67ec8d644268c4d8b72aa9489a943f4afa239218f9127948a

readonly VH_EDGES=${PREVIOUS_AUDIT}/vh.identity80_coverage80.crossing_edges.jsonl
readonly VL_EDGES=${PREVIOUS_AUDIT}/vl.identity80_coverage80.crossing_edges.jsonl
readonly CONCAT_EDGES=${PREVIOUS_AUDIT}/concatenated_antibody.identity80_coverage80.crossing_edges.jsonl
readonly ANTIGEN_EDGES=${PREVIOUS_AUDIT}/antigen.identity80_coverage80.crossing_edges.jsonl
readonly ASSIGNMENT=${CURRENT}/PINDER_ANTIBODY_S2_TRAIN_VALIDATION_PROVISIONAL_V3.json

test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum "${BUILDER}" | cut -d' ' -f1)" = "${BUILDER_SHA}"
test "$(sha256sum "${PROTOCOL}" | cut -d' ' -f1)" = "${PROTOCOL_SHA}"
test "$(sha256sum "${AUDITOR}" | cut -d' ' -f1)" = "${AUDITOR_SHA}"
test "$(sha256sum "${CANONICAL}" | cut -d' ' -f1)" = "${CANONICAL_SHA}"
test "$(sha256sum "${PREVIOUS_ASSIGNMENT}" | cut -d' ' -f1)" = "${PREVIOUS_ASSIGNMENT_SHA}"
test "$(sha256sum "${PREVIOUS_AUDIT_RESULT}" | cut -d' ' -f1)" = "${PREVIOUS_AUDIT_RESULT_SHA}"
test "$(sha256sum "${VH_EDGES}" | cut -d' ' -f1)" = "14dd23ed185c209065d95b8c7980036365293cf7e572ca5d35045af4a4a75ae5"
test "$(sha256sum "${VL_EDGES}" | cut -d' ' -f1)" = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
test "$(sha256sum "${CONCAT_EDGES}" | cut -d' ' -f1)" = "2d73cd61936ba71c52705b790fd0b8b5ef7f2ba7b284e1afea2dcdf913b010da"
test "$(sha256sum "${ANTIGEN_EDGES}" | cut -d' ' -f1)" = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
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
  --previous-assignment "${PREVIOUS_ASSIGNMENT}" \
  --previous-audit-result "${PREVIOUS_AUDIT_RESULT}" \
  --vh-edges "${VH_EDGES}" \
  --vl-edges "${VL_EDGES}" \
  --concatenated-antibody-edges "${CONCAT_EDGES}" \
  --antigen-edges "${ANTIGEN_EDGES}" \
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
