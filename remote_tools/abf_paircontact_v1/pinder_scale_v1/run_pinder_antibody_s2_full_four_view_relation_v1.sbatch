#!/usr/bin/env bash
#SBATCH -J pinder-ab-s2-fullrel
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/development/pinder_antibody_s2_v1/full-four-view-relation-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/development/pinder_antibody_s2_v1/full-four-view-relation-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/remote_tools/abf_paircontact_v1/pinder_scale_v1
readonly OUTPUT=${ROOT}/results/development/pinder_antibody_s2_v1
readonly RELATION_OUTPUT=${OUTPUT}/full_four_view_relation_v1
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly AUDITOR=${SOURCE}/audit_pinder_antibody_s2_full_four_view_relation_v1.py
readonly AUDITOR_SHA=a3e00b8a279f2528a2aba87bdd7a6578f498cde23c4b4b44f8a00e8816356308
readonly CANONICAL=${OUTPUT}/assets/PINDER_ANTIBODY_S2_CANONICAL_V1.csv
readonly CANONICAL_SHA=624612079838536934c81fa3918486e322c16caed19fbba70b6f8b33adc5fb6c

test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum "${AUDITOR}" | cut -d' ' -f1)" = "${AUDITOR_SHA}"
test "$(sha256sum "${CANONICAL}" | cut -d' ' -f1)" = "${CANONICAL_SHA}"
test "$("${PYTHON}" -c 'import Bio; print(Bio.__version__)')" = "1.87"

mkdir -p "${RELATION_OUTPUT}"
test ! -e "${RELATION_OUTPUT}/PINDER_ANTIBODY_S2_FULL_FOUR_VIEW_RELATION_RESULT_V1.json"
test ! -e "${RELATION_OUTPUT}/PINDER_ANTIBODY_S2_FULL_FOUR_VIEW_RELATION_RECEIPT_V1.json"

"${PYTHON}" "${AUDITOR}" \
  --canonical-manifest "${CANONICAL}" \
  --canonical-manifest-sha256 "${CANONICAL_SHA}" \
  --auditor-source-sha256 "${AUDITOR_SHA}" \
  --output-dir "${RELATION_OUTPUT}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --chunk-size 256
