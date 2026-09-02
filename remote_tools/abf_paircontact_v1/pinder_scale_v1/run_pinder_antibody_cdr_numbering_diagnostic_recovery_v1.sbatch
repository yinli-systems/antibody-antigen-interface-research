#!/usr/bin/env bash
#SBATCH -J pinder-cdr-numbering-diagnostic-recovery
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/development/pinder_antibody_cdr_antigen_s40_v1/diagnostic-recovery-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/development/pinder_antibody_cdr_antigen_s40_v1/diagnostic-recovery-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/remote_tools/abf_paircontact_v1/pinder_scale_v1
readonly OUTPUT=${ROOT}/results/development/pinder_antibody_cdr_antigen_s40_v1
readonly ASSETS=${ROOT}/results/development/pinder_antibody_s2_v1/assets
readonly PYTHON=${ROOT}/envs/oas-imgt-anarcii-2.0.8-torch-2.7.1-cpu-py3.12/bin/python
readonly ANARCII_WHEEL=${ROOT}/artifacts/oas-imgt-wheelhouse-anarcii-2.0.8-torch-2.7.1-cpu/anarcii-2.0.8-py3-none-any.whl
readonly PROTOCOL=${SOURCE}/PINDER_ANTIBODY_CDR_ANTIGEN_S40_PROTOCOL_V1.json
readonly DIAGNOSTIC=${SOURCE}/diagnose_pinder_antibody_cdr_numbering_v1.py
readonly CANONICAL=${ASSETS}/PINDER_ANTIBODY_S2_CANONICAL_V1.csv
readonly RESULT=${OUTPUT}/PINDER_ANTIBODY_CDR_NUMBERING_DIAGNOSTIC_RECOVERY_V1.json

test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum "${PROTOCOL}" | cut -d' ' -f1)" = 48f46a64e0eccb5ec6b49798685b98939f45c3b598e99c66d6f2d4f39b79c98f
test "$(sha256sum "${DIAGNOSTIC}" | cut -d' ' -f1)" = 685d1215496e68b8be44a9dbaff37fe1728a899c07c2061772f777667af4fdba
test "$(sha256sum "${SOURCE}/materialize_pinder_antibody_cdr_views_v1.py" | cut -d' ' -f1)" = fcfd77894f6c3eb7a8102bf7bd42c4d9a7e198bb71d4b953efd6e6bee0ade939
test "$(sha256sum "${CANONICAL}" | cut -d' ' -f1)" = 624612079838536934c81fa3918486e322c16caed19fbba70b6f8b33adc5fb6c
test ! -e "${RESULT}"
"${PYTHON}" "${DIAGNOSTIC}" \
  --protocol "${PROTOCOL}" \
  --canonical-manifest "${CANONICAL}" \
  --anarcii-wheel "${ANARCII_WHEEL}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --output "${RESULT}"
sha256sum "${RESULT}"
