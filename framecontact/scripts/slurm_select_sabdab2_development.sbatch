#!/usr/bin/env bash
#SBATCH --job-name=fc-sab-select
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=fc-sab-select-%j.out
#SBATCH --error=fc-sab-select-%j.err

set -euo pipefail

: "${FRAMECONTACT_REPOSITORY_ROOT:?missing FRAMECONTACT_REPOSITORY_ROOT}"
: "${FRAMECONTACT_ENSEMBLE_ROOT:?missing FRAMECONTACT_ENSEMBLE_ROOT}"
: "${FRAMECONTACT_SELECTION_ROOT:?missing FRAMECONTACT_SELECTION_ROOT}"
: "${FRAMECONTACT_PYTHON_BIN:?missing FRAMECONTACT_PYTHON_BIN}"

if [[ -f "${FRAMECONTACT_SELECTION_ROOT}/SELECTION_RECEIPT.json" && -f "${FRAMECONTACT_SELECTION_ROOT}/SABDAB2_DEVELOPMENT_SELECTION.json" ]]; then
  exit 0
fi
if [[ -e "${FRAMECONTACT_SELECTION_ROOT}" ]]; then
  echo "Partial selection output exists without a complete receipt" >&2
  exit 2
fi

cd "${FRAMECONTACT_REPOSITORY_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${FRAMECONTACT_REPOSITORY_ROOT}/src"
"${FRAMECONTACT_PYTHON_BIN}" -B scripts/freeze_framecontact_development_selection.py \
  --ensemble-roots \
    "${FRAMECONTACT_ENSEMBLE_ROOT}/sequence_dense" \
    "${FRAMECONTACT_ENSEMBLE_ROOT}/spatial_topology" \
    "${FRAMECONTACT_ENSEMBLE_ROOT}/distance" \
    "${FRAMECONTACT_ENSEMBLE_ROOT}/distance_direction" \
    "${FRAMECONTACT_ENSEMBLE_ROOT}/full_pose" \
    "${FRAMECONTACT_ENSEMBLE_ROOT}/frozen_frame_ipa" \
    "${FRAMECONTACT_ENSEMBLE_ROOT}/framecontact" \
    "${FRAMECONTACT_ENSEMBLE_ROOT}/framecontact_interleaved" \
  --protocol "${FRAMECONTACT_REPOSITORY_ROOT}/protocols/framecontact_primary_v5.json" \
  --protocol-receipt "${FRAMECONTACT_REPOSITORY_ROOT}/protocols/FRAMECONTACT_PROTOCOL_V5_RECEIPT.json" \
  --protocol-receipt-sha256 395b8d86648abb348b41ef2513319c86ca20e3e235eb070dd81fb4516a760537 \
  --output-root "${FRAMECONTACT_SELECTION_ROOT}"
