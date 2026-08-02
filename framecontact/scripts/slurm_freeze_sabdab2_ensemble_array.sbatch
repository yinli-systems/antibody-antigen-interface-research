#!/usr/bin/env bash
#SBATCH --job-name=fc-sab-ensemble
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --array=0-7%4
#SBATCH --output=fc-sab-ensemble-%A_%a.out
#SBATCH --error=fc-sab-ensemble-%A_%a.err

set -euo pipefail

: "${FRAMECONTACT_REPOSITORY_ROOT:?missing FRAMECONTACT_REPOSITORY_ROOT}"
: "${FRAMECONTACT_RUN_ROOT:?missing FRAMECONTACT_RUN_ROOT}"
: "${FRAMECONTACT_ENSEMBLE_ROOT:?missing FRAMECONTACT_ENSEMBLE_ROOT}"
: "${FRAMECONTACT_PYTHON_BIN:?missing FRAMECONTACT_PYTHON_BIN}"

experiments=(
  sequence_dense
  spatial_topology
  distance
  distance_direction
  full_pose
  frozen_frame_ipa
  framecontact
  framecontact_interleaved
)
task_index="${SLURM_ARRAY_TASK_ID}"
if (( task_index < 0 || task_index >= 8 )); then
  echo "SLURM_ARRAY_TASK_ID is outside the frozen 0-7 ensemble matrix" >&2
  exit 2
fi
experiment="${experiments[${task_index}]}"
data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
output_root="${FRAMECONTACT_ENSEMBLE_ROOT}/${experiment}"
if [[ -f "${output_root}/ENSEMBLE_RECEIPT.json" && -f "${output_root}/FRAMECONTACT_DEVELOPMENT_ENSEMBLE.json" ]]; then
  exit 0
fi
if [[ -e "${output_root}" ]]; then
  echo "Partial ensemble output exists without a complete receipt: ${output_root}" >&2
  exit 2
fi

cd "${FRAMECONTACT_REPOSITORY_ROOT}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${FRAMECONTACT_REPOSITORY_ROOT}/src"
"${FRAMECONTACT_PYTHON_BIN}" -B scripts/freeze_framecontact_development_ensemble.py \
  --run-roots \
    "${FRAMECONTACT_RUN_ROOT}/${experiment}/seed-17" \
    "${FRAMECONTACT_RUN_ROOT}/${experiment}/seed-42" \
    "${FRAMECONTACT_RUN_ROOT}/${experiment}/seed-73" \
    "${FRAMECONTACT_RUN_ROOT}/${experiment}/seed-101" \
    "${FRAMECONTACT_RUN_ROOT}/${experiment}/seed-137" \
  --structure-cache-root "${data_root}/sabdab2/structure_cohort_v2" \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --contact-cache-root "${data_root}/sabdab2/contact_cohort_v2" \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --embedding-root /ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest "${data_root}/split_matrix_v3_1/antigen_fold.csv" \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --validation-partner-panel "${data_root}/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json" \
  --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \
  --output-root "${output_root}" \
  --evaluation-batch-size 1 \
  --workers 4 \
  --device cuda \
  --precision bf16
