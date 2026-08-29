#!/usr/bin/env bash
#SBATCH --job-name=fc-pair-dump
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-pair-dump-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-pair-dump-%A_%a.err

set -euo pipefail

if [[ "${SLURM_JOB_PARTITION:-}" != "gpu_4090" ]]; then
  echo "refusing partition ${SLURM_JOB_PARTITION:-unset}; gpu_4090 only" >&2
  exit 2
fi

seeds=(17 42 73 101 137)
task_id="${SLURM_ARRAY_TASK_ID}"
if (( task_id < 5 )); then
  arm="candidate"
  seed="${seeds[task_id]}"
  repository_root="/ssd/scxi253/model_deployment/data/framecontact_20260730/asep/local-decoy-speed-source-20260820"
  run_root="/ssd/scxi253/model_deployment/data/framecontact_20260730/sabdab2/development-runs/local-interface-decoy-gain-speed-v1/framecontact/seed-${seed}"
else
  arm="control"
  seed="${seeds[task_id - 5]}"
  repository_root="/ssd/scxi253/model_deployment/data/framecontact_20260730/sabdab_source_v2"
  run_root="/ssd/scxi253/model_deployment/data/framecontact_20260730/sabdab2/development-runs/antigen-fold-v3_1-v5/framecontact/seed-${seed}"
fi

data_root="/ssd/scxi253/model_deployment/data/framecontact_20260730"
source_root="${data_root}/asep/pair-dump-source-20260821-v1"
output_root="${data_root}/asep/development-results/local-interface-decoy-gain-speed-v1-pair-dumps-20260821-v1/${arm}/seed-${seed}"
python_bin="/ssd/scxi253/model_deployment/envs/contact/bin/python3.10"

for required in \
  "${source_root}/export_framecontact_pair_predictions.py" \
  "${repository_root}/scripts/train_framecontact.py" \
  "${run_root}/FRAMECONTACT_DEVELOPMENT_RESULT.json" \
  "${run_root}/FRAMECONTACT_SELECTED.pt" \
  "${run_root}/RUN_RECEIPT.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done
if [[ -e "${output_root}" ]]; then
  echo "refusing to overwrite output: ${output_root}" >&2
  exit 2
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${python_bin}" "${source_root}/export_framecontact_pair_predictions.py" \
  --repository-root "${repository_root}" \
  --run-root "${run_root}" \
  --arm "${arm}" \
  --structure-cache-root "${data_root}/sabdab2/structure_cohort_v2" \
  --contact-cache-root "${data_root}/sabdab2/contact_cohort_v2" \
  --embedding-root /ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings \
  --split-manifest "${data_root}/split_matrix_v3_1/antigen_fold.csv" \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --output-root "${output_root}" \
  --workers 4
