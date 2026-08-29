#!/usr/bin/env bash
#SBATCH --job-name=fc-threeway-stress
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

: "${FRAMECONTACT_REPOSITORY_ROOT:?missing FRAMECONTACT_REPOSITORY_ROOT}"
base="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
python_bin="/ssd/scxi253/model_deployment/envs/epiformer/bin/python"
output_root="${FRAMECONTACT_STRESS_OUTPUT_ROOT:-${base}/sabdab2/three-way-stress-v1}"
mkdir -p "${output_root}" "${base}/slurm"
exec > >(tee -a "${base}/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out") \
  2> >(tee -a "${base}/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >&2)
cd "${FRAMECONTACT_REPOSITORY_ROOT}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${FRAMECONTACT_REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${python_bin}" scripts/stress_sabdab2_three_way_training_step.py \
  --structure-root "${base}/sabdab2/structure_cohort_v2" \
  --contact-root "${base}/sabdab2/contact_cohort_v2" \
  --embedding-root /ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings \
  --split-manifest "${base}/split_matrix_v3_1/antigen_fold.csv" \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --training-panel "${base}/partner_metadata_antigen_fold_v3/train-reciprocal-31.json" \
  --output "${output_root}/THREE_WAY_STRESS.json" \
  --experiment frozen_frame_ipa \
  --offload-saved-tensors
sha256sum "${output_root}/THREE_WAY_STRESS.json"
