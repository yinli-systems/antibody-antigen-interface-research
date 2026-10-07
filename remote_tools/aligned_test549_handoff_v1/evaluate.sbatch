#!/usr/bin/env bash
#SBATCH --job-name=aligned549-eval
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/ssd/scxi253/model_deployment/handoffs/framecontact_case_study_20260904/logs/aligned549-eval-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/handoffs/framecontact_case_study_20260904/logs/aligned549-eval-%A_%a.err
set -euo pipefail
umask 027
readonly project=/ssd/scxi253/model_deployment
readonly py=${project}/envs/contact/bin/python3.10
readonly bundle=${project}/handoffs/framecontact_case_study_20260904/sabdab2_antigen_fold_test549_predicted_geometry_v2
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test "${SLURM_JOB_PARTITION}" = gpu_4090
test -f "${bundle}/READY/RECEIPT.json"
bundle_sha=$(sha256sum "${bundle}/BUNDLE.json" | awk '{print $1}')
exec "${py}" "${bundle}/run_bundle.py" --bundle "${bundle}" --bundle-sha256 "${bundle_sha}" \
  --component "${SLURM_ARRAY_TASK_ID}" --mode full --output "${bundle}/evaluation/${SLURM_ARRAY_TASK_ID}"
