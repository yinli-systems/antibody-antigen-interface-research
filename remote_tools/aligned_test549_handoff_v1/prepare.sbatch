#!/usr/bin/env bash
#SBATCH --job-name=aligned549-prepare
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=/ssd/scxi253/model_deployment/handoffs/framecontact_case_study_20260904/logs/aligned549-prepare-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/handoffs/framecontact_case_study_20260904/logs/aligned549-prepare-%j.err
set -euo pipefail
umask 027
readonly project=/ssd/scxi253/model_deployment
readonly py=${project}/envs/contact/bin/python3.10
readonly tools=${project}/tools/aligned_test549_handoff_v1
readonly bundle=${project}/handoffs/framecontact_case_study_20260904/sabdab2_antigen_fold_test549_predicted_geometry_v2
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test "${SLURM_JOB_PARTITION}" = gpu_4090
"${py}" "${tools}/build_bundle.py" --output "${bundle}"
bundle_sha=$(sha256sum "${bundle}/BUNDLE.json" | awk '{print $1}')
for component in 0 1 2 3 4 5; do
  "${py}" "${bundle}/run_bundle.py" --bundle "${bundle}" --bundle-sha256 "${bundle_sha}" \
    --component "${component}" --mode smoke --output "${bundle}/verification/${component}"
done
"${py}" "${bundle}/finalize_bundle.py" --bundle "${bundle}" --bundle-sha256 "${bundle_sha}" \
  --runs "${bundle}/verification" --mode smoke --output "${bundle}/READY"
