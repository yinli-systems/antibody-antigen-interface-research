#!/usr/bin/env bash
#SBATCH --job-name=aligned549-summary
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/ssd/scxi253/model_deployment/handoffs/framecontact_case_study_20260904/logs/aligned549-summary-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/handoffs/framecontact_case_study_20260904/logs/aligned549-summary-%j.err
set -euo pipefail
umask 027
readonly project=/ssd/scxi253/model_deployment
readonly bundle=${project}/handoffs/framecontact_case_study_20260904/sabdab2_antigen_fold_test549_predicted_geometry_v2
export PYTHONDONTWRITEBYTECODE=1
bundle_sha=$(sha256sum "${bundle}/BUNDLE.json" | awk '{print $1}')
exec "${project}/envs/contact/bin/python3.10" "${bundle}/finalize_bundle.py" \
  --bundle "${bundle}" --bundle-sha256 "${bundle_sha}" --runs "${bundle}/evaluation" \
  --mode full --output "${bundle}/FULL_SUMMARY"
