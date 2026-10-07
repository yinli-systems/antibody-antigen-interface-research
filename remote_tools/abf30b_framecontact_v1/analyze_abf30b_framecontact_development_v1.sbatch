#!/usr/bin/env bash
#SBATCH --job-name=abf30b-dev-analyze
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/analyze-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/analyze-%j.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/abf30b_framecontact_v1
readonly control=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1/PREDICTED_GEOMETRY_NO_PRIVILEGED_DISTANCE_ANALYSIS_V1.json
readonly python_bin=${root}/envs/contact/bin/python3.10
test "${SLURM_JOB_PARTITION}" = gpu_4090
echo '1382c6fc18624b53f50657c0c0d61408fa03f0acc5aa3a812ffa3f84bd23b78c' "${control}" | sha256sum --check --strict -
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${root}/tools/abf30b_framecontact_v1/analyze_abf30b_framecontact_development_v1.py" \
  --campaign-root "${campaign}" \
  --control-analysis "${control}" \
  --output "${campaign}/ABF30B_FRAMECONTACT_DEVELOPMENT_ANALYSIS_V1.json" \
  --receipt-output "${campaign}/ABF30B_FRAMECONTACT_DEVELOPMENT_ANALYSIS_V1_RECEIPT.json"
