#!/usr/bin/env bash
#SBATCH --job-name=abf30b-sable-analyze
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/sable-analyze-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/sable-analyze-%j.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/abf30b_framecontact_v1
readonly sable=${root}/results/development/sable_v0_2_0_framecontact_external_v1
readonly script=${root}/tools/abf30b_framecontact_v1/analyze_abf30b_sable_measurement_v1.py
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly authority=${campaign}/authority/sable_measurement_v1
readonly binding=${authority}/frozen_binding_v1

test "${SLURM_JOB_PARTITION}" = gpu_4090
echo '60ca023b19055e9cf2ff017945d7d0fb53be8401f18fc1692c99f0c97ce30011' "${script}" | sha256sum --check --strict -
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${script}" \
  --measurement-plan "${authority}/ABF30B_SABLE_MEASUREMENT_PLAN_V1.json" \
  --authorization "${binding}/ABF30B_SABLE_MEASUREMENT_AUTHORIZATION_V1.json" \
  --checkpoint-binding "${binding}/ABF30B_SABLE_CHECKPOINT_BINDING_V1.json" \
  --label-root "${sable}/bound_labels_once_v1" \
  --component-root "${campaign}/sable_measurement_v1/components" \
  --output "${campaign}/sable_measurement_v1/analysis/ABF30B_SABLE_MEASUREMENT_ANALYSIS_V1.json" \
  --receipt-output "${campaign}/sable_measurement_v1/analysis/ABF30B_SABLE_MEASUREMENT_ANALYSIS_V1_RECEIPT.json"
