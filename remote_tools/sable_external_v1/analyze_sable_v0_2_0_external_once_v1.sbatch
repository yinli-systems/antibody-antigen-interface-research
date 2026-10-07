#!/usr/bin/env bash
#SBATCH --job-name=sable-analyze-once
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/analyze-once-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/analyze-once-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly sable=${root}/results/development/sable_v0_2_0_framecontact_external_v1
readonly tools_root=${root}/tools/sable_external_v1
readonly script=${tools_root}/analyze_sable_v0_2_0_external_once_v1.py
readonly script_sha256=418bf9a78f094eba859771b92ffb7b13e48801bcea397f7c3cf44aed3f7f2159
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
echo "${script_sha256}  ${script}" | sha256sum --check --strict -
mkdir -p "${sable}/analysis_external_once_v1"

"${python_bin}" "${script}" \
  --external-plan "${tools_root}/SABLE_V0_2_0_EXTERNAL_EXECUTION_PLAN_V1.json" \
  --authorization "${sable}/authority/SABLE_V0_2_0_BOUND_LABEL_OPENING_AUTHORIZATION_V1.json" \
  --checkpoint-binding "${sable}/authority/SABLE_V0_2_0_NO_DISTANCE_CHECKPOINT_BINDING_V1.json" \
  --label-root "${sable}/bound_labels_once_v1" \
  --component-root "${sable}/runs_external_once_v1" \
  --output "${sable}/analysis_external_once_v1/SABLE_V0_2_0_EXTERNAL_ANALYSIS_V1.json" \
  --receipt-output "${sable}/analysis_external_once_v1/SABLE_V0_2_0_EXTERNAL_ANALYSIS_V1_RECEIPT.json"
