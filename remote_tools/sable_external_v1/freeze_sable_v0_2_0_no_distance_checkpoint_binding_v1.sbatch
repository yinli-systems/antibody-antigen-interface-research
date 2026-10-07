#!/usr/bin/env bash
#SBATCH --job-name=sable-bind-nodist
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/bind-no-distance-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/bind-no-distance-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly predgeom=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly sable=${root}/results/development/sable_v0_2_0_framecontact_external_v1
readonly tools_root=${root}/tools/sable_external_v1
readonly script=${tools_root}/freeze_sable_v0_2_0_no_distance_checkpoint_binding_v1.py
readonly script_sha256=9be0b190f80df09fcf0033fe61c53fca4f2302afe3de862fa8b6a1a79c6dafd2
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
echo "${script_sha256}  ${script}" | sha256sum --check --strict -
mkdir -p "${sable}/authority"

"${python_bin}" "${script}" \
  --campaign-root "${predgeom}" \
  --analysis "${predgeom}/PREDICTED_GEOMETRY_NO_PRIVILEGED_DISTANCE_ANALYSIS_V1.json" \
  --analysis-receipt "${predgeom}/PREDICTED_GEOMETRY_NO_PRIVILEGED_DISTANCE_ANALYSIS_V1_RECEIPT.json" \
  --external-plan "${tools_root}/SABLE_V0_2_0_EXTERNAL_EXECUTION_PLAN_V1.json" \
  --output "${sable}/authority/SABLE_V0_2_0_NO_DISTANCE_CHECKPOINT_BINDING_V1.json" \
  --receipt-output "${sable}/authority/SABLE_V0_2_0_NO_DISTANCE_CHECKPOINT_BINDING_V1_RECEIPT.json"
