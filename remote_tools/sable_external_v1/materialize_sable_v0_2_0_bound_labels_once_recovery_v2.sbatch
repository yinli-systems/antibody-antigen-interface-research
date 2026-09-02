#!/usr/bin/env bash
#SBATCH --job-name=sable-label-recovery-v2
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/materialize-bound-labels-recovery-v2-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/materialize-bound-labels-recovery-v2-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly sable=${root}/results/development/sable_v0_2_0_framecontact_external_v1
readonly data=${root}/data/external/sable_v0_2_0_label_free
readonly tools_root=${root}/tools/sable_external_v1
readonly script=${tools_root}/materialize_sable_v0_2_0_bound_labels_once_recovery_v2.py
readonly script_sha256=d00406f71889367e0d77bd7a856b6e33b80eea879cfdfe2746c77d5322e9bf8c
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
echo "${script_sha256}  ${script}" | sha256sum --check --strict -

"${python_bin}" "${script}" \
  --authorization "${sable}/authority/SABLE_V0_2_0_BOUND_LABEL_OPENING_AUTHORIZATION_V1.json" \
  --authorization-receipt "${sable}/authority/SABLE_V0_2_0_BOUND_LABEL_OPENING_AUTHORIZATION_V1_RECEIPT.json" \
  --external-plan "${tools_root}/SABLE_V0_2_0_EXTERNAL_EXECUTION_PLAN_V1.json" \
  --metadata "${data}/sable_v0.2.0_metadata.csv" \
  --input-plan "${sable}/structure_prediction/af3_label_free_v1/AF3_LABEL_FREE_INPUT_PLAN.json" \
  --download-root "${sable}/bound_structures_once_v1" \
  --output-root "${sable}/bound_labels_once_v1"
