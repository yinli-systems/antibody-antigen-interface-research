#!/usr/bin/env bash
#SBATCH --job-name=abf30b-sable-bind
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/sable-bind-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/sable-bind-%j.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/abf30b_framecontact_v1
readonly sable=${root}/results/development/sable_v0_2_0_framecontact_external_v1
readonly script=${root}/tools/abf30b_framecontact_v1/bind_authorize_abf30b_sable_measurement_v1.py
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
echo '56f24323a621f7fd43b3627f80766b6b442ffad9755224ec08ae23c575033781' "${script}" | sha256sum --check --strict -
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${script}" \
  --campaign-root "${campaign}" \
  --sable-root "${sable}" \
  --output-root "${campaign}/authority/sable_measurement_v1/frozen_binding_v1"
