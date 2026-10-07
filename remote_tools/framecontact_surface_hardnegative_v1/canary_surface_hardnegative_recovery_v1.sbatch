#!/usr/bin/env bash
#SBATCH --job-name=fc-surface-hard-r1-canary
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --array=0-1%2
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_surface_hardnegative_recovery_v1/logs/canary-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_surface_hardnegative_recovery_v1/logs/canary-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly output=${root}/results/development/sabdab2_surface_hardnegative_recovery_v1
readonly tools_root=${root}/tools/framecontact_surface_hardnegative_v1
readonly python_bin=${root}/envs/contact/bin/python3.10

case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly arm=baseline; readonly source_root=${root}/immutable_stage_a1_candidates/framecontact_surface_hardnegative_base_v1 ;;
  1) readonly arm=edge_topk2_supervised; readonly source_root=${root}/immutable_stage_a1_candidates/framecontact_surface_hardnegative_topk2_v1 ;;
  *) exit 64 ;;
esac

test "${SLURM_JOB_PARTITION}" = gpu_4090
test -s "${output}/authority/CAMPAIGN_SURFACE_HARDNEGATIVE_V1.json"
mkdir -p "${output}/canary"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${source_root}/src
"${python_bin}" "${tools_root}/test_surface_hardnegative_v1.py" \
  --source-root "${source_root}" --device cuda \
  > "${output}/canary/${arm}.json"
test -s "${output}/canary/${arm}.json"
