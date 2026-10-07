#!/usr/bin/env bash
#SBATCH --job-name=abf30b-beyond-marginals
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/beyond-marginals-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/beyond-marginals-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/abf30b_framecontact_v1
readonly tool_root=${root}/remote_tools/abf30b_framecontact_v1
readonly authority=${campaign}/authority/beyond_marginals_v1
readonly result_root=${campaign}/results/beyond_marginals_v1
readonly python_bin=${root}/envs/contact/bin/python3.10

case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly seed=17 ;;
  1) readonly seed=29 ;;
  2) readonly seed=43 ;;
  *) exit 64 ;;
esac

readonly output_root=${result_root}/seed-${seed}
test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${output_root}"
mkdir -p "${output_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${python_bin}" "${tool_root}/evaluate_abf30b_beyond_marginals_component_v1.py" \
  --protocol "${authority}/ABF30B_BEYOND_MARGINALS_PROTOCOL_V1.json" \
  --protocol-receipt "${authority}/ABF30B_BEYOND_MARGINALS_PROTOCOL_V1_RECEIPT.json" \
  --seed "${seed}" \
  --output "${output_root}/ABF30B_BEYOND_MARGINALS_COMPONENT_V1.json" \
  --receipt-output "${output_root}/ABF30B_BEYOND_MARGINALS_COMPONENT_V1_RECEIPT.json"
