#!/usr/bin/env bash
#SBATCH --job-name=bilateral-topk-readout-v1
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-5%6
#SBATCH --no-requeue
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_bilateral_topk_readout_v2/logs/eval-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_bilateral_topk_readout_v2/logs/eval-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly tool_root=${root}/tools/bilateral_topk_readout_v1
readonly campaign=${root}/results/development/sabdab2_bilateral_topk_readout_v2
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly protocol=${campaign}/authority/BILATERAL_TOPK_READOUT_PROTOCOL_V1.json
readonly receipt=${campaign}/authority/BILATERAL_TOPK_READOUT_PROTOCOL_V1_RECEIPT.json

case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly task=paratope; readonly seed=17 ;;
  1) readonly task=paratope; readonly seed=29 ;;
  2) readonly task=paratope; readonly seed=43 ;;
  3) readonly task=epitope; readonly seed=17 ;;
  4) readonly task=epitope; readonly seed=29 ;;
  5) readonly task=epitope; readonly seed=43 ;;
  *) exit 64 ;;
esac

readonly output=${campaign}/runs/${task}/seed-${seed}
test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${output}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${python_bin}" "${tool_root}/evaluate_bilateral_topk_readout_v1.py" \
  --protocol "${protocol}" --protocol-receipt "${receipt}" \
  --task "${task}" --seed "${seed}" --output-root "${output}" --workers 4
