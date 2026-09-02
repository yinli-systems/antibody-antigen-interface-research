#!/usr/bin/env bash
#SBATCH --job-name=analyze-bilateral-topk-v1
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_bilateral_topk_readout_v2/logs/analyze-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_bilateral_topk_readout_v2/logs/analyze-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_bilateral_topk_readout_v2
readonly recovery=${campaign}/analysis_recovery_authority_v1
readonly output=${campaign}/analysis_recovery_v1
test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${output}"
"${root}/envs/contact/bin/python3.10" \
  "${root}/tools/bilateral_topk_readout_v1/analyze_bilateral_topk_readout_v1.py" \
  --campaign-root "${campaign}" \
  --recovery-protocol "${recovery}/BILATERAL_TOPK_ANALYSIS_RECOVERY_PROTOCOL_V1.json" \
  --recovery-receipt "${recovery}/BILATERAL_TOPK_ANALYSIS_RECOVERY_PROTOCOL_V1_RECEIPT.json" \
  --output-root "${output}"
