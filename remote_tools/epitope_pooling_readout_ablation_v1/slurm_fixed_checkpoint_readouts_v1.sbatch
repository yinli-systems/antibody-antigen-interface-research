#!/bin/bash
#SBATCH --job-name=pool-readout-v1
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_epitope_pooling_readout_v1/logs/evaluate-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_epitope_pooling_readout_v1/logs/evaluate-%j.err

set -euo pipefail

root=/ssd/scxi253/model_deployment
campaign_root=${root}/results/development/sabdab2_epitope_pooling_readout_v1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=17

mkdir -p "${campaign_root}/results"
exec ${root}/envs/contact/bin/python3.10 \
  ${root}/tools/epitope_pooling_readout_ablation_v1/evaluate_fixed_checkpoint_readouts_v1.py \
  --protocol "${campaign_root}/authority/PROTOCOL_FIXED_CHECKPOINT_EPITOPE_READOUT_V1.json" \
  --output "${campaign_root}/results/FIXED_CHECKPOINT_EPITOPE_READOUT_RESULT_V1.json" \
  --receipt "${campaign_root}/results/FIXED_CHECKPOINT_EPITOPE_READOUT_RECEIPT_V1.json" \
  --workers 4
