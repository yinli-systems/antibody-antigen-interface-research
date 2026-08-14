#!/usr/bin/env bash
#SBATCH --job-name=fc-joint-smoke
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-joint-smoke-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-joint-smoke-%j.err

set -euo pipefail

data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
repository_root="${FRAMECONTACT_REPOSITORY_ROOT:-${data_root}/asep/joint-coupling-source-v1}"
python_bin="/ssd/scxi253/model_deployment/envs/contact/bin/python3.10"
cache_root="${data_root}/asep/development-cache"
protocol="${repository_root}/protocols/framecontact_asep_joint_coupling_dev_v1.json"
protocol_receipt="${repository_root}/protocols/FRAMECONTACT_ASEP_JOINT_COUPLING_DEV_V1_RECEIPT.json"
training_panel="${data_root}/asep/panels/train-reciprocal-31.json"
validation_panel="${data_root}/asep/panels/validation-reciprocal-31.json"
smoke_output="${data_root}/asep/smoke-gates/joint-coupling-a3-seed-17"

if [[ -e "${smoke_output}" ]]; then
  echo "Smoke output path must be absent before execution: ${smoke_output}" >&2
  exit 2
fi

cd "${repository_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${repository_root}/src"
"${python_bin}" scripts/train_framecontact.py \
  --dataset-backend asep \
  --asep-cache-root "${cache_root}" \
  --asep-cache-manifest-sha256 69cc62f0d77858b38508821bf9d56504dd397787a37dad62fbd1a5072a348cc7 \
  --asep-feature-mode official_esm2_35m \
  --protocol "${protocol}" \
  --protocol-receipt "${protocol_receipt}" \
  --protocol-receipt-sha256 51bd60d32e5024f6f6ef18fa4b18dff3767d27911a339ccb75500e1322a69145 \
  --training-partner-panel "${training_panel}" \
  --training-partner-panel-sha256 a0c5f3e8ea1a0edf7cd9a2043b326ee3be7b054c81dff9fbfcb510db6f2969d1 \
  --validation-partner-panel "${validation_panel}" \
  --validation-partner-panel-sha256 10f50aa122568d4ed1a8261637f2e807f2702997a1aa5d9925983d37cc0d83ec \
  --output-root "${smoke_output}" \
  --seed 17 \
  --experiment framecontact_joint_a3_coupled_heads \
  --epochs 120 \
  --early-stopping-patience 20 \
  --batch-size 1 \
  --evaluation-batch-size 1 \
  --workers 4 \
  --d-model 128 \
  --num-heads 8 \
  --num-self-layers 2 \
  --num-rbf 16 \
  --neighbor-k 32 \
  --neighbor-radius 16.0 \
  --sequence-window 32 \
  --dropout 0.1 \
  --attention-kernel reference \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --contact-loss-weight 1.0 \
  --paratope-loss-weight 0.25 \
  --epitope-loss-weight 0.25 \
  --specificity-loss-weight 0.25 \
  --distogram-loss-weight 0.0 \
  --contact-distance-consistency-loss-weight 0.0 \
  --pair-node-consistency-loss-weight 0.25 \
  --triangle-loss-weight 0.0 \
  --far-patch-exclusion-loss-weight 0.0 \
  --edm-loss-weight 0.0 \
  --residue-partner-contrastive-loss-weight 0.0 \
  --specificity-margin 0.05 \
  --score-fraction 0.01 \
  --calibration-max-pairs 2000000 \
  --contact-gate-warmup-fraction 0.05 \
  --contact-gate-ramp-fraction 0.10 \
  --coarse-contact-crop-floor 0.05 \
  --coarse-contact-crop-temperature 1.0 \
  --coarse-contact-crop-strength 1.0 \
  --checkpoint-wrong-partners \
  --offload-saved-tensors \
  --single-batch-smoke-only \
  --device cuda \
  --precision bf16

if [[ -e "${smoke_output}" ]]; then
  echo "Single-batch smoke unexpectedly created an output root" >&2
  exit 2
fi
