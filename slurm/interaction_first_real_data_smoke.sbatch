#!/bin/bash
#SBATCH --job-name=interaction-real-smoke
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
SCRIPT=$DEPLOY_ROOT/jobs/interaction_first/sabdab2_interaction_first.py
PREPARE_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_prepared
EMBEDDING_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_embeddings
INTERACTION_DATA=$DEPLOY_ROOT/data/interaction_first
OUTPUT_DIR=$DEPLOY_ROOT/results/sabdab2/interaction_first/v2_real_data_smoke/checkpoints

mkdir -p "$OUTPUT_DIR" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$PYTHON_BIN" "$SCRIPT" train \
  --prepare-dir "$PREPARE_DIR" \
  --contact-dir "$INTERACTION_DATA/contacts" \
  --embedding-dir "$EMBEDDING_DIR" \
  --split-manifest "$INTERACTION_DATA/double_remote_split.json" \
  --output-dir "$OUTPUT_DIR" \
  --d-model 64 \
  --pair-dim 32 \
  --num-heads 4 \
  --num-layers 1 \
  --negative-count 2 \
  --hard-pool-size 4 \
  --max-train-instances 8 \
  --max-validation-instances 4 \
  --max-epochs 1 \
  --patience 1 \
  --log-every 1 \
  --seed 42 \
  --device cuda \
  --no-resume
