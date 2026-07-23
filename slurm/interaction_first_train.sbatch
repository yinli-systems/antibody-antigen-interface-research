#!/bin/bash
#SBATCH --job-name=interaction-train
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
SCRIPT=$DEPLOY_ROOT/jobs/interaction_first/sabdab2_interaction_first.py
PREPARE_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_prepared
EMBEDDING_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_embeddings
INTERACTION_DATA=$DEPLOY_ROOT/data/interaction_first
SEED=${INTERACTION_SEED:-42}
RUN_NAME=${INTERACTION_RUN_NAME:-sequence_pair_seed$SEED}
OUTPUT_DIR=$DEPLOY_ROOT/results/sabdab2/interaction_first/$RUN_NAME/checkpoints

mkdir -p "$OUTPUT_DIR" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8

/usr/bin/time -v "$PYTHON_BIN" "$SCRIPT" train \
  --prepare-dir "$PREPARE_DIR" \
  --contact-dir "$INTERACTION_DATA/contacts" \
  --embedding-dir "$EMBEDDING_DIR" \
  --split-manifest "$INTERACTION_DATA/double_remote_split.json" \
  --output-dir "$OUTPUT_DIR" \
  --d-model 256 \
  --pair-dim 128 \
  --num-heads 8 \
  --num-layers 3 \
  --negative-count 2 \
  --hard-pool-size 32 \
  --partner-rank-weight 0.25 \
  --learning-rate 0.000065 \
  --weight-decay 0.0001 \
  --max-epochs 50 \
  --patience 10 \
  --seed "$SEED" \
  --device cuda \
  --resume
