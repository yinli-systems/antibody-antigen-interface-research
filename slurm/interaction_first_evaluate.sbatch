#!/bin/bash
#SBATCH --job-name=interaction-eval
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
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
SPLIT=${INTERACTION_EVAL_SPLIT:-validation}
CHECKPOINT=$DEPLOY_ROOT/results/sabdab2/interaction_first/$RUN_NAME/checkpoints/best.pt
OUTPUT_DIR=$DEPLOY_ROOT/results/sabdab2/interaction_first/$RUN_NAME/$SPLIT

mkdir -p "$OUTPUT_DIR" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8

/usr/bin/time -v "$PYTHON_BIN" "$SCRIPT" evaluate \
  --prepare-dir "$PREPARE_DIR" \
  --contact-dir "$INTERACTION_DATA/contacts" \
  --embedding-dir "$EMBEDDING_DIR" \
  --split-manifest "$INTERACTION_DATA/double_remote_split.json" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --negative-count 9 \
  --hard-pool-size 64 \
  --bootstrap-iterations 2000 \
  --seed "$SEED" \
  --device cuda
