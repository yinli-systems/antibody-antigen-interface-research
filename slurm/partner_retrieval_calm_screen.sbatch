#!/bin/bash
#SBATCH --job-name=partner-retrieval-calm-screen
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
SCRIPT=$DEPLOY_ROOT/jobs/interaction_first/sabdab2_partner_retrieval.py
PREPARE_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_prepared
EMBEDDING_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_embeddings
SPLIT_MANIFEST=$DEPLOY_ROOT/data/interaction_first/double_remote_split.json
SEED=${RETRIEVAL_SEED:-42}
RUN_NAME=${RETRIEVAL_RUN_NAME:-partner_retrieval_calm_screen_seed$SEED}
OUTPUT_DIR=$DEPLOY_ROOT/results/sabdab2/partner_retrieval/$RUN_NAME/checkpoints

mkdir -p "$OUTPUT_DIR" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

/usr/bin/time -v "$PYTHON_BIN" "$SCRIPT" train \
  --prepare-dir "$PREPARE_DIR" \
  --embedding-dir "$EMBEDDING_DIR" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --embedding-dim 128 \
  --attention-hidden-dim 128 \
  --projection-hidden-dim 256 \
  --dropout 0.15 \
  --initial-temperature 0.20 \
  --minimum-temperature 0.02 \
  --maximum-temperature 0.50 \
  --batch-size 512 \
  --validation-batch-size 64 \
  --bucket-width 128 \
  --learning-rate 0.001 \
  --minimum-learning-rate 0.00001 \
  --weight-decay 0.20 \
  --warmup-fraction 0.10 \
  --gradient-clip 1.0 \
  --max-epochs 100 \
  --patience 15 \
  --min-delta 0.0001 \
  --seed "$SEED" \
  --device cuda \
  --resume
