#!/bin/bash
#SBATCH --job-name=sppider-embed
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%A_%a.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%A_%a.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON=$DEPLOY_ROOT/envs/epiformer/bin/python
BENCHMARK_ROOT=$DEPLOY_ROOT/data/sabdab2-baseline-outsourcing
PREPARE_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_prepared
EMBEDDING_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_embeddings
ESM_MODEL_DIR=$DEPLOY_ROOT/models/sppider_seq/esm2_t33_650M_UR50D
SHARD_COUNT=${SPPIDER_EMBED_SHARDS:-4}

mkdir -p "$EMBEDDING_DIR" "$DEPLOY_ROOT/cache/tmp" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export TORCH_HOME=$DEPLOY_ROOT/cache/torch
export HF_HOME=$DEPLOY_ROOT/cache/huggingface

/usr/bin/time -v "$PYTHON" "$DEPLOY_ROOT/jobs/sppider_seq/sabdab2_sppider_seq.py" embed \
  --benchmark-root "$BENCHMARK_ROOT" \
  --prepare-dir "$PREPARE_DIR" \
  --embedding-dir "$EMBEDDING_DIR" \
  --esm-model-name "$ESM_MODEL_DIR" \
  --hf-cache "$HF_HOME" \
  --local-files-only \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-count "$SHARD_COUNT" \
  --device cuda \
  --seed 42 \
  --resume
