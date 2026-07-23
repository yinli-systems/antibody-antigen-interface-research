#!/bin/bash
#SBATCH --job-name=sppider-infer
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%A_%a.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%A_%a.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON=$DEPLOY_ROOT/envs/epiformer/bin/python
PREPARE_DIR=${SPPIDER_PREPARE_DIR:-$DEPLOY_ROOT/data/sppider_seq_sabdab2_prepared}
EMBEDDING_DIR=${SPPIDER_EMBEDDING_DIR:-$DEPLOY_ROOT/data/sppider_seq_sabdab2_embeddings}
ROLE=${SPPIDER_ROLE:?Set SPPIDER_ROLE to paratope or epitope}
RUN_NAME=${SPPIDER_RUN_NAME:?Set SPPIDER_RUN_NAME to a stable result name}
CHECKPOINT=${SPPIDER_CHECKPOINT:?Set SPPIDER_CHECKPOINT to an official or trained checkpoint}
OUTPUT_DIR=$DEPLOY_ROOT/results/sabdab2/sppider_seq/$RUN_NAME/$ROLE/inference
SHARD_COUNT=${SPPIDER_INFER_SHARDS:-4}

mkdir -p "$OUTPUT_DIR" "$DEPLOY_ROOT/cache/tmp" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp

/usr/bin/time -v "$PYTHON" "$DEPLOY_ROOT/jobs/sppider_seq/sabdab2_sppider_seq.py" infer \
  --prepare-dir "$PREPARE_DIR" \
  --embedding-dir "$EMBEDDING_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --role "$ROLE" \
  --checkpoint "$CHECKPOINT" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-count "$SHARD_COUNT" \
  --seed 42 \
  --device cuda \
  --resume
