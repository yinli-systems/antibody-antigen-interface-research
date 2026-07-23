#!/bin/bash
#SBATCH --job-name=sppider-eval
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON=$DEPLOY_ROOT/envs/epiformer/bin/python
PREPARE_DIR=${SPPIDER_PREPARE_DIR:-$DEPLOY_ROOT/data/sppider_seq_sabdab2_prepared}
EMBEDDING_DIR=${SPPIDER_EMBEDDING_DIR:-$DEPLOY_ROOT/data/sppider_seq_sabdab2_embeddings}
ROLE=${SPPIDER_ROLE:?Set SPPIDER_ROLE to paratope or epitope}
SPLIT=${SPPIDER_SPLIT:-validation}
CONTEXT_MODE=${SPPIDER_CONTEXT_MODE:-matched}
CONTEXT_SEED=${SPPIDER_CONTEXT_SEED:-20260722}
SEED=${SPPIDER_SEED:-42}
CHECKPOINT=${SPPIDER_CHECKPOINT:?Set SPPIDER_CHECKPOINT to the evaluated checkpoint}
OUTPUT=${SPPIDER_EVAL_OUTPUT:?Set SPPIDER_EVAL_OUTPUT to a JSON path}
PREDICTIONS=${SPPIDER_EVAL_PREDICTIONS:-${OUTPUT%.json}.residue_predictions.parquet}

mkdir -p "$(dirname "$OUTPUT")" "$DEPLOY_ROOT/cache/tmp" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8

/usr/bin/time -v "$PYTHON" "$DEPLOY_ROOT/jobs/sppider_seq/sabdab2_sppider_seq.py" evaluate \
  --prepare-dir "$PREPARE_DIR" \
  --embedding-dir "$EMBEDDING_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --residue-predictions "$PREDICTIONS" \
  --role "$ROLE" \
  --split "$SPLIT" \
  --context-mode "$CONTEXT_MODE" \
  --context-seed "$CONTEXT_SEED" \
  --threshold 0.50 \
  --seed "$SEED" \
  --device cuda
