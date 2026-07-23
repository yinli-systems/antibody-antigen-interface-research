#!/bin/bash
#SBATCH --job-name=sppider-train
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON=$DEPLOY_ROOT/envs/epiformer/bin/python
PREPARE_DIR=${SPPIDER_PREPARE_DIR:-$DEPLOY_ROOT/data/sppider_seq_sabdab2_prepared}
EMBEDDING_DIR=${SPPIDER_EMBEDDING_DIR:-$DEPLOY_ROOT/data/sppider_seq_sabdab2_embeddings}
ROLE=${SPPIDER_ROLE:?Set SPPIDER_ROLE to paratope or epitope}
RUN_NAME=${SPPIDER_RUN_NAME:-trained_seed42}
OUTPUT_DIR=$DEPLOY_ROOT/results/sabdab2/sppider_seq/$RUN_NAME/$ROLE/checkpoints
EPOCHS_THIS_RUN=${SPPIDER_EPOCHS_THIS_RUN:-5}
INITIAL_CHECKPOINT=${SPPIDER_INITIAL_CHECKPOINT:-}
LEARNING_RATE=${SPPIDER_LEARNING_RATE:-1e-4}
WEIGHT_DECAY=${SPPIDER_WEIGHT_DECAY:-0}
SEED=${SPPIDER_SEED:-42}

INITIAL_CHECKPOINT_ARGS=()
if [[ -n "$INITIAL_CHECKPOINT" ]]; then
  INITIAL_CHECKPOINT_ARGS=(--initial-checkpoint "$INITIAL_CHECKPOINT")
fi

mkdir -p "$OUTPUT_DIR" "$DEPLOY_ROOT/cache/tmp" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8

/usr/bin/time -v "$PYTHON" "$DEPLOY_ROOT/jobs/sppider_seq/sabdab2_sppider_seq.py" train \
  --prepare-dir "$PREPARE_DIR" \
  --embedding-dir "$EMBEDDING_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --role "$ROLE" \
  "${INITIAL_CHECKPOINT_ARGS[@]}" \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --max-epochs 50 \
  --max-epochs-this-run "$EPOCHS_THIS_RUN" \
  --patience 10 \
  --seed "$SEED" \
  --device cuda \
  --resume
