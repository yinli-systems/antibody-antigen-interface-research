#!/bin/bash
#SBATCH --job-name=interaction-v2-screen
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
NUM_LAYERS=${INTERACTION_NUM_LAYERS:-1}
MAX_TRAIN=${INTERACTION_MAX_TRAIN:-512}
MAX_VALIDATION=${INTERACTION_MAX_VALIDATION:-216}
MAX_EPOCHS=${INTERACTION_MAX_EPOCHS:-12}
SCORE_CENTERING=${INTERACTION_SCORE_CENTERING:-on}
RUN_NAME=${INTERACTION_RUN_NAME:-v2_screen_l${NUM_LAYERS}_seed${SEED}}
OUTPUT_DIR=$DEPLOY_ROOT/results/sabdab2/interaction_first/$RUN_NAME/checkpoints

SCORE_ARGS=()
if [[ "$SCORE_CENTERING" == "off" ]]; then
  SCORE_ARGS+=(--disable-score-centering)
fi

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
  --d-model 128 \
  --pair-dim 64 \
  --num-heads 8 \
  --num-layers "$NUM_LAYERS" \
  --dropout 0.20 \
  --marginal-pooling logmeanexp \
  --score-pooling logmeanexp \
  --score-temperature 1.0 \
  "${SCORE_ARGS[@]}" \
  --negative-count 4 \
  --negative-directions both \
  --hard-pool-size 64 \
  --partner-rank-weight 0.5 \
  --edge-count-weight 0.0 \
  --epitope-count-weight 0.0 \
  --paratope-count-weight 0.0 \
  --max-positive-weight 100 \
  --selection-metric partner_rank \
  --learning-rate 0.00005 \
  --weight-decay 0.0005 \
  --max-train-instances "$MAX_TRAIN" \
  --max-validation-instances "$MAX_VALIDATION" \
  --max-epochs "$MAX_EPOCHS" \
  --patience 4 \
  --seed "$SEED" \
  --device cuda \
  --no-resume
