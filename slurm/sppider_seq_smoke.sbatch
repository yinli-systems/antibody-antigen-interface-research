#!/bin/bash
#SBATCH --job-name=sppider-smoke
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON=$DEPLOY_ROOT/envs/epiformer/bin/python
SCRIPT=$DEPLOY_ROOT/jobs/sppider_seq/sabdab2_sppider_seq.py
PREPARE_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_prepared
EMBEDDING_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_embeddings
MODEL_DIR=$DEPLOY_ROOT/models/sppider_seq
SMOKE_ROOT=$DEPLOY_ROOT/results/sabdab2/sppider_seq/smoke

mkdir -p "$SMOKE_ROOT" "$DEPLOY_ROOT/cache/tmp" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for ROLE in paratope epitope; do
  "$PYTHON" "$SCRIPT" infer \
    --prepare-dir "$PREPARE_DIR" \
    --embedding-dir "$EMBEDDING_DIR" \
    --output-dir "$SMOKE_ROOT/official_receptor/$ROLE" \
    --role "$ROLE" \
    --checkpoint "$MODEL_DIR/crossattn_rec_run27_best.pt" \
    --shard-index 0 \
    --shard-count 1 \
    --max-instances 1 \
    --device cuda \
    --seed 42 \
    --resume

  "$PYTHON" "$SCRIPT" train \
    --prepare-dir "$PREPARE_DIR" \
    --embedding-dir "$EMBEDDING_DIR" \
    --output-dir "$SMOKE_ROOT/trained/$ROLE" \
    --role "$ROLE" \
    --learning-rate 1e-4 \
    --weight-decay 0 \
    --max-epochs 1 \
    --max-epochs-this-run 1 \
    --patience 10 \
    --max-instances 2 \
    --max-validation-instances 2 \
    --log-every 1 \
    --device cuda \
    --seed 42 \
    --no-resume
done

sha256sum \
  "$SMOKE_ROOT/trained/paratope/best.pt" \
  "$SMOKE_ROOT/trained/epitope/best.pt" \
  > "$SMOKE_ROOT/checkpoint_sha256.txt"
