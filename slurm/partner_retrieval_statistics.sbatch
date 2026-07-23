#!/bin/bash
#SBATCH --job-name=partner-retrieval-stats
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
SCRIPT=$DEPLOY_ROOT/jobs/interaction_first/sabdab2_partner_retrieval_statistics.py
PREPARE_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_prepared
EMBEDDING_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_embeddings
SPLIT_MANIFEST=$DEPLOY_ROOT/data/interaction_first/double_remote_split.json
SEED=${RETRIEVAL_SEED:-42}
RUN_NAME=${RETRIEVAL_RUN_NAME:-partner_retrieval_seed$SEED}
RUN_ROOT=$DEPLOY_ROOT/results/sabdab2/partner_retrieval/$RUN_NAME
CHECKPOINT=$RUN_ROOT/checkpoints/best.pt
OUTPUT=$RUN_ROOT/validation/statistical_analysis.json
PERMUTATIONS=${RETRIEVAL_PERMUTATIONS:-10000}
BOOTSTRAP_SAMPLES=${RETRIEVAL_BOOTSTRAP_SAMPLES:-10000}
ANALYSIS_SEED=${RETRIEVAL_ANALYSIS_SEED:-20260723}

mkdir -p "$(dirname "$OUTPUT")" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

/usr/bin/time -v "$PYTHON_BIN" "$SCRIPT" \
  --prepare-dir "$PREPARE_DIR" \
  --embedding-dir "$EMBEDDING_DIR" \
  --split-manifest "$SPLIT_MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --batch-size 64 \
  --bucket-width 128 \
  --permutations "$PERMUTATIONS" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --analysis-seed "$ANALYSIS_SEED" \
  --device cuda
