#!/bin/bash
#SBATCH --job-name=antiberty-dev-embed
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
SCRIPT=$DEPLOY_ROOT/jobs/interaction_first/sabdab2_antiberty_embed.py
CANONICAL_MANIFEST=$DEPLOY_ROOT/data/interaction_first/canonical_manifest.csv
INSTANCES_PARQUET=$DEPLOY_ROOT/data/sppider_seq_sabdab2_prepared/instances.parquet
SPLIT_MANIFEST=$DEPLOY_ROOT/data/interaction_first/double_remote_split.json
OUTPUT_ROOT=${ANTIBERTY_OUTPUT_ROOT:-$DEPLOY_ROOT/data/antiberty_sabdab2_development_embeddings}
BATCH_SIZE=${ANTIBERTY_BATCH_SIZE:-64}

mkdir -p "$OUTPUT_ROOT" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

/usr/bin/time -v "$PYTHON_BIN" "$SCRIPT" \
  --canonical-manifest "$CANONICAL_MANIFEST" \
  --instances-parquet "$INSTANCES_PARQUET" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --splits train validation \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  --resume
