#!/bin/bash
#SBATCH --job-name=epi-sabdab2-smoke
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
BENCHMARK_ROOT=$DEPLOY_ROOT/data/sabdab2-baseline-outsourcing
OUTPUT_DIR=$DEPLOY_ROOT/results/sabdab2/epiformer_smoke

mkdir -p "$OUTPUT_DIR" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export TORCH_HOME=$DEPLOY_ROOT/cache/torch
export HF_HOME=$DEPLOY_ROOT/cache/huggingface

/usr/bin/time -v "$DEPLOY_ROOT/envs/epiformer/bin/python" \
  "$DEPLOY_ROOT/jobs/sabdab2_epiformer_smoke.py" \
  --epiformer-code "$DEPLOY_ROOT/src/epiformer/code" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --checkpoint "$DEPLOY_ROOT/data/results/epiformer/checkpoints/formal_epi_group_seed42/epiformer_best.pt" \
  --output-dir "$OUTPUT_DIR"
