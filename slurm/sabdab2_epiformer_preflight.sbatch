#!/bin/bash
#SBATCH --job-name=epi-sab-preflight
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
BENCHMARK_ROOT=$DEPLOY_ROOT/data/sabdab2-baseline-outsourcing
OUTPUT_ROOT=$DEPLOY_ROOT/results/sabdab2/epiformer_formal
PYTHON=$DEPLOY_ROOT/envs/epiformer/bin/python

mkdir -p "$OUTPUT_ROOT" "$DEPLOY_ROOT/cache/tmp" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp

/usr/bin/time -v "$PYTHON" "$DEPLOY_ROOT/jobs/sabdab2_epiformer_adapter.py" preflight \
  --benchmark-root "$BENCHMARK_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --split test
