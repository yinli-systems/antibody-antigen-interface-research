#!/bin/bash
#SBATCH --job-name=epi-sab-assemble
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
BENCHMARK_ROOT=$DEPLOY_ROOT/data/sabdab2-baseline-outsourcing
OUTPUT_ROOT=$DEPLOY_ROOT/results/sabdab2/epiformer_formal
CHECKPOINT=$DEPLOY_ROOT/data/results/epiformer/checkpoints/formal_epi_group_seed42/epiformer_best.pt
PYTHON=$DEPLOY_ROOT/envs/epiformer/bin/python

mkdir -p "$OUTPUT_ROOT" "$DEPLOY_ROOT/cache/tmp" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp

/usr/bin/time -v "$PYTHON" "$DEPLOY_ROOT/jobs/sabdab2_epiformer_adapter.py" assemble \
  --benchmark-root "$BENCHMARK_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output-root "$OUTPUT_ROOT" \
  --split test \
  --sasa-threshold 5.0 \
  --threshold 0.3
