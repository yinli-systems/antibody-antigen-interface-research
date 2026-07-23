#!/bin/bash
#SBATCH --job-name=sppider-sim-audit
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON=$DEPLOY_ROOT/envs/epiformer/bin/python
BENCHMARK_ROOT=$DEPLOY_ROOT/data/sabdab2-baseline-outsourcing
MMSEQS=$DEPLOY_ROOT/tools/mmseqs2-latest/bin/mmseqs
WORK_DIR=$DEPLOY_ROOT/results/sabdab2/sppider_seq/research/mmseqs
OUTPUT=$DEPLOY_ROOT/results/sabdab2/sppider_seq/research/train_test_similarity.json

mkdir -p "$WORK_DIR" "$(dirname "$OUTPUT")" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH

/usr/bin/time -v "$PYTHON" "$DEPLOY_ROOT/jobs/sppider_seq/sppider_seq_research.py" mmseqs-audit \
  --benchmark-root "$BENCHMARK_ROOT" \
  --mmseqs "$MMSEQS" \
  --work-dir "$WORK_DIR" \
  --output "$OUTPUT" \
  --identity-thresholds 0.5 0.7 0.8 0.9 0.95 \
  --coverage-thresholds 0.5 0.8 0.9 \
  --sensitivity 7.5 \
  --evalue 1000 \
  --max-hits 10000 \
  --threads 8
