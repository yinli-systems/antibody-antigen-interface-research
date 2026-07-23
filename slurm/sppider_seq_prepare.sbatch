#!/bin/bash
#SBATCH --job-name=sppider-prepare
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
BENCHMARK_ROOT=$DEPLOY_ROOT/data/sabdab2-baseline-outsourcing
PREPARE_DIR=${SPPIDER_PREPARE_DIR:-$DEPLOY_ROOT/data/sppider_seq_sabdab2_prepared}
CANONICAL_MANIFEST=${SPPIDER_CANONICAL_MANIFEST:-}
VALIDATION_FRACTION=${SPPIDER_VALIDATION_FRACTION:-0.10}

CANONICAL_ARGS=()
if [[ -n "$CANONICAL_MANIFEST" ]]; then
  CANONICAL_ARGS=(--canonical-manifest "$CANONICAL_MANIFEST")
fi

mkdir -p "$PREPARE_DIR" "$DEPLOY_ROOT/cache/tmp" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp

/usr/bin/time -v "$PYTHON" "$DEPLOY_ROOT/jobs/sppider_seq/sabdab2_sppider_seq.py" prepare \
  --benchmark-root "$BENCHMARK_ROOT" \
  --output-dir "$PREPARE_DIR" \
  "${CANONICAL_ARGS[@]}" \
  --validation-fraction "$VALIDATION_FRACTION" \
  --seed 42 \
  --resume
