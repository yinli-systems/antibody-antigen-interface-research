#!/bin/bash
#SBATCH --job-name=sppider-assemble
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON=$DEPLOY_ROOT/envs/epiformer/bin/python
BENCHMARK_ROOT=$DEPLOY_ROOT/data/sabdab2-baseline-outsourcing
PREPARE_DIR=${SPPIDER_PREPARE_DIR:-$DEPLOY_ROOT/data/sppider_seq_sabdab2_prepared}
ROLE=${SPPIDER_ROLE:?Set SPPIDER_ROLE to paratope or epitope}
RUN_NAME=${SPPIDER_RUN_NAME:?Set SPPIDER_RUN_NAME to a stable result name}
RUN_ROOT=$DEPLOY_ROOT/results/sabdab2/sppider_seq/$RUN_NAME/$ROLE
INFERENCE_DIR=${SPPIDER_INFERENCE_DIR:-$RUN_ROOT/inference}
OUTPUT_DIR=${SPPIDER_ASSEMBLE_OUTPUT_DIR:-$RUN_ROOT/final}
THRESHOLD=${SPPIDER_THRESHOLD:-0.50}

mkdir -p "$OUTPUT_DIR" "$DEPLOY_ROOT/cache/tmp" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp

/usr/bin/time -v "$PYTHON" "$DEPLOY_ROOT/jobs/sppider_seq/sabdab2_sppider_seq.py" assemble \
  --benchmark-root "$BENCHMARK_ROOT" \
  --prepare-dir "$PREPARE_DIR" \
  --inference-dir "$INFERENCE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --role "$ROLE" \
  --threshold "$THRESHOLD"
