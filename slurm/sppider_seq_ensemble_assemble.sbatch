#!/bin/bash
#SBATCH --job-name=sppider-ensemble-final
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
PREPARE_DIR=${SPPIDER_PREPARE_DIR:?Set SPPIDER_PREPARE_DIR}
ROLE=${SPPIDER_ROLE:?Set SPPIDER_ROLE to paratope or epitope}
RUN_NAME=${SPPIDER_RUN_NAME:?Set SPPIDER_RUN_NAME}
INPUT_DIR_A=${SPPIDER_INPUT_DIR_A:?Set SPPIDER_INPUT_DIR_A}
INPUT_DIR_B=${SPPIDER_INPUT_DIR_B:?Set SPPIDER_INPUT_DIR_B}
THRESHOLD=${SPPIDER_THRESHOLD:?Set the frozen validation-selected threshold}
RUN_ROOT=$DEPLOY_ROOT/results/sabdab2/sppider_seq/$RUN_NAME/$ROLE

mkdir -p "$RUN_ROOT/inference" "$RUN_ROOT/final" "$DEPLOY_ROOT/cache/tmp" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp

"$PYTHON" "$DEPLOY_ROOT/jobs/sppider_seq/sabdab2_sppider_seq.py" ensemble-inference \
  --prepare-dir "$PREPARE_DIR" \
  --input-dirs "$INPUT_DIR_A" "$INPUT_DIR_B" \
  --weights 0.5 0.5 \
  --output-dir "$RUN_ROOT/inference" \
  --role "$ROLE"

/usr/bin/time -v "$PYTHON" "$DEPLOY_ROOT/jobs/sppider_seq/sabdab2_sppider_seq.py" assemble \
  --benchmark-root "$BENCHMARK_ROOT" \
  --prepare-dir "$PREPARE_DIR" \
  --inference-dir "$RUN_ROOT/inference" \
  --output-dir "$RUN_ROOT/final" \
  --role "$ROLE" \
  --threshold "$THRESHOLD"
