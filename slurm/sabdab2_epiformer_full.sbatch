#!/bin/bash
#SBATCH --job-name=epi-sabdab2
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%A_%a.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%A_%a.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
BENCHMARK_ROOT=$DEPLOY_ROOT/data/sabdab2-baseline-outsourcing
OUTPUT_ROOT=${SABDAB2_OUTPUT_ROOT:-$DEPLOY_ROOT/results/sabdab2/epiformer_formal}
CHECKPOINT=$DEPLOY_ROOT/data/results/epiformer/checkpoints/formal_epi_group_seed42/epiformer_best.pt
PYTHON=$DEPLOY_ROOT/envs/epiformer/bin/python
NUM_SHARDS=${SABDAB2_NUM_SHARDS:-3}

EXTRA_ARGS=()
if [[ -n ${SABDAB2_MAX_INSTANCES:-} ]]; then
  EXTRA_ARGS+=(--max-instances "$SABDAB2_MAX_INSTANCES")
fi
if [[ -n ${SABDAB2_INSTANCE_ID:-} ]]; then
  EXTRA_ARGS+=(--instance-id "$SABDAB2_INSTANCE_ID")
fi

mkdir -p "$OUTPUT_ROOT" "$DEPLOY_ROOT/cache/tmp" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export TORCH_HOME=$DEPLOY_ROOT/cache/torch
export HF_HOME=$DEPLOY_ROOT/cache/huggingface

/usr/bin/time -v "$PYTHON" "$DEPLOY_ROOT/jobs/sabdab2_epiformer_adapter.py" infer \
  --epiformer-code "$DEPLOY_ROOT/src/epiformer/code" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output-root "$OUTPUT_ROOT" \
  --split test \
  --sasa-threshold 5.0 \
  --device cuda:0 \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS" \
  "${EXTRA_ARGS[@]}"
