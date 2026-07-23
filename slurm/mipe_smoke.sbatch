#!/bin/bash
#SBATCH --job-name=mipe-smoke
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
JOB_ROOT=$DEPLOY_ROOT/jobs
SOURCE_ROOT=$DEPLOY_ROOT/src/MIPE
PYTHON_BIN=$DEPLOY_ROOT/envs/mipe/bin/python
RUN_DIR=$DEPLOY_ROOT/reports/mipe-smoke-$SLURM_JOB_ID

mkdir -p "$DEPLOY_ROOT/logs" "$RUN_DIR" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export PATH=$DEPLOY_ROOT/envs/mipe/bin:$PATH

nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 2 > "$RUN_DIR/gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

echo "job_id=$SLURM_JOB_ID"
echo "node=$(hostname)"
date --iso-8601=seconds
nvidia-smi

/usr/bin/time -v "$PYTHON_BIN" "$JOB_ROOT/mipe_smoke.py" \
  --source-root "$SOURCE_ROOT" \
  --data "$DEPLOY_ROOT/data/mipe/dataset/cvdata.pkl" \
  --output-dir "$RUN_DIR" \
  --train-samples 2 \
  --val-samples 2

sha256sum "$RUN_DIR/mipe_smoke_checkpoint.pt" > "$RUN_DIR/sha256.txt"
date --iso-8601=seconds
