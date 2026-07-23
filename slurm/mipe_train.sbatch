#!/bin/bash
#SBATCH --job-name=mipe-train
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/mipe/bin/python
FOLD=${MIPE_FOLD:-0}
EPOCHS=${MIPE_EPOCHS:-1}
RUN_NAME=${MIPE_RUN_NAME:-formal_seed42}
RUN_DIR=$DEPLOY_ROOT/results/mipe/$RUN_NAME
REPORT_DIR=$DEPLOY_ROOT/reports/mipe-train-$SLURM_JOB_ID

mkdir -p "$RUN_DIR" "$REPORT_DIR" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/mipe/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export TORCH_EXTENSIONS_DIR=$DEPLOY_ROOT/cache/torch_extensions

nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 2 > "$REPORT_DIR/gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

/usr/bin/time -v "$PYTHON_BIN" "$DEPLOY_ROOT/jobs/mipe_train.py" \
  --source-root "$DEPLOY_ROOT/src/MIPE" \
  --data "$DEPLOY_ROOT/data/mipe/dataset/cvdata.pkl" \
  --output-dir "$RUN_DIR" \
  --fold "$FOLD" \
  --epochs "$EPOCHS" \
  --accumulation 16 \
  --seed 42

sha256sum "$RUN_DIR/fold_${FOLD}_last.pt" "$RUN_DIR/fold_${FOLD}_best.pt" > "$REPORT_DIR/sha256.txt"
