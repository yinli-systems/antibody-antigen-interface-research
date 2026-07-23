#!/bin/bash
#SBATCH --job-name=mipe-interactions
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
RUN_DIR=$DEPLOY_ROOT/results/mipe/formal_seed42
OUTPUT_DIR=$RUN_DIR/test_interactions
REPORT_DIR=$DEPLOY_ROOT/reports/mipe-interactions-$SLURM_JOB_ID

mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/mipe/bin:$PATH
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export TORCH_EXTENSIONS_DIR=$DEPLOY_ROOT/cache/torch_extensions

/usr/bin/time -v "$DEPLOY_ROOT/envs/mipe/bin/python" \
  "$DEPLOY_ROOT/jobs/mipe_export_interactions.py" \
  --source-root "$DEPLOY_ROOT/src/MIPE" \
  --data "$DEPLOY_ROOT/data/mipe/dataset/cvdata.pkl" \
  --checkpoint-dir "$RUN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --seed 42

sha256sum "$OUTPUT_DIR/summary.json" > "$REPORT_DIR/sha256.txt"
